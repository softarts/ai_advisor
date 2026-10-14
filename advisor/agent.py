import json
import logging
import os
from pathlib import Path
import time
from datetime import datetime
from typing import TypedDict, Annotated
from zoneinfo import ZoneInfo

import httpx
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, ToolMessage

from .tools import execute


class State(TypedDict):
    messages: Annotated[list, add_messages]


class Advisor:
    def __init__(self, tools, logger, user_id="default", timezone_name=None, data_as_of=None):
        self.tools = {t.name: t for t in tools}
        self.logger = logger
        self.trace_id = None
        self.store = InMemoryStore()
        self.user_id = user_id
        self.timezone_name = timezone_name or os.environ.get("APP_TIMEZONE", "Asia/Singapore")
        self.data_as_of = data_as_of
        self.checkpointer = MemorySaver()
        graph = StateGraph(State)
        graph.add_node("llm", self._llm)
        graph.add_node("tools", self._tools)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", self._route, {"tools": "tools", "end": END})
        graph.add_edge("tools", "llm")
        self.graph = graph.compile(checkpointer=self.checkpointer, store=self.store)

    def _llm(self, state):
        memories = self.store.search((self.user_id,))
        memory_text = "\n".join(f"{item.key}: {item.value['data']}" for item in memories)
        now = datetime.now(ZoneInfo(self.timezone_name))
        as_of = self.data_as_of() if callable(self.data_as_of) else self.data_as_of
        time_context = (f"\n本轮当前日期时间：{now.isoformat(timespec='seconds')}（时区 {self.timezone_name}）。"
                        f"账户报告数据截止日：{as_of or '未知'}。当前日期与数据截止日不同，不能假设报告已更新到今天。")
        system = ("你是投资组合数据助手。账户相关问题必须用工具取得数据，不能自行计算工具可计算的数字。"
                  "月度收益率查询或排序使用 get_monthly_returns；资产价值区间查询使用 get_portfolio_detail；"
                  "用户询问“为什么某月收益高/低”或单月收益原因必须先调用 get_month_detail（参数为显式 YYYY-MM），不得用其他窗口（如 3Y、YTD 等）的宏观归因推断。"
                  "涉及报告不支持或缺数据的问题，先查询 get_capabilities，再说明缺少什么并拒答不可验证的结论。"
                  "使用工具返回的区间、币种、质量、warnings 和口径说明；自开户盈亏需结合资金流解读。"
                  "相对日期以本轮当前日期和时区为准，报告数据截止日仅表示数据覆盖范围。") + time_context
        policy_path = Path(__file__).resolve().parent.parent / "prompts" / "why_month_policy.md"
        if policy_path.exists():
            system += "\n\n" + policy_path.read_text(encoding="utf-8")
        if memory_text: system += "\n用户长期记忆：\n" + memory_text
        messages = [{"role": "system", "content": system}]
        for m in state["messages"]:
            role = "assistant" if m.type == "ai" else "user" if m.type == "human" else "tool"
            item = {"role": role, "content": m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)}
            if getattr(m, "tool_calls", None):
                item["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": json.dumps(t["args"])} } for t in m.tool_calls]
                reasoning = getattr(m, "additional_kwargs", {}).get("reasoning_content")
                if reasoning:
                    item["reasoning_content"] = reasoning
            if isinstance(m, ToolMessage): item["tool_call_id"] = m.tool_call_id
            messages.append(item)
        payload = {"model": os.environ["LLM_MODEL"], "messages": messages,
                   "tools": [t.api_schema() for t in self.tools.values()]}
        headers = {"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"}
        url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
        self.logger.info(json.dumps({"event":"llm_http_request","trace_id":self.trace_id,"url":url,"payload":payload}, ensure_ascii=False))
        response = None
        for attempt in range(2):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=120)
                self.logger.info(json.dumps({"event":"llm_http_response","trace_id":self.trace_id,"attempt":attempt + 1,"status":response.status_code,"body":response.text}, ensure_ascii=False))
                if response.status_code >= 500 and attempt == 0:
                    time.sleep(0.5)
                    continue
                response.raise_for_status()
                break
            except httpx.RequestError as exc:
                self.logger.warning(json.dumps({"event":"llm_http_retryable_error","trace_id":self.trace_id,"attempt":attempt + 1,"error":str(exc)}, ensure_ascii=False))
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise
        if response is None:
            raise RuntimeError("LLM request did not produce a response")
        msg = response.json()["choices"][0]["message"]
        calls = [{"id": c["id"], "name": c["function"]["name"], "args": json.loads(c["function"]["arguments"])} for c in msg.get("tool_calls", [])]
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=msg.get("content") or "", tool_calls=calls,
                                       additional_kwargs={"reasoning_content": msg["reasoning_content"]}
                                       if msg.get("reasoning_content") else {})]}

    def _route(self, state):
        return "tools" if state["messages"][-1].tool_calls else "end"

    def _tools(self, state):
        from langchain_core.messages import ToolMessage
        out=[]
        for call in state["messages"][-1].tool_calls:
            tool=self.tools[call["name"]]
            self.logger.info(json.dumps({"event":"tool_call","trace_id":self.trace_id,"tool":tool.name,"arguments":call["args"]},ensure_ascii=False))
            result=execute(tool,call["args"])
            self.logger.info(json.dumps({"event":"tool_result","trace_id":self.trace_id,"tool":tool.name,"result":result},ensure_ascii=False))
            out.append(ToolMessage(content=json.dumps(result,ensure_ascii=False),tool_call_id=call["id"]))
        return {"messages":out}

    def ask(self, text, thread_id, trace_id=None):
        self.trace_id = trace_id
        result=self.graph.invoke({"messages":[HumanMessage(content=text)]},{"configurable":{"thread_id":thread_id}})
        return result["messages"][-1].content

    def remember(self, key, value):
        self.store.put((self.user_id,), key, {"data": value})

    def recall(self):
        return {item.key: item.value.get("data") for item in self.store.search((self.user_id,))}
