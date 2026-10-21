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
from .checklist_config import GLOBAL_CHECKS, GLOBAL_CHECKLIST, INTENT_CHECKLISTS
from .llm_settings import MAX_COMPLETION_TOKENS
from .output_review import CHECK_PLUGINS, CheckContext
from .router import (
    INTENT_PROFILES,
    classify_intent,
    load_compose_template,
    load_output_review_config,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str   # "data_query" | "product_info" | "unsupported" | ""


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
        graph.add_node("router", self._router)
        graph.add_node("llm", self._llm)
        graph.add_node("tools", self._tools)
        graph.add_node("output_review", self._output_review)
        graph.add_node("fixed_reply", self._fixed_reply)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._route_intent,
            {"llm": "llm", "fixed_reply": "fixed_reply"},
        )
        graph.add_conditional_edges("llm", self._route, {"tools": "tools", "review": "output_review"})
        graph.add_edge("tools", "llm")
        graph.add_edge("output_review", END)
        graph.add_edge("fixed_reply", END)
        self.graph = graph.compile(checkpointer=self.checkpointer, store=self.store)

    # ------------------------------------------------------------------
    # Router node
    # ------------------------------------------------------------------

    def _router(self, state):
        last_human = next(
            (m for m in reversed(state["messages"]) if m.type == "human"), None
        )
        user_text = last_human.content if last_human else ""
        intent = classify_intent(user_text, self.logger, self.trace_id)
        return {"intent": intent}

    def _route_intent(self, state):
        return "fixed_reply" if state.get("intent") == "unsupported" else "llm"

    # ------------------------------------------------------------------
    # Fixed reply node (unsupported intent — no LLM call)
    # ------------------------------------------------------------------

    def _fixed_reply(self, state):
        from langchain_core.messages import AIMessage
        template = load_compose_template("unsupported")
        # Extract only the reply text (lines after the YAML front-matter and heading)
        lines = template.splitlines()
        reply_lines = []
        in_front_matter = False
        past_front_matter = False
        for line in lines:
            if line.strip() == "---":
                if not past_front_matter:
                    in_front_matter = not in_front_matter
                    if not in_front_matter:
                        past_front_matter = True
                continue
            if not past_front_matter:
                continue
            # Skip heading lines before body
            if not reply_lines:
                if line.startswith("#") or not line.strip():
                    continue
            reply_lines.append(line)
        reply = "\n".join(reply_lines).strip()
        if not reply:
            reply = "很抱歉，您的问题超出了本系统的服务范围。"
        return {"messages": [AIMessage(content=reply)]}

    # ------------------------------------------------------------------
    # LLM node
    # ------------------------------------------------------------------

    def _llm(self, state):
        intent = state.get("intent") or "data_query"
        profile = INTENT_PROFILES.get(intent, INTENT_PROFILES["data_query"])
        allowed_tool_names = set(profile["allowed_tools"])

        memories = self.store.search((self.user_id,))
        memory_text = "\n".join(f"{item.key}: {item.value['data']}" for item in memories)
        now = datetime.now(ZoneInfo(self.timezone_name))
        as_of = self.data_as_of() if callable(self.data_as_of) else self.data_as_of
        time_context = (f"\n本轮当前日期时间：{now.isoformat(timespec='seconds')}（时区 {self.timezone_name}）。"
                        f"账户报告数据截止日：{as_of or '未知'}。当前日期与数据截止日不同，不能假设报告已更新到今天。")

        compose = load_compose_template(intent)
        if compose:
            system = compose + time_context
        else:
            # Fallback base prompt if template is missing
            system = ("你是投资组合数据助手。账户相关问题必须用工具取得数据，不能自行计算工具可计算的数字。") + time_context

        if memory_text:
            system += "\n用户长期记忆：\n" + memory_text

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

        # Only pass tools that are permitted for this intent
        allowed_tools = [t for name, t in self.tools.items() if name in allowed_tool_names]
        payload = {"model": os.environ["LLM_MODEL"], "messages": messages,
                   "max_tokens": MAX_COMPLETION_TOKENS}
        if allowed_tools:
            payload["tools"] = [t.api_schema() for t in allowed_tools]

        headers = {"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"}
        url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
        self.logger.info(json.dumps({"event": "llm_http_request", "trace_id": self.trace_id, "url": url, "intent": intent, "payload": payload}, ensure_ascii=False))
        response = None
        for attempt in range(2):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=120)
                self.logger.info(json.dumps({"event": "llm_http_response", "trace_id": self.trace_id, "attempt": attempt + 1, "status": response.status_code, "body": response.text}, ensure_ascii=False))
                if response.status_code >= 500 and attempt == 0:
                    time.sleep(0.5)
                    continue
                response.raise_for_status()
                break
            except httpx.RequestError as exc:
                self.logger.warning(json.dumps({"event": "llm_http_retryable_error", "trace_id": self.trace_id, "attempt": attempt + 1, "error": str(exc)}, ensure_ascii=False))
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                # Surface the server's own explanation: a 500 from the model
                # gateway is usually caused by a request-shape problem, and
                # the response body says which.
                self.logger.error(json.dumps({
                    "event": "llm_http_failed",
                    "trace_id": self.trace_id,
                    "intent": intent,
                    "status": exc.response.status_code,
                    "body": exc.response.text[:2000],
                    "error": str(exc),
                }, ensure_ascii=False))
                raise
        if response is None:
            raise RuntimeError("LLM request did not produce a response")
        msg = response.json()["choices"][0]["message"]
        calls = [{
            "id": c["id"], "name": c["function"]["name"],
            "args": json.loads(c["function"]["arguments"])
        } for c in msg.get("tool_calls", [])]
        # Filter tool calls to only allowed tools (safety guard)
        calls = [c for c in calls if c["name"] in allowed_tool_names]
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=msg.get("content") or "", tool_calls=calls,
                                       additional_kwargs={"reasoning_content": msg["reasoning_content"]}
                                       if msg.get("reasoning_content") else {})]}

    def _route(self, state):
        return "tools" if state["messages"][-1].tool_calls else "review"

    def _output_review(self, state):
        from langchain_core.messages import AIMessage

        intent = state.get("intent") or ""
        user_message = next(
            (message for message in reversed(state["messages"]) if message.type == "human"),
            None,
        )
        candidate = state["messages"][-1].content
        tool_results = [
            {"tool": getattr(message, "name", None) or "tool", "content": message.content}
            for message in state["messages"]
            if message.type == "tool"
        ]
        if user_message is None or not isinstance(candidate, str) or not candidate.strip():
            self.logger.error(json.dumps({
                "event": "output_review_unavailable_input",
                "trace_id": self.trace_id,
                "intent": intent,
            }, ensure_ascii=False))
            notice = "⚠️ 答案检查输入不完整，数据准确性无法确认。"
            safe_candidate = candidate if isinstance(candidate, str) else ""
            self.logger.info(json.dumps({
                "event": "output_review_summary",
                "trace_id": self.trace_id,
                "intent": intent,
                "checks_run": [],
                "failed_checks": [{"check_id": "checklist", "reason": "output review input is incomplete"}],
                "notes": [notice],
                "annotated": True,
                "passed": False,
            }, ensure_ascii=False))
            return {"messages": [AIMessage(content=f"{safe_candidate}\n\n{notice}".strip())]}

        review_config = load_output_review_config(intent)
        context = CheckContext(
            intent=intent,
            question=user_message.content,
            candidate=candidate,
            tool_results=tool_results,
            logger=self.logger,
            trace_id=self.trace_id,
            review_config=review_config,
        )
        failed_checks = []
        unavailable_checks = []
        checks_run: list[str] = []
        if intent not in INTENT_CHECKLISTS:
            self.logger.error(json.dumps({
                "event": "output_review_configuration_error",
                "trace_id": self.trace_id,
                "intent": intent,
                "error": "intent has no configured checklist",
            }, ensure_ascii=False))
            unavailable_checks.append(("checklist", "答案检查配置缺失，数据准确性无法确认。", "intent has no configured checklist"))
        for check_id in dict.fromkeys(
            GLOBAL_CHECKLIST + INTENT_CHECKLISTS.get(intent, [])
        ):
            definition = GLOBAL_CHECKS.get(check_id)
            checker = CHECK_PLUGINS.get(check_id)
            if definition is None or checker is None:
                error = f"unknown or unregistered check: {check_id}"
                self.logger.error(json.dumps({
                    "event": "output_review_configuration_error",
                    "trace_id": self.trace_id,
                    "intent": intent,
                    "check_id": check_id,
                    "error": error,
                }, ensure_ascii=False))
                unavailable_checks.append((check_id, definition["unavailable_note"] if definition else error, error))
                continue
            if not definition["enabled"]:
                continue
            try:
                outcome = checker(context)
                if outcome.candidate is not None:
                    context.candidate = outcome.candidate
                checks_run.append(check_id)
                if not outcome.passed:
                    failed_checks.append((check_id, outcome.note or definition["failure_note"], outcome.reason))
                self.logger.info(json.dumps({
                    "event": "output_review_check",
                    "trace_id": self.trace_id,
                    "intent": intent,
                    "check_id": check_id,
                    "passed": outcome.passed,
                    "reason": outcome.reason,
                }, ensure_ascii=False))
            except Exception as exc:
                self.logger.error(json.dumps({
                    "event": "output_review_check_error",
                    "trace_id": self.trace_id,
                    "intent": intent,
                    "check_id": check_id,
                    "error": str(exc),
                }, ensure_ascii=False))
                unavailable_checks.append((check_id, definition["unavailable_note"], str(exc)))

        notes = [note for _, note, _ in failed_checks + unavailable_checks if note]
        if notes:
            context.candidate += "\n\n" + "\n".join(dict.fromkeys(f"⚠️ {note}" for note in notes))
        self.logger.info(json.dumps({
            "event": "output_review_summary",
            "trace_id": self.trace_id,
            "intent": intent,
            "checks_run": checks_run,
            "failed_checks": [
                {"check_id": check_id, "reason": reason}
                for check_id, _, reason in failed_checks
            ],
            "unavailable_checks": [
                {"check_id": check_id, "reason": reason}
                for check_id, _, reason in unavailable_checks
            ],
            "notes": notes,
            "annotated": context.candidate != candidate,
            "passed": not failed_checks and not unavailable_checks,
        }, ensure_ascii=False))
        if not failed_checks and not unavailable_checks and context.candidate == candidate:
            return {}
        return {"messages": [AIMessage(content=context.candidate)]}

    def _tools(self, state):
        from langchain_core.messages import ToolMessage
        out = []
        for call in state["messages"][-1].tool_calls:
            tool = self.tools[call["name"]]
            self.logger.info(json.dumps({"event": "tool_call", "trace_id": self.trace_id, "tool": tool.name, "arguments": call["args"]}, ensure_ascii=False))
            result = execute(tool, call["args"])
            self.logger.info(json.dumps({"event": "tool_result", "trace_id": self.trace_id, "tool": tool.name, "result": result}, ensure_ascii=False))
            out.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                   tool_call_id=call["id"], name=tool.name))
        return {"messages": out}

    def ask(self, text, thread_id, trace_id=None):
        self.trace_id = trace_id
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=text)], "intent": ""},
            {"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content

    def remember(self, key, value):
        self.store.put((self.user_id,), key, {"data": value})

    def recall(self):
        return {item.key: item.value.get("data") for item in self.store.search((self.user_id,))}
