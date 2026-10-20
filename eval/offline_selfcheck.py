#!/usr/bin/env python3
"""Offline self-check for the evaluation runner.

Drives the real agent graph with a mocked LLM so the assertion and reporting
logic can be validated without spending model calls. The mock answers with the
exact values the real tools return, which lets the numeric grounding check run
against genuine tool output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import os

os.environ.setdefault("LLM_MODEL", "offline-mock")
os.environ.setdefault("LLM_API_KEY", "offline-mock")
os.environ.setdefault("LLM_BASE_URL", "https://offline.invalid")

from advisor.agent import Advisor
from advisor.tools import create_tools
from eval.report import CaseResult, evaluate_case, summarize
from eval.runner import EventCollector, load_cases
from pa_analytics import Store

INTENT_BY_ID = {}


class MockResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _assistant(content, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _tool_call(call_id, name, args):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class ScriptedLLM:
    """Replays a scripted sequence of chat-completions responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        if self.calls >= len(self.script):
            raise AssertionError("mock LLM ran out of scripted responses")
        response = self.script[self.calls]
        self.calls += 1
        return MockResponse(response)


REVIEW_PASS = _assistant(json.dumps({"decision": "pass", "reason": "符合策略。"}))


def build_mocks(case, tools_by_name):
    """Return (router_payload, llm_responses) for a case."""
    intent = case["intent"]
    if intent == "unsupported":
        return (
            _assistant(json.dumps({"intent": "unsupported", "reason": "超出范围"})),
            [],
        )
    router = _assistant(json.dumps({"intent": intent, "reason": "测试"}))
    if intent == "product_info":
        responses = []
        tool = (case.get("expected_tools") or [None])[0]
        if tool:
            responses.append(_assistant("", [_tool_call("c1", tool, TOOL_ARGS.get((case["id"], tool), {}))]))
        responses.append(_assistant(MOCK_ANSWERS.get(case["id"], "这是答案。")))
        responses.append(REVIEW_PASS)
        return router, responses
    tool = (case.get("expected_tools") or [None])[0]
    if tool is None:
        return router, [_assistant("这是答案。")]
    args = TOOL_ARGS.get((case["id"], tool), {})
    return router, [
        _assistant("", [_tool_call("c1", tool, args)]),
        _assistant(MOCK_ANSWERS.get(case["id"], "答案包含 1.00%。")),
    ]


TOOL_ARGS = {
    ("DQ01", "get_monthly_returns"): {"window": "3Y"},
    ("DQ02", "get_portfolio_detail"): {"start_month": "2026-09", "end_month": "2026-10"},
    ("DQ03", "get_window_summary"): {"window": "1Y"},
    ("DQ04", "get_window_summary"): {"window": "YTD"},
    ("DQ05", "get_attribution"): {"window": "YTD"},
    ("DQ06", "get_yearly_table"): {},
    ("DQ07", "get_cash_interest"): {"window": "YTD"},
    ("DQ08", "get_top_contributors"): {"n": 3},
    ("DQ09", "get_risk_measures"): {},
    ("DQ10", "get_open_positions"): {},
    ("DQ11", "get_month_detail"): {"month": "2025-03"},
    ("DQ12", "get_month_detail"): {"month": "2025-04"},
    ("DQ13", "get_month_detail"): {"month": "2025-03"},
    ("DQ14", "get_capabilities"): {},
    ("DQ15", "get_capabilities"): {},
    ("PI03", "get_capabilities"): {},
}

MOCK_ANSWERS = {
    "DQ01": "最高是 2025-03，账户月度 TWR +4.49%；最低是 2025-05，为 -4.02%。",
    "DQ02": "2026-09 月末资产 707,216.54；2026-10 报告截至 10 月 9 日，为 713,223.18。",
    "DQ03": "期初 532,216.57，期末 713,223.18，资产变化 +181,006.61；净流入 77,000.42，区间投资盈亏 104,006.19，TWR 19.35%。基准 SPXTR 15.50%、EFA 13.17%、VT 15.57%。",
    "DQ04": "2026 YTD TWR 16.73%，SPXTR 15.14%，领先 1.58 个百分点。",
    "DQ05": "YTD 盈亏 +92,088.92。股票/ETF +4,669.06，股票期权 +3,898.15，期货/期货期权/外汇合并 +77,660.39，利息 5,930.28，其他费用 -68.96。",
    "DQ06": "按自然年 TWR，最好 2023 年 +31.88%，最差 2018 年 -28.16%。",
    "DQ07": "利息 5,930.28，其他费用 -68.96，平均月末现金 592,820.54，隐含年化收益率约 1.29%。",
    "DQ08": "赢家首位 ESM5，+107,932.36；输家首位 SPY，-53,431.09。",
    "DQ09": "最大回撤 65.20%，峰值至谷底 2018-01 至 2020-03，恢复期 75 个月，夏普比率 0.0610。",
    "DQ10": "截至 2026-10-09，持有 SGD 与 USD 现金及若干 ETF。",
    "DQ11": "2025-03 TWR +4.49%，同月 SPXTR -5.63%，月度盈亏 18,935.70 SGD，当月入金 60,000.00 SGD，入金不计入收益。期货与期货期权合并贡献约 +3.32 个百分点（K3），现金类未解释 4,470.38 SGD（K13）。",
    "DQ12": "2025-04 TWR -3.75%，各品种贡献与 TWR 不自洽，触发门禁，不提供品种拆分（K3）。",
    "DQ13": "现金类未解释 4,470.38 SGD，来源无法验证，不能理解为利息、费用或外汇（K13）。",
    "DQ14": "无法回答月度成交笔数，报告没有逐笔成交时间（K8）。",
    "DQ15": "无法回答杠杆，报告没有保证金数据（K7）。",
    "PI03": "报告覆盖到 2026-10-09。",
    "PI01": "本系统可查询收益、归因、持仓等。",
    "PI02": "CFD 即差价合约，是一种差价合约工具。",
    "PI04": "TWR 是时间加权收益率。",
    "UN01": "很抱歉，您的问题超出了本系统的服务范围。本助手专注于 IBKR PortfolioAnalyst 报告数据的查询与解读。",
    "UN02": "很抱歉，您的问题超出了本系统的服务范围。本助手专注于 IBKR PortfolioAnalyst 报告数据的查询与解读。",
    "UN03": "很抱歉，您的问题超出了本系统的服务范围。本助手专注于 IBKR PortfolioAnalyst 报告数据的查询与解读。",
}


def run_offline(case_ids=None):
    cases = load_cases(case_ids)
    store = Store("data/pa.db", "data/raw")
    tools = create_tools(store)
    logger = logging.getLogger("eval-offline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    collector = EventCollector()
    logger.addHandler(collector)

    results = []
    for case in cases:
        router_payload, llm_responses = build_mocks(case, None)
        advisor = Advisor(tools, logger, data_as_of=lambda: store.current().period_end)
        scripted = ScriptedLLM([router_payload] + llm_responses)
        import advisor.agent as agent_module
        import advisor.router as router_module

        def fake_post(url, headers=None, json=None, timeout=None, _scripted=scripted):
            return _scripted(url, headers=headers, json=json, timeout=timeout)

        trace_id = "offline-trace"
        with patch.object(router_module.httpx, "post", fake_post), \
             patch.object(agent_module.httpx, "post", fake_post):
            advisor.trace_id = trace_id
            from langchain_core.messages import HumanMessage
            result = advisor.graph.invoke(
                {"messages": [HumanMessage(content=case["question"])], "intent": ""},
                {"configurable": {"thread_id": f"offline-{case['id']}"}},
            )
            answer = result["messages"][-1].content

        events = [e for e in collector.events if e.get("trace_id") == trace_id]
        collector.events = [e for e in collector.events if e.get("trace_id") != trace_id]
        tool_events = [e for e in events if e.get("event") == "tool_call"]
        tool_results = [
            {"tool": e.get("tool", "tool"), "content": json.dumps(e.get("result"), ensure_ascii=False)}
            for e in events if e.get("event") == "tool_result"
        ]
        review_events = [e for e in events if e.get("event") == "output_review_summary"]
        case_result = CaseResult(
            case_id=case["id"],
            name=case.get("name", case["id"]),
            intent=case["intent"],
            question=case["question"],
            answer=answer,
            actual_tools=[e.get("tool") for e in tool_events],
            tool_results=tool_results,
            review=review_events[-1] if review_events else {},
            notes=[l.strip() for l in answer.splitlines() if l.strip().startswith("⚠️")],
        )
        results.append(evaluate_case(case, case_result))

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id} {result.name}")
        for failure in result.failures:
            print(f"    ✗ {failure}")
    summary = summarize(results)
    print(f"\n通过 {summary['passed']}/{summary['total']} ({summary['pass_rate']}%)")
    return results, summary


if __name__ == "__main__":
    args = sys.argv[1:]
    run_offline(args or None)
