#!/usr/bin/env python3
"""Offline self-check for deterministic runner assertions.

Drives the real agent graph with a mocked LLM so the assertion and reporting
logic can be inspected without live model calls. Semantic judge checks are
marked unavailable rather than given fabricated pass results. The mock answers
use tool-backed values so deterministic numeric checks can run against genuine
tool output.
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
        tool = (case.get("expected_tools") or case.get("optional_tools") or [None])[0]
        if tool:
            responses.append(_assistant("", [_tool_call("c1", tool, TOOL_ARGS.get((case["id"], tool), {}))]))
        responses.append(_assistant(MOCK_ANSWERS.get(case["id"], "这是答案。")))
        responses.append(REVIEW_PASS)
        return router, responses
    tool = (case.get("expected_tools") or case.get("optional_tools") or [None])[0]
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
    "DQ05": (
        "YTD 盈亏 +92,088.92。股票/ETF 估算 +4,669.06 SGD，股票期权估算 +3,898.15 SGD，"
        "期货/期货期权/外汇（合并）是剩余项 +77,660.39 SGD，利息精确值 5,930.28 SGD，其他费用精确值 -68.96 SGD。"
        "估算与剩余项不是已实现盈亏，也不能证明具体交易因果。"
    ),
    "DQ06": "按自然年 TWR，最好 2023 年 +31.88%，最差 2020 年 -50.96%。",
    "DQ07": "利息 5,930.28，其他费用 -68.96，平均月末现金 592,820.54，隐含年化收益率约 1.29%。",
    "DQ08": "赢家首位 ESM5，+107,932.36；输家首位 SPY，-53,431.09。",
    "DQ09": "最大回撤 65.20%，峰值至谷底 2018-01 至 2020-03，恢复期 75 个月，夏普比率 0.0610。",
    "DQ10": "截至 2026-10-09，持有 SGD 与 USD 现金及若干 ETF，以上为市值口径，并非风险口径。",
    "DQ11": (
        "2025-03 TWR +4.49%，同月 SPXTR -5.63%，月度盈亏 18,935.70 SGD，当月入金 60,000.00 SGD，"
        "入金不是收益，已从盈亏与 TWR 中排除。PA 原始贡献为期货 -2.8159 pp、期货期权 +6.1316 pp，"
        "衍生品合并 +3.3157 pp；两类可能因行权发生分类转移，但月度数据不能证明本月发生行权。"
        "现金类贡献 +1.1393 pp，精确利息 406.20 SGD、费用 -33.99 SGD，未解释估算 4,470.38 SGD；"
        "未解释来源无法确认，不能仅凭无资金流归因汇率，也不能断言为利息或费用。类别贡献是 PA 口径，不等于已实现盈亏。"
    ),
    "DQ12": (
        "2025-04 TWR 为 -3.75%。PA 原始类别贡献（百分点）：期货 -12.2867 pp，期货期权 -16.3489 pp，"
        "合并衍生品 -28.6356 pp，现金 +10.7273 pp。类别合计 -17.91 pp 与 TWR 相差 -14.16 pp，"
        "说明这些类别值不能作为整体 TWR 的完整对账拆分，也不能据此估算各类 SGD 盈亏或确定整体亏损原因。"
        "期货与期货期权可能因行权发生类别转移，但这些月度数据不能证明本月发生行权或两类来自同一合约；"
        "类别贡献也不等于已实现交易盈亏。"
    ),
    "DQ13": (
        "现金类贡献（PA）+1.1393 pp，精确利息 406.20 SGD、费用 -33.99 SGD，未解释金额估算为 4,470.38 SGD。"
        "具体来源无法确认，可能包括汇率折算或其他未分类因素；仅凭没有入金出金不能断言是美元升值，"
        "也不能将未解释金额断言为利息或费用。"
    ),
    "DQ14": "无法回答月度成交笔数，报告没有逐笔成交时间（K8）。",
    "DQ15": "无法回答杠杆，报告没有保证金数据（K7）。",
    "PI03": "报告覆盖到 2026-10-09。",
    "PI01": "本系统可查询收益、归因、持仓等。",
    "PI02": "CFD 即差价合约，是一种差价合约工具。",
    "PI04": "TWR 是时间加权收益率。",
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
        router_events = [e for e in events if e.get("event") == "router_classified"]
        case_result = CaseResult(
            case_id=case["id"],
            name=case.get("name", case["id"]),
            intent=case["intent"],
            question=case["question"],
            answer=answer,
            actual_tools=[e.get("tool") for e in tool_events],
            tool_results=tool_results,
            review=review_events[-1] if review_events else {},
            semantic_review=(
                {"status": "unavailable", "error": "离线自检不运行语义评审"}
                if case.get("semantic_rubric") else {}
            ),
            actual_intent=router_events[-1].get("intent", "") if router_events else "",
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
