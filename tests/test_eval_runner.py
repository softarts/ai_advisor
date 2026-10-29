#!/usr/bin/env python3
"""Tests for the evaluation runner's wiring and assertions.

These use a mocked LLM so the runner logic (case loading, event capture,
tool assertion, content assertion, review assertion and reporting) can be
verified without model calls.
"""
import io
import json
import logging
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.example")

from advisor.agent import Advisor
from advisor.tools import create_tools
from eval.report import CaseResult, evaluate_case as _evaluate_case, summarize
from eval.runner import (
    EventCollector,
    load_cases,
    _extract_notes,
    _llm_failure_detail,
    print_case_answer,
)
from pa_analytics import Store


def evaluate_case(case, result):
    """Mock the separately exercised route and semantic-review layers."""
    if not result.actual_intent:
        result.actual_intent = case.get("intent", result.intent)
    rubric = case.get("semantic_rubric") or []
    if rubric and not result.semantic_review:
        result.semantic_review = {
            "status": "completed",
            "results": [
                {
                    "id": item["id"],
                    "decision": "pass",
                    "evidence": result.answer[:120],
                    "reason": "mock",
                }
                for item in rubric
            ],
        }
    return _evaluate_case(case, result)


def _tool_result(tool_name, payload):
    return {"tool": tool_name, "content": json.dumps(payload, ensure_ascii=False)}


def _dq12_tool_result(contributions=None):
    return _tool_result("get_month_detail", {
        "return": {"twr_pct": -3.75},
        "contribution_pp": {
            "consistency": "inconsistent",
            "etfs": 0.0,
            "stocks": 0.0,
            "options": 0.0,
            "futures": -12.286702847,
            "futures_options": -16.348891819,
            "future_style_options": 0.0,
            "cash": 10.727310563,
            "derivatives_merged": -28.635594666,
            "sum": -17.908284103,
            "twr_gap_pp": -14.16149678,
            **(contributions or {}),
        },
    })


def _dq12_answer():
    return (
        "| PA 类别 | 贡献（百分点） |\n"
        "|--------|----------------|\n"
        "| 期货 | **-12.29 pp [1]** |\n"
        "| 期货期权 | **-16.35 pp [2]** |\n"
        "| 现金 | **+10.73 pp [3]** |\n"
        "| 合并衍生品 | **-28.64 pp [4]** |\n"
        "| 类别合计 | **-17.91 pp [5]** |\n"
        "TWR 为 -3.75% [6]；"
        "与 TWR 的差值为 -14.16 pp [7]。\n"
        "类别合计与 TWR 不一致，说明这些类别值不能作为整体 TWR 的完整对账拆分，"
        "也不能据此估算各类 SGD 盈亏或确定整体亏损原因。期货与期货期权可能因行权发生类别转移，"
        "但这些月度数据不能证明本月发生行权或两类来自同一合约；类别贡献也不等于已实现交易盈亏。\n\n"
        "数字来源：\n"
        "[1] get_month_detail$.contribution_pp.futures\n"
        "[2] get_month_detail$.contribution_pp.futures_options\n"
        "[3] get_month_detail$.contribution_pp.cash\n"
        "[4] get_month_detail$.contribution_pp.derivatives_merged\n"
        "[5] get_month_detail$.contribution_pp.sum\n"
        "[6] get_month_detail$.return.twr_pct\n"
        "[7] get_month_detail$.contribution_pp.twr_gap_pp"
    )


class CaseLoadingTests(unittest.TestCase):
    def test_all_case_files_load(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 20)
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)), "case ids must be unique")

    def test_every_case_declares_its_intent(self):
        for case in load_cases():
            self.assertIn(case["intent"], {"data_query", "product_info", "unsupported"})

    def test_case_filter_selects_single_case(self):
        cases = load_cases(["DQ01"])
        self.assertEqual([case["id"] for case in cases], ["DQ01"])

    def test_answer_prompts_follow_the_user_question_language(self):
        for prompt_name in ("data_query.compose.md", "product_info.compose.md"):
            with self.subTest(prompt=prompt_name):
                prompt = (ROOT / "prompts" / prompt_name).read_text(encoding="utf-8")
                self.assertIn("使用用户本轮问题的语言作答", prompt)

    def test_data_query_prompt_prioritizes_concise_explanation_when_attribution_is_unavailable(self):
        prompt = (ROOT / "prompts" / "data_query.compose.md").read_text(encoding="utf-8")
        self.assertIn("仍可忠实报告 PA 原始类别贡献", prompt)
        self.assertIn("不能作为整体 TWR 的完整对账拆分", prompt)


class AssertionTests(unittest.TestCase):
    def _result(self, **kwargs):
        defaults = dict(
            case_id="X1",
            name="case",
            intent="data_query",
            question="q",
            answer="",
            actual_tools=[],
            tool_results=[],
            review={},
            failures=[],
            notes=[],
        )
        defaults.update(kwargs)
        return CaseResult(**defaults)

    def test_expected_tool_must_be_called(self):
        case = {"expected_tools": ["get_month_detail"]}
        result = self._result(actual_tools=[])
        evaluate_case(case, result)
        self.assertTrue(any("未调用期望工具" in f for f in result.failures))

    def test_optional_tool_may_be_used(self):
        case = {"expected_tools": [], "optional_tools": ["get_capabilities"]}
        result = self._result(actual_tools=["get_capabilities"])
        evaluate_case(case, result)
        self.assertFalse(any("未期望的工具" in f for f in result.failures))

    def test_optional_tool_may_be_omitted(self):
        case = {"expected_tools": [], "optional_tools": ["get_capabilities"]}
        result = self._result(actual_tools=[])
        evaluate_case(case, result)
        self.assertFalse(any("未期望的工具" in f for f in result.failures))

    def test_unexpected_tool_is_reported(self):
        case = {"expected_tools": ["get_month_detail"]}
        result = self._result(actual_tools=["get_month_detail", "get_capabilities"])
        evaluate_case(case, result)
        self.assertTrue(any("未期望的工具" in f for f in result.failures))

    def test_numeric_expectation_matches_across_formatting(self):
        case = {"must_include": ["18,935"], "expected_tools": [], "check_numbers_grounded": False}
        result = self._result(answer="月度盈亏 18935 SGD")
        evaluate_case(case, result)
        self.assertEqual(result.failures, [])

    def test_forbidden_content_is_reported(self):
        case = {"must_not": ["因为美元升值"], "expected_tools": [], "check_numbers_grounded": False}
        result = self._result(answer="这是因为美元升值导致的。")
        evaluate_case(case, result)
        self.assertTrue(any("禁止出现" in f for f in result.failures))

    def test_interchangeable_phrasings_are_accepted(self):
        case = {
            "must_include": [["入金不是收益", "入金不计入收益"]],
            "expected_tools": [],
            "check_numbers_grounded": False,
        }
        for phrasing in ("入金不是收益，已从盈亏中排除", "入金不计入收益"):
            with self.subTest(phrasing=phrasing):
                result = self._result(answer=phrasing)
                evaluate_case(case, result)
                self.assertEqual(result.failures, [])

    def test_dq12_accepts_a_plain_language_explanation_of_attribution_mismatch(self):
        case = load_cases(["DQ12"])[0]
        result = self._result(
            case_id="DQ12",
            name=case["name"],
            intent=case["intent"],
            question=case["question"],
            answer=_dq12_answer(),
            actual_tools=["get_month_detail"],
            tool_results=[_dq12_tool_result()],
            review={"passed": True, "failed_checks": [], "unavailable_checks": []},
        )
        evaluate_case(case, result)
        self.assertEqual(result.failures, [])
        self.assertTrue(all(
            assertion.get("require_source_path")
            for assertion in case["tool_value_assertions"]
        ))
        self.assertTrue(all(
            "label" not in assertion and "labels" not in assertion
            for assertion in case["tool_value_assertions"]
        ))

    def test_dq12_accepts_the_user_reported_answer_phrasing(self):
        case = load_cases(["DQ12"])[0]
        result = self._result(
            case_id="DQ12",
            name=case["name"],
            intent=case["intent"],
            question=case["question"],
            answer=_dq12_answer(),
            actual_tools=["get_month_detail"],
            tool_results=[_dq12_tool_result()],
            review={"passed": True, "failed_checks": [], "unavailable_checks": []},
        )
        evaluate_case(case, result)
        self.assertFalse(
            any("缺少必须包含的内容" in failure for failure in result.failures),
            result.failures,
        )

    def test_dq12_rejects_unsupported_market_causality_but_accepts_actual_boundary_wording(self):
        case = load_cases(["DQ12"])[0]
        result = self._result(
            case_id="DQ12",
            name=case["name"],
            intent=case["intent"],
            question=case["question"],
            answer=(
                _dq12_answer()
                + "亏损主要是由于市场整体回调加上账户内部组合的表现略逊于基准 SPXTR。"
            ),
            actual_tools=["get_month_detail"],
            tool_results=[_dq12_tool_result()],
            review={"passed": True, "failed_checks": [], "unavailable_checks": []},
            semantic_review={
                "status": "completed",
                "results": [
                    {
                        "id": "identify-attribution-inconsistency",
                        "decision": "pass",
                        "reason": "说明归因不一致且无法据此确定具体原因。",
                        "evidence": "品类贡献总计与 TWR 不一致。",
                    },
                    {
                        "id": "avoid-unsupported-loss-causality",
                        "decision": "fail",
                        "reason": "将亏损归因于市场回调和组合表现。",
                        "evidence": (
                            "亏损主要是由于市场整体回调加上账户内部组合的表现略逊于基准 SPXTR。"
                        ),
                    },
                    {
                        "id": "explain-derivative-classification",
                        "decision": "pass",
                        "reason": "说明两类之间可能有行权造成的分类转移。",
                        "evidence": "期货与期货期权可能因行权发生类别转移",
                    },
                ],
            },
        )
        evaluate_case(case, result)
        self.assertFalse(
            any("缺少必须包含的内容" in failure for failure in result.failures),
            result.failures,
        )
        self.assertTrue(
            any("语义评审 fail：avoid-unsupported-loss-causality" in failure for failure in result.failures),
            result.failures,
        )

    def test_dq12_accepts_original_category_contributions_with_reconciliation_caveat(self):
        case = load_cases(["DQ12"])[0]
        result = self._result(
            case_id="DQ12",
            name=case["name"],
            intent=case["intent"],
            question=case["question"],
            answer=_dq12_answer(),
            actual_tools=["get_month_detail"],
            tool_results=[_dq12_tool_result()],
            review={"passed": True, "failed_checks": [], "unavailable_checks": []},
        )
        evaluate_case(case, result)
        self.assertEqual(result.failures, [])

    def test_dq12_rejects_missing_number_source_pair(self):
        case = load_cases(["DQ12"])[0]
        result = self._result(
            case_id="DQ12",
            name=case["name"],
            intent=case["intent"],
            question=case["question"],
            answer=_dq12_answer().replace(
                "get_month_detail$.contribution_pp.futures",
                "get_month_detail$.contribution_pp.stocks",
            ),
            actual_tools=["get_month_detail"],
            tool_results=[_dq12_tool_result()],
            review={"passed": True, "failed_checks": [], "unavailable_checks": []},
        )
        evaluate_case(case, result)
        self.assertTrue(any("没有与工具 get_month_detail 字段 contribution_pp.futures 对应的来源索引" in failure for failure in result.failures))

    def test_forbidden_phrase_inside_a_negation_is_not_reported(self):
        case = {
            "must_not": ["因为美元升值"],
            "expected_tools": [],
            "check_numbers_grounded": False,
        }
        answer = "不能断言这 1.14 % 是因为美元升值（或利息、外汇变动）。"
        result = self._result(answer=answer)
        evaluate_case(case, result)
        self.assertEqual(result.failures, [])

    def test_forbidden_phrase_in_an_assertion_is_reported(self):
        case = {
            "must_not": ["因为美元升值"],
            "expected_tools": [],
            "check_numbers_grounded": False,
        }
        result = self._result(answer="本期收益确实因为美元升值而上升。")
        evaluate_case(case, result)
        self.assertTrue(any("禁止出现" in f for f in result.failures))

    def test_forbidden_phrase_in_a_question_heading_is_not_reported(self):
        case = {
            "must_not": ["因为美元升值"],
            "expected_tools": [],
            "check_numbers_grounded": False,
        }
        answer = "### 是否因为美元升值\n\n现金类 1.14 pp，不可断言为汇率变动。"
        result = self._result(answer=answer)
        evaluate_case(case, result)
        self.assertFalse(any("禁止出现" in f for f in result.failures))

    def test_date_in_an_expectation_does_not_become_a_negative_number(self):
        """A date must not yield a stray negative that matches the wrong figure."""
        case = {
            "must_not": ["最高的是 2022-10"],
            "expected_tools": [],
            "check_numbers_grounded": False,
        }
        result = self._result(answer="最高是 2025-03，+4.49%，区间 -10.0。")
        evaluate_case(case, result)
        self.assertFalse(any("禁止出现" in f for f in result.failures))

    def test_review_failure_is_reported_when_not_expected(self):
        case = {"expected_tools": [], "check_numbers_grounded": False, "expect_note": False}
        result = self._result(
            answer="答案。",
            review={"failed_checks": [{"check_id": "numeric_provenance", "reason": "无依据"}]},
        )
        evaluate_case(case, result)
        self.assertTrue(any("检查清单判定失败" in f for f in result.failures))

    def test_review_failure_is_expected_when_expect_note_is_set(self):
        case = {"expected_tools": [], "check_numbers_grounded": False, "expect_note": True}
        result = self._result(
            answer="答案。\n\n⚠️ 数字校验未通过。",
            review={"failed_checks": [{"check_id": "numeric_provenance", "reason": "无依据"}]},
            notes=["⚠️ 数字校验未通过。"],
        )
        evaluate_case(case, result)
        self.assertEqual(result.failures, [])

    def test_missing_note_is_reported_when_expected(self):
        case = {"expected_tools": [], "check_numbers_grounded": False, "expect_note": True}
        result = self._result(
            answer="答案。",
            review={"failed_checks": [{"check_id": "numeric_provenance", "reason": "无依据"}]},
        )
        evaluate_case(case, result)
        self.assertTrue(any("缺少 note 注解" in f for f in result.failures))

    def test_unavailable_check_is_reported_separately(self):
        case = {"expected_tools": [], "check_numbers_grounded": False, "expect_note": False}
        result = self._result(
            answer="答案。",
            review={"unavailable_checks": [{"check_id": "product_policy", "reason": "审核不可用"}]},
        )
        evaluate_case(case, result)
        self.assertTrue(any("检查清单无法执行" in f for f in result.failures))

    def test_ungrounded_number_is_reported(self):
        case = {"expected_tools": [], "check_numbers_grounded": True}
        result = self._result(
            answer="收益是 99.99%。",
            tool_results=[_tool_result("get_window_summary", {"value": {"twr_pct": 4.5}, "currency": "SGD"})],
        )
        evaluate_case(case, result)
        self.assertTrue(any("数字无工具依据" in f for f in result.failures))

    def test_grounded_number_passes(self):
        case = {"expected_tools": [], "check_numbers_grounded": True}
        result = self._result(
            answer="收益是 4.50%。",
            tool_results=[_tool_result("get_window_summary", {"value": {"twr_pct": 4.5}, "currency": "SGD"})],
        )
        evaluate_case(case, result)
        self.assertEqual(result.failures, [])


class NoteExtractionTests(unittest.TestCase):
    def test_notes_are_read_from_answer_tail(self):
        answer = "正文内容。\n\n⚠️ 数字校验未通过。\n⚠️ 第二条。"
        self.assertEqual(len(_extract_notes(answer)), 2)

    def test_answer_without_notes_returns_empty(self):
        self.assertEqual(_extract_notes("只有正文。"), [])


class AnswerOutputTests(unittest.TestCase):
    def test_show_answer_prints_original_question_before_answer(self):
        result = CaseResult(
            case_id="DQ12",
            name="case",
            intent="data_query",
            question="为什么 2025-04 亏损？",
            answer="PA 数据不自洽。",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_case_answer(result)
        rendered = output.getvalue()
        self.assertLess(rendered.index(result.question), rendered.index(result.answer))
        self.assertIn("原始问题:", rendered)


class SummaryTests(unittest.TestCase):
    def test_summary_counts_pass_rate_and_intents(self):
        results = [
            CaseResult("A", "a", "data_query", "q", "", failures=[]),
            CaseResult("B", "b", "data_query", "q", "", failures=["x"]),
            CaseResult("C", "c", "product_info", "q", "", failures=[]),
        ]
        summary = summarize(results)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["by_intent"]["data_query"], {"total": 2, "passed": 1})
        self.assertEqual(summary["by_intent"]["product_info"], {"total": 1, "passed": 1})


class AgentIntegrationTests(unittest.TestCase):
    """The runner must observe the agent's real tool calls and review events."""

    def test_tool_result_generation_gets_larger_completion_budget(self):
        from langchain_core.messages import HumanMessage, ToolMessage

        captured = []

        class Response:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "简短回答。"}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.append(json)
            return Response()

        advisor = Advisor([], logging.getLogger("eval-token-budget-test"))
        inputs = [
            [HumanMessage(content="问题")],
            [
                HumanMessage(content="问题"),
                ToolMessage(content="工具结果", tool_call_id="call-1"),
            ],
        ]
        with patch("advisor.agent.httpx.post", fake_post):
            for messages in inputs:
                advisor._llm({
                    "intent": "data_query",
                    "messages": messages,
                    "review_repair_request": "",
                })

        self.assertEqual(captured[0]["max_tokens"], 2048)
        self.assertEqual(captured[1]["max_tokens"], 4096)

    def test_tool_calls_and_review_are_captured_from_the_agent(self):
        store = Store("data/pa.db", "data/raw")
        tools = create_tools(store)
        logger = logging.getLogger("eval-capture-test")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers = []
        collector = EventCollector()
        logger.addHandler(collector)

        advisor = Advisor(tools, logger, data_as_of=lambda: store.current().period_end)

        class Response:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {
                    "content": json.dumps({"intent": "data_query", "reason": "测试"}),
                }}]}

        from langchain_core.messages import HumanMessage

        with patch("advisor.router.httpx.post", return_value=Response()):
            advisor.trace_id = "capture-trace"
            result = advisor.graph.invoke(
                {"messages": [HumanMessage(content="过去三年哪个月份收益率最高？")], "intent": ""},
                {"configurable": {"thread_id": "capture-thread"}},
            )

        events = [e for e in collector.events if e.get("trace_id") == "capture-trace"]
        router_events = [e for e in events if e.get("event") == "router_classified"]
        self.assertTrue(router_events, "router classification must be captured")
        self.assertEqual(router_events[0]["intent"], "data_query")
        self.assertTrue(result["messages"][-1].content)


class PreflightTests(unittest.TestCase):
    """Configuration errors must be reported clearly instead of failing every case."""

    def _preflight(self, status_code, body="", model="test/model"):
        from eval import runner as runner_module

        class Response:
            def __init__(self):
                self.status_code = status_code
                self.text = body

            def raise_for_status(self):
                return None

        env = {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.example/v1",
            "LLM_MODEL": model,
        }
        with patch.dict(os.environ, env), patch(
            "httpx.post", return_value=Response()
        ):
            return runner_module.preflight_llm()

    def test_404_is_reported_as_unknown_model(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._preflight(404, model="unknown/model")
        self.assertIn("404", str(ctx.exception))
        self.assertIn("unknown/model", str(ctx.exception))

    def test_410_is_reported_as_retired_model(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._preflight(410, model="retired/model")
        self.assertIn("410", str(ctx.exception))
        self.assertIn("已下线", str(ctx.exception))

    def test_401_is_reported_as_bad_key(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._preflight(401)
        self.assertIn("LLM_API_KEY", str(ctx.exception))

    def test_other_errors_include_response_body(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._preflight(400, body="bad request details")
        self.assertIn("bad request details", str(ctx.exception))

    def test_500_generation_failure_mentions_token_budget(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._preflight(500, body='{"message":"Failed to generate completions"}')
        message = str(ctx.exception)
        self.assertIn("500", message)
        self.assertIn("max_tokens", message)

    def test_preflight_allows_room_for_reasoning_tokens(self):
        """The preflight cap must exceed the reasoning budget these models use."""
        from eval import runner as runner_module

        captured = {}

        class Response:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(json or {})
            return Response()

        env = {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.example/v1",
            "LLM_MODEL": "test/model",
        }
        with patch.dict(os.environ, env), patch("httpx.post", fake_post):
            runner_module.preflight_llm()
        self.assertGreaterEqual(captured["max_tokens"], 2048)

    def test_success_returns_status_line(self):
        status = self._preflight(200)
        self.assertIn("可用", status)


class FailureDetailTests(unittest.TestCase):
    """A failed run must explain itself, not just say '运行异常'."""

    def _collector(self, events):
        from eval.runner import EventCollector

        collector = EventCollector()

        class Record(logging.LogRecord):
            def __init__(self, msg):
                super().__init__("eval", 20, "", 0, msg, (), None)

        for event in events:
            collector.emit(Record(json.dumps(event)))
        return collector

    def test_failure_detail_includes_status_and_body(self):
        collector = self._collector([
            {"event": "llm_http_request", "intent": "data_query", "payload": {
                "model": "test/model", "max_tokens": 2048, "tools": [1] * 11, "messages": [1, 2]}},
            {"event": "llm_http_response", "status": 500,
             "body": '{"message":"Failed to generate completions"}'},
        ])
        detail = _llm_failure_detail(collector)
        self.assertIn("500", detail)
        self.assertIn("Failed to generate completions", detail)

    def test_failure_detail_includes_request_shape(self):
        collector = self._collector([
            {"event": "llm_http_request", "intent": "data_query", "payload": {
                "model": "test/model", "max_tokens": 2048, "tools": [1] * 11, "messages": [1, 2]}},
            {"event": "llm_http_response", "status": 500, "body": "boom"},
        ])
        detail = _llm_failure_detail(collector)
        self.assertIn("test/model", detail)
        self.assertIn("tools=11", detail)
        self.assertIn("max_tokens=2048", detail)

    def test_no_detail_when_all_calls_succeeded(self):
        collector = self._collector([
            {"event": "llm_http_response", "status": 200, "body": '{"choices":[]}'},
        ])
        self.assertEqual(_llm_failure_detail(collector), "")


class RouterParsingTests(unittest.TestCase):
    """The router must classify correctly for reasoning-model output shapes."""

    def _classify(self, content=None, reasoning=None):
        from advisor import router as router_module

        class Response:
            status_code = 200
            text = ""

            def __init__(self):
                self._message = {"content": content}
                if reasoning is not None:
                    self._message["reasoning_content"] = reasoning

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": self._message}]}

        with patch("advisor.router.httpx.post", return_value=Response()):
            return router_module.classify_intent("为什么 2025-03 收益高", logging.getLogger("router-test"), "t")

    def test_plain_json_is_classified(self):
        self.assertEqual(self._classify(content='{"intent": "data_query", "reason": "x"}'), "data_query")

    def test_markdown_fenced_json_is_classified(self):
        self.assertEqual(
            self._classify(content='```json\n{"intent": "data_query", "reason": "x"}\n```'),
            "data_query",
        )

    def test_json_wrapped_in_prose_is_classified(self):
        self.assertEqual(
            self._classify(content='好的，分类结果：{"intent": "data_query", "reason": "x"} 完毕'),
            "data_query",
        )

    def test_json_in_reasoning_content_is_classified(self):
        self.assertEqual(
            self._classify(content=None, reasoning='We must output {"intent": "data_query", "reason": "x"} done'),
            "data_query",
        )

    def test_multiple_json_objects_in_reasoning_use_the_first(self):
        self.assertEqual(
            self._classify(content=None, reasoning='First {"intent": "data_query"} then {"intent":"product_info"}'),
            "data_query",
        )

    def test_unparseable_response_falls_back_to_unsupported(self):
        self.assertEqual(
            self._classify(content=None, reasoning='I think this is a data query but there is no JSON'),
            "unsupported",
        )

    def test_empty_response_falls_back_to_unsupported(self):
        self.assertEqual(self._classify(content=None), "unsupported")

    def test_placeholder_intent_is_rejected(self):
        from advisor.router import _normalize_intent
        self.assertIsNone(_normalize_intent("<data_query|product_info|unsupported>"))
        self.assertIsNone(_normalize_intent("data_query|product_info"))
        self.assertEqual(_normalize_intent("data_query"), "data_query")

    def test_placeholder_intent_triggers_a_retry(self):
        """A model copying the prompt placeholder must not silently become unsupported."""
        from advisor import router as router_module

        class Response:
            status_code = 200
            text = ""

            def __init__(self, content):
                self._content = content

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        responses = iter([
            Response('{"intent": "<data_query|product_info|unsupported>", "reason": "x"}'),
            Response('{"intent": "product_info", "reason": "ok"}'),
        ])
        with patch("advisor.router.httpx.post", side_effect=lambda *a, **k: next(responses)):
            intent = router_module.classify_intent("q", logging.getLogger("router-test"), "t")
        self.assertEqual(intent, "product_info")


class ToolFactAssertionTests(unittest.TestCase):
    def test_tool_value_assertion_matches_label_and_value_in_markdown_table_row(self):
        answer = (
            "| PA 类别 | 贡献（百分点） |\n"
            "|--------|----------------|\n"
            "| 期货 | **-12.29 pp〔来源：工具字段〕** |\n"
        )
        result = CaseResult(
            "DQ12", "monthly attribution", "data_query", "q",
            answer,
            actual_tools=["get_month_detail"],
            tool_results=[
                _tool_result("get_month_detail", {
                    "contribution_pp": {"futures": -12.286702847}
                })
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_month_detail"],
                "tool_value_assertions": [{
                    "tool": "get_month_detail",
                    "path": "contribution_pp.futures",
                    "label": "期货",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertEqual(result.failures, [])

    def test_tool_value_assertion_does_not_match_value_from_another_table_row(self):
        answer = (
            "| 股票 | **-12.29 pp** |\n"
            "| 期货 | **+10.73 pp** |\n"
        )
        result = CaseResult(
            "DQ12", "monthly attribution", "data_query", "q",
            answer,
            actual_tools=["get_month_detail"],
            tool_results=[
                _tool_result("get_month_detail", {
                    "contribution_pp": {"futures": -12.286702847}
                })
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_month_detail"],
                "tool_value_assertions": [{
                    "tool": "get_month_detail",
                    "path": "contribution_pp.futures",
                    "label": "期货",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertTrue(any("回答未包含工具" in failure for failure in result.failures))

    def test_tool_value_assertion_skips_nearer_number_from_previous_sentence(self):
        answer = "YTD 盈亏 +92,088.92。股票/ETF 估算 +4,669.06 SGD。"
        result = CaseResult(
            "DQ05", "attribution", "data_query", "q",
            answer,
            actual_tools=["get_attribution"],
            tool_results=[
                _tool_result("get_attribution", {
                    "value": {
                        "buckets": [
                            {"label": "股票/ETF", "amount": 4669.0555520049165}
                        ]
                    }
                })
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_attribution"],
                "tool_value_assertions": [{
                    "tool": "get_attribution",
                    "path": "value.buckets.0.amount",
                    "label_path": "value.buckets.0.label",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertEqual(result.failures, [])

    def test_tool_value_assertion_checks_value_near_dynamic_label(self):
        result = CaseResult(
            "DQ01", "extrema", "data_query", "q",
            "最高月份 2025-03，收益 4.49%；最低月份 2025-05，收益 -4.02%。",
            actual_tools=["get_monthly_returns"],
            tool_results=[
                _tool_result("get_monthly_returns", {
                    "value": {
                        "best_month": {"month": "2025-03", "account_return_pct": 4.49},
                        "worst_month": {"month": "2025-05", "account_return_pct": -4.02},
                    }
                })
            ],
        )
        case = {
            "expected_tools": ["get_monthly_returns"],
            "tool_value_assertions": [
                {
                    "tool": "get_monthly_returns",
                    "path": "value.best_month.account_return_pct",
                    "label_path": "value.best_month.month",
                },
                {
                    "tool": "get_monthly_returns",
                    "path": "value.worst_month.account_return_pct",
                    "label_path": "value.worst_month.month",
                },
            ],
            "check_numbers_grounded": False,
        }
        _evaluate_case(case, result)
        self.assertEqual(result.failures, [])

    def test_tool_value_assertion_accepts_alternative_natural_labels(self):
        result = CaseResult(
            "DQ12", "attribution", "data_query", "q",
            "期货/期权 -16.35 pp；类别合计与 TWR 差 -14.16 pp。",
            actual_tools=["get_month_detail"],
            tool_results=[
                _tool_result("get_month_detail", {
                    "contribution_pp": {
                        "futures_options": -16.348891819,
                        "twr_gap_pp": -14.16149678,
                    }
                })
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_month_detail"],
                "tool_value_assertions": [
                    {
                        "tool": "get_month_detail",
                        "path": "contribution_pp.futures_options",
                        "labels": ["期货期权", "期货/期权"],
                    },
                    {
                        "tool": "get_month_detail",
                        "path": "contribution_pp.twr_gap_pp",
                        "labels": ["相差", "差"],
                    },
                ],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertEqual(result.failures, [])

    def test_tool_value_assertion_requires_the_matching_source_path_when_requested(self):
        result = CaseResult(
            "DQ12", "attribution", "data_query", "q",
            "-12.29 pp [1]\n\n数字来源：\n"
            "[1] get_month_detail$.contribution_pp.derivatives_merged",
            actual_tools=["get_month_detail"],
            tool_results=[
                _tool_result("get_month_detail", {
                    "contribution_pp": {"futures": -12.286702847}
                })
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_month_detail"],
                "tool_value_assertions": [{
                    "tool": "get_month_detail",
                    "path": "contribution_pp.futures",
                    "require_source_path": True,
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertTrue(any("没有与工具 get_month_detail 字段 contribution_pp.futures 对应的来源索引" in failure for failure in result.failures))

    def test_tool_value_assertion_rejects_value_attached_to_wrong_label(self):
        result = CaseResult(
            "DQ01", "extrema", "data_query", "q",
            "2025-03 为 -4.02%，2025-05 为 4.49%。",
            actual_tools=["get_monthly_returns"],
            tool_results=[
                _tool_result("get_monthly_returns", {
                    "value": {
                        "best_month": {"month": "2025-03", "account_return_pct": 4.49},
                    }
                })
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_monthly_returns"],
                "tool_value_assertions": [{
                    "tool": "get_monthly_returns",
                    "path": "value.best_month.account_return_pct",
                    "label_path": "value.best_month.month",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertTrue(any("回答未包含工具" in failure for failure in result.failures))

    def test_tool_value_assertion_accepts_matching_value_in_any_tool_call(self):
        result = CaseResult(
            "DQ01", "extrema", "data_query", "q", "收益 4.49%。",
            actual_tools=["get_monthly_returns", "get_monthly_returns"],
            tool_results=[
                _tool_result("get_monthly_returns", {"value": {"account_return_pct": 2.0}}),
                _tool_result("get_monthly_returns", {"value": {"account_return_pct": 4.49}}),
            ],
        )
        _evaluate_case(
            {
                "expected_tools": ["get_monthly_returns"],
                "tool_value_assertions": [{
                    "tool": "get_monthly_returns",
                    "path": "value.account_return_pct",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertEqual(result.failures, [])


class SemanticJudgeTests(unittest.TestCase):
    def test_judge_validates_semantic_decision_without_requiring_verbatim_evidence(self):
        from eval.semantic_judge import review_case

        rubric = [{"id": "cause-boundary", "criterion": "不得作无依据因果归因"}]
        content = json.dumps({
            "results": [{
                "id": "cause-boundary",
                "decision": "pass",
                "evidence": "无法确定原因",
                "reason": "回答明确表达不确定性",
            }]
        }, ensure_ascii=False)
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        response.raise_for_status.return_value = None
        with patch("eval.semantic_judge.httpx.post", return_value=response):
            reviewed = review_case("DQ12", "q", "无法确定原因", [], rubric)
        self.assertEqual(reviewed["results"][0]["decision"], "pass")
        self.assertEqual(reviewed["results"][0]["evidence"], "无法确定原因")

    def test_semantic_facts_projection_excludes_unrequested_tool_fields(self):
        from eval.semantic_judge import select_tool_facts

        selected = select_tool_facts(
            [_tool_result("get_capabilities", {
                "value": {
                    "period": ["2017-08-01", "2026-10-09"],
                    "snapshots": [{"account": "private"}],
                }
            })],
            [{"tool": "get_capabilities", "paths": ["value.period"]}],
        )
        self.assertEqual(selected, [{
            "tool": "get_capabilities",
            "facts": {"value.period": ["2017-08-01", "2026-10-09"]},
        }])

    def test_semantic_facts_projection_fails_when_a_required_path_is_missing(self):
        from eval.semantic_judge import SemanticJudgeError, select_tool_facts

        with self.assertRaises(SemanticJudgeError):
            select_tool_facts(
                [_tool_result("get_capabilities", {"value": {}})],
                [{"tool": "get_capabilities", "paths": ["value.period"]}],
            )

    def test_judge_does_not_change_semantic_decision_for_paraphrased_evidence(self):
        from eval.semantic_judge import review_case

        rubric = [
            {"id": "identify-inconsistency", "criterion": "说明归因数据不一致"},
            {"id": "cause-boundary", "criterion": "不得作无依据因果归因"},
        ]
        content = json.dumps({
            "results": [
                {
                    "id": "identify-inconsistency",
                    "decision": "pass",
                    "evidence": "not quoted",
                    "reason": "理由",
                },
                {
                    "id": "cause-boundary",
                    "decision": "pass",
                    "evidence": "无法根据归因确定亏损原因",
                    "reason": "回答没有作出因果推断",
                },
            ]
        }, ensure_ascii=False)
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        response.raise_for_status.return_value = None
        with patch("eval.semantic_judge.httpx.post", return_value=response):
            reviewed = review_case(
                "DQ12",
                "q",
                "归因数据不一致。无法根据归因确定亏损原因。",
                [],
                rubric,
            )
        self.assertEqual(reviewed["status"], "completed")
        self.assertEqual(reviewed["results"][0]["decision"], "pass")
        self.assertEqual(reviewed["results"][0]["evidence"], "not quoted")
        self.assertEqual(reviewed["results"][1]["decision"], "pass")

    def test_judge_does_not_retry_for_non_verbatim_or_other_language_evidence(self):
        from eval.semantic_judge import review_case

        rubric = [{"id": "identify-inconsistency", "criterion": "说明归因数据不一致"}]
        invalid = json.dumps({
            "results": [{
                "id": "identify-inconsistency",
                "decision": "pass",
                "evidence": "The answer identifies the inconsistency.",
                "reason": "The response says that the values do not reconcile.",
            }]
        })
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": invalid}}]}
        response.raise_for_status.return_value = None
        with patch("eval.semantic_judge.httpx.post", return_value=response) as post:
            reviewed = review_case(
                "DQ12",
                "q",
                "类别贡献与 TWR 不一致，无法作为完整对账拆分。",
                [],
                rubric,
            )
        self.assertEqual(post.call_count, 1)
        self.assertEqual(reviewed["results"][0]["decision"], "pass")
        self.assertEqual(
            reviewed["results"][0]["evidence"],
            "The answer identifies the inconsistency.",
        )

    def test_judge_matches_evidence_with_unicode_whitespace_variants(self):
        from eval.semantic_judge import review_case

        rubric = [{"id": "attribution-boundary", "criterion": "说明不能可靠归因"}]
        content = json.dumps({
            "results": [{
                "id": "attribution-boundary",
                "decision": "pass",
                "evidence": "无法可靠地判断\n亏损来源。",
                "reason": "回答指出限制",
            }]
        }, ensure_ascii=False)
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        response.raise_for_status.return_value = None
        answer = "无法可靠地判断 亏损来源。\u00a0更多说明。"
        with patch("eval.semantic_judge.httpx.post", return_value=response):
            reviewed = review_case("DQ12", "q", answer, [], rubric)
        self.assertEqual(reviewed["results"][0]["decision"], "pass")
        self.assertEqual(reviewed["results"][0]["evidence"], "无法可靠地判断\n亏损来源。")

    def test_semantic_fail_remains_a_failure_alongside_deterministic_checks(self):
        result = CaseResult(
            "DQ12", "attribution", "data_query", "q", "TWR -3.75%。亏损由市场回调造成。",
            semantic_review={
                "status": "completed",
                "results": [{
                    "id": "cause-boundary",
                    "decision": "fail",
                    "evidence": "亏损由市场回调造成",
                    "reason": "因果结论缺乏工具依据",
                }],
            },
        )
        _evaluate_case(
            {
                "semantic_rubric": [{
                    "id": "cause-boundary",
                    "criterion": "不得作无依据因果归因",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertTrue(any("语义评审 fail" in failure for failure in result.failures))

    def test_semantic_rubric_without_review_fails_closed(self):
        result = CaseResult("PI01", "capabilities", "product_info", "q", "答案。")
        _evaluate_case(
            {
                "semantic_rubric": [{
                    "id": "coverage",
                    "criterion": "准确回答问题",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertTrue(any("语义评审未执行" in failure for failure in result.failures))

    def test_semantic_pass_without_optional_evidence_is_accepted(self):
        result = CaseResult(
            "PI01", "capabilities", "product_info", "q", "答案。",
            semantic_review={
                "status": "completed",
                "results": [{
                    "id": "coverage",
                    "decision": "pass",
                    "evidence": "",
                    "reason": "符合",
                }],
            },
        )
        _evaluate_case(
            {
                "semantic_rubric": [{
                    "id": "coverage",
                    "criterion": "准确回答问题",
                }],
                "check_numbers_grounded": False,
            },
            result,
        )
        self.assertEqual(result.failures, [])


class FixedReplyAssertionTests(unittest.TestCase):
    def test_unsupported_requires_expected_route_and_exact_fixed_reply(self):
        from advisor.router import extract_fixed_reply, load_compose_template

        expected = extract_fixed_reply(load_compose_template("unsupported"))
        case = {"intent": "unsupported", "expect_fixed_reply": True, "expected_tools": []}
        result = CaseResult(
            "UN01", "unsupported", "unsupported", "q", expected,
            actual_intent="unsupported",
        )
        _evaluate_case(case, result)
        self.assertEqual(result.failures, [])

    def test_unsupported_wrong_route_fails_even_if_reply_text_matches(self):
        from advisor.router import extract_fixed_reply, load_compose_template

        expected = extract_fixed_reply(load_compose_template("unsupported"))
        case = {"intent": "unsupported", "expect_fixed_reply": True, "expected_tools": []}
        result = CaseResult(
            "UN01", "unsupported", "unsupported", "q", expected,
            actual_intent="product_info",
        )
        _evaluate_case(case, result)
        self.assertTrue(any("固定拒答用例未路由" in failure for failure in result.failures))


if __name__ == "__main__":
    unittest.main()
