#!/usr/bin/env python3
"""Tests for the evaluation runner's wiring and assertions.

These use a mocked LLM so the runner logic (case loading, event capture,
tool assertion, content assertion, review assertion and reporting) can be
verified without model calls.
"""
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.example")

from advisor.agent import Advisor
from advisor.tools import create_tools
from eval.report import CaseResult, evaluate_case, summarize
from eval.runner import EventCollector, load_cases, _extract_notes, _llm_failure_detail
from pa_analytics import Store


def _tool_result(tool_name, payload):
    return {"tool": tool_name, "content": json.dumps(payload, ensure_ascii=False)}


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


if __name__ == "__main__":
    unittest.main()
