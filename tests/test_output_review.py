import json
import logging
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from advisor.agent import Advisor
from advisor.checklist_config import GLOBAL_CHECKS, GLOBAL_CHECKLIST, INTENT_CHECKLISTS
from advisor.output_review import CheckContext, check_numeric_provenance, parse_review_verdict
from advisor.router import load_output_review_config


class OutputReviewTests(unittest.TestCase):
    def test_parse_review_verdict_accepts_supported_decisions(self):
        for decision in ("pass", "fail", "uncertain"):
            with self.subTest(decision=decision):
                parsed = parse_review_verdict(json.dumps({
                    "decision": decision,
                    "reason": "Policy review result.",
                }))
                self.assertEqual(parsed["decision"], decision)

    def test_parse_review_verdict_rejects_invalid_shapes(self):
        invalid_responses = (
            "not json",
            json.dumps({"decision": "pass"}),
            json.dumps({"decision": "unknown", "reason": "No."}),
            json.dumps({"decision": "pass", "reason": "OK", "extra": "field"}),
            json.dumps({"decision": "pass", "reason": ""}),
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_review_verdict(response)

    def _advisor(self):
        return Advisor([], logging.getLogger("output-review-test"))

    def _state(self, answer, intent="product_info", tool_results=()):
        messages = [HumanMessage(content="What does this system support?")]
        messages.extend(tool_results)
        messages.append(AIMessage(content=answer))
        return {"intent": intent, "messages": messages}

    def _mock_response(self, decision):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": json.dumps({
                    "decision": decision,
                    "reason": "Review result.",
                })}}]}

        return Response()

    def _llm_env(self):
        return patch.dict(os.environ, {
            "LLM_MODEL": "test-model",
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.example",
        })

    def test_pass_verdict_keeps_candidate_answer(self):
        advisor = self._advisor()
        with self._llm_env(), patch(
            "advisor.output_review.httpx.post", return_value=self._mock_response("pass")
        ):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertEqual(result, {})

    def test_non_pass_verdict_preserves_answer_and_adds_note(self):
        advisor = self._advisor()
        for decision in ("fail", "uncertain"):
            with self.subTest(decision=decision):
                with self._llm_env(), patch(
                    "advisor.output_review.httpx.post",
                    return_value=self._mock_response(decision),
                ):
                    result = advisor._output_review(self._state("Candidate answer."))
                answer = result["messages"][0].content
                self.assertIn("Candidate answer.", answer)
                self.assertIn(GLOBAL_CHECKS["product_policy"]["failure_note"], answer)

    def test_invalid_verdict_fails_open_with_note(self):
        advisor = self._advisor()

        class InvalidResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "not JSON"}}]}

        with self._llm_env(), patch("advisor.output_review.httpx.post", return_value=InvalidResponse()):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertIn("Candidate answer.", result["messages"][0].content)
        self.assertIn(GLOBAL_CHECKS["product_policy"]["failure_note"], result["messages"][0].content)

    def test_review_error_fails_open_with_note(self):
        advisor = self._advisor()
        with self._llm_env(), patch(
            "advisor.output_review.httpx.post",
            side_effect=RuntimeError("review service unavailable"),
        ):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertIn("Candidate answer.", result["messages"][0].content)
        self.assertIn(GLOBAL_CHECKS["product_policy"]["failure_note"], result["messages"][0].content)

    def test_numeric_values_are_normalized_and_annotated(self):
        context = CheckContext(
            intent="data_query",
            question="Is my return 20%?",
            candidate="收益 -19.35%，金额为 -5万 USD。",
            tool_results=[{
                "tool": "get_window_summary",
                "content": json.dumps({
                    "value": {"twr_pct": 19.346, "profit": 50_000},
                    "currency": "USD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        outcome = check_numeric_provenance(context)
        self.assertTrue(outcome.passed)
        self.assertIn("-19.35%〔来源：get_window_summary$.value.twr_pct〕", outcome.candidate)
        self.assertIn("-5万 USD〔来源：get_window_summary$.value.profit〕", outcome.candidate)
        self.assertIn("数字来源：", outcome.candidate)

    def test_numeric_values_in_tool_text_are_valid_evidence(self):
        context = CheckContext(
            intent="data_query",
            question="How large was the return gap?",
            candidate="工具说明中的差异为 2.50%。",
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({"warnings": ["差异为 -2.5%"]}, ensure_ascii=False),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        outcome = check_numeric_provenance(context)
        self.assertTrue(outcome.passed)
        self.assertIn("get_month_detail$.warnings[0]", outcome.candidate)

    def test_amount_rounding_uses_displayed_decimal_places(self):
        context = CheckContext(
            intent="data_query",
            question="What was the amount?",
            candidate="金额为 123.45 USD。",
            tool_results=[{
                "tool": "get_window_summary",
                "content": json.dumps({
                    "value": {"profit": 123.454},
                    "currency": "USD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        self.assertTrue(check_numeric_provenance(context).passed)

    def test_unmatched_number_fails_check_without_rejecting_answer(self):
        advisor = self._advisor()
        state = self._state("收益是 19.35%。", intent="data_query")
        result = advisor._output_review(state)
        answer = result["messages"][0].content
        self.assertIn("19.35%〔来源：未在工具结果中匹配〕", answer)
        self.assertIn(GLOBAL_CHECKS["numeric_provenance"]["failure_note"], answer)

    def test_enabled_checklist_is_declared_per_intent(self):
        self.assertEqual(GLOBAL_CHECKLIST, ["numeric_provenance"])
        self.assertEqual(INTENT_CHECKLISTS["data_query"], [])
        self.assertEqual(INTENT_CHECKLISTS["product_info"], ["product_policy"])
        self.assertIn("unsupported", INTENT_CHECKLISTS)

    def test_disabled_check_is_skipped(self):
        advisor = self._advisor()
        with patch.dict(GLOBAL_CHECKS["numeric_provenance"], {"enabled": False}):
            result = advisor._output_review(self._state("Number 123.", intent="data_query"))
        self.assertEqual(result, {})

    def test_missing_review_context_keeps_answer_with_warning(self):
        advisor = self._advisor()
        result = advisor._output_review({
            "intent": "data_query",
            "messages": [AIMessage(content="Candidate answer.")],
        })
        answer = result["messages"][0].content
        self.assertIn("Candidate answer.", answer)
        self.assertIn("答案检查输入不完整", answer)


if __name__ == "__main__":
    unittest.main()
