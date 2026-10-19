import json
import logging
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from advisor.agent import Advisor
from advisor.output_review import parse_review_verdict
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

    def _state(self, answer):
        return {
            "intent": "product_info",
            "messages": [
                HumanMessage(content="What does this system support?"),
                AIMessage(content=answer),
            ],
        }

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
        with self._llm_env(), patch("advisor.agent.httpx.post", return_value=self._mock_response("pass")):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertEqual(result, {})

    def test_non_pass_verdict_replaces_candidate_with_configured_reply(self):
        advisor = self._advisor()
        config = load_output_review_config("product_info")
        for decision in ("fail", "uncertain"):
            with self.subTest(decision=decision):
                with self._llm_env(), patch(
                    "advisor.agent.httpx.post",
                    return_value=self._mock_response(decision),
                ):
                    result = advisor._output_review(self._state("Candidate answer."))
                self.assertEqual(result["messages"][0].content, config["failure_reply"])

    def test_invalid_verdict_fails_closed(self):
        advisor = self._advisor()
        config = load_output_review_config("product_info")

        class InvalidResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "not JSON"}}]}

        with self._llm_env(), patch("advisor.agent.httpx.post", return_value=InvalidResponse()):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertEqual(result["messages"][0].content, config["failure_reply"])

    def test_intents_without_review_configuration_pass_through(self):
        advisor = self._advisor()
        result = advisor._output_review({
            "intent": "data_query",
            "messages": [AIMessage(content="Answer.")],
        })
        self.assertEqual(result, {})

    def test_review_error_fails_closed(self):
        advisor = self._advisor()
        config = load_output_review_config("product_info")
        with self._llm_env(), patch("advisor.agent.httpx.post", side_effect=RuntimeError("review service unavailable")):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertEqual(result["messages"][0].content, config["failure_reply"])


if __name__ == "__main__":
    unittest.main()
