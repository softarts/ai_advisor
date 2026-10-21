import json
import logging
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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

    def test_parse_review_verdict_tolerates_surrounding_prose(self):
        responses = (
            '好的，判定结果：{"decision": "pass", "reason": "符合策略"}',
            '```json\n{"decision": "fail", "reason": "越界"}\n```',
            '{"decision": "pass", "reason": "符合策略"} 完毕。',
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIn(parse_review_verdict(response)["decision"], {"pass", "fail", "uncertain"})

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
        self.assertIn(GLOBAL_CHECKS["product_policy"]["unavailable_note"], result["messages"][0].content)

    def test_review_error_fails_open_with_note(self):
        advisor = self._advisor()
        with self._llm_env(), patch(
            "advisor.output_review.httpx.post",
            side_effect=RuntimeError("review service unavailable"),
        ):
            result = advisor._output_review(self._state("Candidate answer."))
        self.assertIn("Candidate answer.", result["messages"][0].content)
        self.assertIn(GLOBAL_CHECKS["product_policy"]["unavailable_note"], result["messages"][0].content)

    def test_numeric_values_are_normalized_and_annotated(self):
        context = CheckContext(
            intent="data_query",
            question="Is my return 20%?",
            candidate="收益 +19.35%，金额为 +5万 USD。",
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
        self.assertIn("+19.35%〔来源：get_window_summary$.value.twr_pct〕", outcome.candidate)
        self.assertIn("+5万 USD〔来源：get_window_summary$.value.profit〕", outcome.candidate)
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


class NumericSignTests(unittest.TestCase):
    """A sign flip on a financial figure must not pass the provenance check."""

    def _check(self, candidate, tool_value, key, currency="SGD"):
        context = CheckContext(
            intent="data_query",
            question="q",
            candidate=candidate,
            tool_results=[{
                "tool": "get_window_summary",
                "content": json.dumps({
                    "value": {key: tool_value},
                    "currency": currency,
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        return check_numeric_provenance(context)

    def test_opposite_sign_on_ratio_is_rejected(self):
        outcome = self._check("收益 -4.50%。", 4.50, "twr_pct")
        self.assertFalse(outcome.passed)
        self.assertIn("4.50", outcome.reason)

    def test_matching_sign_on_ratio_passes(self):
        self.assertTrue(self._check("收益 +4.50%。", 4.50, "twr_pct").passed)

    def test_unsigned_number_still_passes(self):
        self.assertTrue(self._check("收益 4.50%。", 4.50, "twr_pct").passed)

    def test_opposite_sign_on_amount_is_rejected(self):
        outcome = self._check("盈亏 -18,935.00 SGD。", 18935.00, "profit")
        self.assertFalse(outcome.passed)

    def test_matching_sign_on_amount_passes(self):
        self.assertTrue(self._check("盈亏 +18,935.00 SGD。", 18935.00, "profit").passed)


class UnavailableCheckTests(unittest.TestCase):
    """A check that could not run must be reported differently from a failed check."""

    def test_unrunnable_check_uses_unavailable_note(self):
        advisor = Advisor([], logging.getLogger("output-review-test"))
        with patch.dict(os.environ, {}, clear=True):
            result = advisor._output_review({
                "intent": "product_info",
                "messages": [HumanMessage(content="什么是 CFD？"), AIMessage(content="CFD 是一种衍生品。")],
            })
        answer = result["messages"][0].content
        self.assertIn("CFD 是一种衍生品。", answer)
        self.assertIn(GLOBAL_CHECKS["product_policy"]["unavailable_note"], answer)
        self.assertNotIn(GLOBAL_CHECKS["product_policy"]["failure_note"], answer)


class OutputReviewLoggingTests(unittest.TestCase):
    """The review must be observable in the JSONL log for every reviewed answer."""

    def _capturing_logger(self):
        logger = logging.getLogger("output-review-log-test")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(json.loads(record.getMessage()))

        logger.addHandler(Capture())
        return logger, records

    def test_passing_review_is_logged_with_summary(self):
        logger, records = self._capturing_logger()
        advisor = Advisor([], logger)
        advisor.trace_id = "trace-pass"
        advisor._output_review({
            "intent": "data_query",
            "messages": [HumanMessage(content="q"), AIMessage(content="没有具体数字的描述。")],
        })
        events = [record.get("event") for record in records]
        self.assertIn("output_review_check", events)
        self.assertIn("output_review_summary", events)
        summary = next(r for r in records if r.get("event") == "output_review_summary")
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["failed_checks"], [])
        self.assertEqual(summary["checks_run"], ["numeric_provenance"])
        self.assertEqual(summary["trace_id"], "trace-pass")

    def test_failing_review_is_logged_with_failed_checks(self):
        logger, records = self._capturing_logger()
        advisor = Advisor([], logger)
        advisor.trace_id = "trace-fail"
        advisor._output_review({
            "intent": "data_query",
            "messages": [HumanMessage(content="q"), AIMessage(content="收益 19.35%。")],
        })
        summary = next(r for r in records if r.get("event") == "output_review_summary")
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["failed_checks"][0]["check_id"], "numeric_provenance")
        self.assertTrue(summary["notes"])
        self.assertTrue(summary["annotated"])

    def test_incomplete_review_input_is_logged_as_summary(self):
        logger, records = self._capturing_logger()
        advisor = Advisor([], logger)
        advisor.trace_id = "trace-incomplete"
        advisor._output_review({
            "intent": "data_query",
            "messages": [AIMessage(content="Candidate answer.")],
        })
        events = [record.get("event") for record in records]
        self.assertIn("output_review_unavailable_input", events)
        self.assertIn("output_review_summary", events)
        summary = next(r for r in records if r.get("event") == "output_review_summary")
        self.assertFalse(summary["passed"])


class AnnotationIdempotencyTests(unittest.TestCase):
    """Re-checking an already annotated answer must not duplicate annotations."""

    def _context(self, candidate):
        return CheckContext(
            intent="data_query",
            question="q",
            candidate=candidate,
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({"value": {"twr_pct": 4.50}, "currency": "SGD"}),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )

    def test_second_pass_produces_identical_output(self):
        first = check_numeric_provenance(self._context("收益 +4.50%。"))
        second = check_numeric_provenance(self._context(first.candidate))
        self.assertEqual(first.candidate, second.candidate)

    def test_annotation_block_is_not_duplicated(self):
        first = check_numeric_provenance(self._context("收益 +4.50%。"))
        second = check_numeric_provenance(self._context(first.candidate))
        self.assertEqual(second.candidate.count("数字来源："), 1)
        self.assertEqual(second.candidate.count("〔来源："), 1)

    def test_prior_annotation_is_stripped_before_matching(self):
        annotated = "收益 +4.50%〔来源：get_month_detail$.value.twr_pct〕。\n\n数字来源：\n-  +4.50%：get_month_detail$.value.twr_pct"
        outcome = check_numeric_provenance(self._context(annotated))
        self.assertTrue(outcome.passed)


class AnnotationReplacementTests(unittest.TestCase):
    """Overlapping-length annotations must not corrupt the answer text."""

    def test_date_and_number_annotations_do_not_corrupt_text(self):
        context = CheckContext(
            intent="data_query",
            question="q",
            candidate="2025-04 TWR -3.75%，触发门禁。",
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({
                    "month": "2025-04",
                    "return": {"twr_pct": -3.746787323},
                    "currency": "SGD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        outcome = check_numeric_provenance(context)
        self.assertIn("2025-04〔来源：", outcome.candidate)
        self.assertIn("-3.75%〔来源：", outcome.candidate)
        self.assertIn("触发门禁。", outcome.candidate)
        self.assertNotIn("〕th", outcome.candidate)
        self.assertNotIn("S-3.75", outcome.candidate)

    def test_annotations_preserve_original_reading_order(self):
        context = CheckContext(
            intent="data_query",
            question="q",
            candidate="2025-04 TWR -3.75%。",
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({
                    "month": "2025-04",
                    "return": {"twr_pct": -3.746787323},
                    "currency": "SGD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        candidate = check_numeric_provenance(context).candidate
        self.assertLess(candidate.index("2025-04"), candidate.index("-3.75%"))
        self.assertIn("TWR -3.75%〔来源：", candidate)


class StructuralNumberTests(unittest.TestCase):
    """List ordinals and heading numbers are structure, not data claims."""

    def _context(self, candidate):
        return CheckContext(
            intent="data_query",
            question="q",
            candidate=candidate,
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({
                    "month": "2025-03",
                    "return": {"twr_pct": 4.494502281},
                    "currency": "SGD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )

    def test_markdown_heading_numbers_are_not_flagged(self):
        outcome = check_numeric_provenance(self._context(
            "### 2. 收益构成（PA 口径）\n### 3. 边界声明\n收益 +4.49%。"
        ))
        self.assertTrue(outcome.passed)
        self.assertNotIn("未在工具结果中匹配", outcome.candidate)
        self.assertNotIn("〔来源：未在", outcome.candidate)

    def test_list_ordinals_are_not_flagged(self):
        outcome = check_numeric_provenance(self._context(
            "1. 规模与资金\n2. 收益构成\n收益 +4.49%。"
        ))
        self.assertTrue(outcome.passed)
        self.assertNotIn("未在工具结果中匹配", outcome.candidate)

    def test_heading_number_is_not_annotated(self):
        outcome = check_numeric_provenance(self._context(
            "### 2. 收益构成\n收益 +4.49%。"
        ))
        self.assertNotIn("2〔来源：", outcome.candidate)
        self.assertIn("### 2. 收益构成", outcome.candidate)

    def test_real_numbers_in_headings_are_still_checked(self):
        """A heading that states a real figure must still be grounded."""
        outcome = check_numeric_provenance(self._context(
            "### 收益 99.99%\n"
        ))
        self.assertFalse(outcome.passed)


class RoundedValueTests(unittest.TestCase):
    """A rounded restatement of a tool value must be accepted as grounded."""

    def _check(self, candidate, tool_value, key="nav_opening", currency="SGD"):
        context = CheckContext(
            intent="data_query",
            question="q",
            candidate=candidate,
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({"value": {key: tool_value}, "currency": currency}),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        return check_numeric_provenance(context)

    def test_rounded_amount_with_padded_zeros_is_grounded(self):
        outcome = self._check("净值 420,879.00 SGD。", 420878.989609007)
        self.assertTrue(outcome.passed, outcome.reason)
        self.assertNotIn("未在工具结果中匹配", outcome.candidate)

    def test_exact_amount_is_grounded(self):
        self.assertTrue(self._check("净值 420,878.99 SGD。", 420878.989609007).passed)

    def test_integer_amount_is_grounded(self):
        self.assertTrue(self._check("净值 420,879 SGD。", 420878.989609007).passed)

    def test_genuinely_different_amount_is_rejected(self):
        self.assertFalse(self._check("净值 420,000.00 SGD。", 420878.989609007).passed)

    def test_rounded_ratio_with_padded_zeros_is_grounded(self):
        outcome = self._check("TWR +4.50%。", 4.494502281, key="twr_pct")
        self.assertTrue(outcome.passed, outcome.reason)

    def test_ratio_with_extra_precision_is_rejected(self):
        self.assertFalse(self._check("TWR +4.56%。", 4.494502281, key="twr_pct").passed)

    def test_coarser_ratio_is_grounded(self):
        self.assertTrue(self._check("TWR +4.5%。", 4.494502281, key="twr_pct").passed)

    def test_different_ratio_is_rejected(self):
        self.assertFalse(self._check("TWR +4.6%。", 4.494502281, key="twr_pct").passed)

    def test_different_ratio_does_not_match_by_collapse(self):
        """-5.7% must not match -5.63% by rounding both to the same integer."""
        self.assertFalse(self._check("SPXTR -5.7%。", -5.634429716, key="spxtr").passed)


class UnicodeSeparatorTests(unittest.TestCase):
    """Models may group digits with Unicode spaces instead of a comma."""

    NBSP = " "
    HYPHEN = "‑"

    def _context(self, candidate):
        return CheckContext(
            intent="data_query",
            question="q",
            candidate=candidate,
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({
                    "month": "2025-03",
                    "nav": {"opening": 420878.989609007, "closing": 499814.690377605},
                    "flows": {"deposits": 60000.0},
                    "currency": "SGD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )

    def test_narrow_no_break_space_grouped_amount_is_one_number(self):
        outcome = check_numeric_provenance(self._context(
            f"期初净值 420{self.NBSP}878.99 SGD，期末 499{self.NBSP}814.69 SGD。"
        ))
        self.assertTrue(outcome.passed, outcome.reason)
        self.assertNotIn("未在工具结果中匹配", outcome.candidate)

    def test_narrow_no_break_space_grouped_amount_is_annotated_as_one(self):
        outcome = check_numeric_provenance(self._context(
            f"期初净值 420{self.NBSP}878.99 SGD。"
        ))
        self.assertIn("420,878.99 SGD〔来源：get_month_detail$.nav.opening〕", outcome.candidate)

    def test_non_breaking_hyphen_date_is_recognised(self):
        outcome = check_numeric_provenance(self._context(
            f"日期 2025{self.HYPHEN}03{self.HYPHEN}24。"
        ))
        # The date must be read as one value, not split into 2025 / 03 / 24.
        self.assertIn("2025-03-24", outcome.candidate)
        self.assertNotIn("2025〔来源：", outcome.candidate)

    def test_normalisation_preserves_length(self):
        from advisor.output_review import _normalize_separators
        text = f"420{self.NBSP}878.99 和 2025{self.HYPHEN}03"
        normalized = _normalize_separators(text)
        self.assertEqual(len(normalized), len(text))
        self.assertIn("420,878.99", normalized)
        self.assertIn("2025-03", normalized)

    def test_emoji_decorated_heading_number_is_structural(self):
        from advisor.output_review import _HEADING_MARKER_RE
        for heading in ("### 2. 收益构成", "### 2️⃣ 规模与资金", "## 1️⃣ 事实"):
            with self.subTest(heading=heading):
                self.assertTrue(_HEADING_MARKER_RE.search(heading))

    def test_array_index_in_path_reference_is_structural(self):
        from advisor.output_review import _is_structural_number, _NUMERIC_RE, _structural_markers
        text = "`composition[2].pp` (note K3)"
        markers = _structural_markers(text)
        matches = list(_NUMERIC_RE.finditer(text))
        self.assertTrue(matches)
        for match in matches:
            self.assertTrue(_is_structural_number(match, markers))


class PercentagePointCategoryTests(unittest.TestCase):
    """Tools report percentage points; the model may write the same figure as %."""

    def _check(self, candidate):
        context = CheckContext(
            intent="data_query",
            question="q",
            candidate=candidate,
            tool_results=[{
                "tool": "get_month_detail",
                "content": json.dumps({
                    "composition": [{"category": "derivatives_merged", "pp": 3.3157367079999998}],
                    "contribution_pp": {"sum": 4.455070521},
                    "currency": "SGD",
                }),
            }],
            logger=logging.getLogger("output-review-test"),
            trace_id="test-trace",
        )
        return check_numeric_provenance(context)

    def test_truncated_percentage_matches_pp_value(self):
        outcome = self._check("贡献 +3.3157 %。")
        self.assertTrue(outcome.passed, outcome.reason)

    def test_rounded_percentage_matches_pp_value(self):
        outcome = self._check("贡献 +3.32 %。")
        self.assertTrue(outcome.passed, outcome.reason)

    def test_percentage_point_wording_matches_too(self):
        outcome = self._check("贡献 +3.32 pp。")
        self.assertTrue(outcome.passed, outcome.reason)

    def test_different_percentage_is_rejected(self):
        self.assertFalse(self._check("贡献 +3.4 %。").passed)

    def test_significant_digits_counts_normalized_digits(self):
        from decimal import Decimal
        from advisor.output_review import _significant_digits
        self.assertEqual(_significant_digits(Decimal("3.3157367079999998")), 17)
        self.assertEqual(_significant_digits(Decimal("3.3157")), 5)
        self.assertEqual(_significant_digits(Decimal("420879.00")), 6)


class MultiTurnReviewTests(unittest.TestCase):
    """Tool evidence from earlier turns must still support later answers."""

    def _advisor(self):
        return Advisor([], logging.getLogger("output-review-test"))

    def test_tool_result_from_earlier_turn_is_still_evidence(self):
        advisor = self._advisor()
        result = advisor._output_review({
            "intent": "data_query",
            "messages": [
                HumanMessage(content="第一轮"),
                ToolMessage(
                    content=json.dumps({"value": {"twr_pct": 4.50}, "currency": "SGD"}),
                    tool_call_id="c1",
                    name="get_window_summary",
                ),
                HumanMessage(content="第二轮"),
                AIMessage(content="收益 +4.50%。"),
            ],
        })
        answer = result["messages"][0].content
        self.assertIn("get_window_summary$.value.twr_pct", answer)
        self.assertNotIn("⚠️", answer)

    def test_invented_number_in_later_turn_is_flagged(self):
        advisor = self._advisor()
        result = advisor._output_review({
            "intent": "data_query",
            "messages": [
                HumanMessage(content="第一轮"),
                ToolMessage(
                    content=json.dumps({"value": {"twr_pct": 4.50}, "currency": "SGD"}),
                    tool_call_id="c1",
                    name="get_window_summary",
                ),
                HumanMessage(content="第二轮"),
                AIMessage(content="第一轮收益 +4.50%，这一轮收益 +99.99%。"),
            ],
        })
        answer = result["messages"][0].content
        self.assertIn("未在工具结果中匹配", answer)
        self.assertIn(GLOBAL_CHECKS["numeric_provenance"]["failure_note"], answer)


if __name__ == "__main__":
    unittest.main()
