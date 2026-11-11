"""Tests for common.output_review — shared answer-quality logic.

No LLM API keys required.
"""
from __future__ import annotations

import json
import logging
import unittest

from common.output_review import (
    CheckContext,
    check_numeric_provenance,
    ungrounded_numbers,
)


def _logger():
    return logging.getLogger("test_output_review")


def _context(candidate: str, tool_results: list | None = None) -> CheckContext:
    return CheckContext(
        intent="data_query",
        question="账户今年收益是多少？",
        candidate=candidate,
        tool_results=tool_results or [],
        logger=_logger(),
        trace_id="test-trace",
    )


class TestCheckNumericProvenance(unittest.TestCase):
    def _tool_result(self, value_dict: dict, currency: str = "SGD") -> dict:
        return {
            "tool": "get_window_summary",
            "content": json.dumps(
                {"value": value_dict, "currency": currency, "status": "ok"},
                ensure_ascii=False,
            ),
        }

    def test_grounded_ratio_passes(self):
        ctx = _context(
            candidate="今年收益 +12.34%。",
            tool_results=[self._tool_result({"twr_pct": 12.34})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertTrue(outcome.passed, outcome.reason)

    def test_ungrounded_ratio_fails(self):
        ctx = _context(
            candidate="今年收益 +99.99%。",
            tool_results=[self._tool_result({"twr_pct": 12.34})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertFalse(outcome.passed)
        self.assertIn("99.99", outcome.reason)

    def test_grounded_amount_passes(self):
        ctx = _context(
            candidate="盈亏 +50,000 SGD。",
            tool_results=[self._tool_result({"profit": 50_000})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertTrue(outcome.passed, outcome.reason)

    def test_ungrounded_amount_fails(self):
        ctx = _context(
            candidate="盈亏 +999,000 SGD。",
            tool_results=[self._tool_result({"profit": 50_000})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertFalse(outcome.passed)

    def test_annotation_marks_grounded_number_with_source_index(self):
        ctx = _context(
            candidate="收益 +12.34%。",
            tool_results=[self._tool_result({"twr_pct": 12.34})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertIn("[1]", outcome.candidate)

    def test_annotation_marks_ungrounded_number(self):
        ctx = _context(
            candidate="收益 +99.99%。",
            tool_results=[self._tool_result({"twr_pct": 12.34})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertIn("未在工具结果中匹配", outcome.candidate)

    def test_rounded_value_is_grounded(self):
        ctx = _context(
            candidate="盈亏 +420,879 SGD。",
            tool_results=[self._tool_result({"profit": 420878.989609007})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertTrue(outcome.passed, outcome.reason)

    def test_no_numbers_in_candidate_always_passes(self):
        ctx = _context(
            candidate="这是一条没有数字的描述。",
            tool_results=[self._tool_result({"twr_pct": 12.34})],
        )
        outcome = check_numeric_provenance(ctx)
        self.assertTrue(outcome.passed)


class TestUngroundedNumbers(unittest.TestCase):
    def test_no_tool_results_returns_empty_list(self):
        result = ungrounded_numbers(
            candidate="今年收益 +12.34%，盈亏 50,000 SGD。",
            tool_results=[],
        )
        self.assertEqual(result, [])

    def test_grounded_numbers_not_in_result(self):
        tool_results = [{
            "tool": "get_window_summary",
            "content": json.dumps(
                {"value": {"twr_pct": 12.34, "profit": 50_000}, "currency": "SGD"},
                ensure_ascii=False,
            ),
        }]
        result = ungrounded_numbers(
            candidate="今年收益 +12.34%，盈亏 +50,000 SGD。",
            tool_results=tool_results,
        )
        self.assertEqual(result, [])

    def test_ungrounded_numbers_are_returned(self):
        tool_results = [{
            "tool": "get_window_summary",
            "content": json.dumps(
                {"value": {"twr_pct": 12.34}, "currency": "SGD"},
                ensure_ascii=False,
            ),
        }]
        result = ungrounded_numbers(
            candidate="今年收益 +99.99%。",
            tool_results=tool_results,
        )
        self.assertTrue(any("99.99%" in s for s in result), f"expected 99.99% in {result}")

    def test_dates_not_included_in_ungrounded(self):
        tool_results = [{
            "tool": "get_window_summary",
            "content": json.dumps({"value": {"twr_pct": 5.0}, "currency": "SGD"}),
        }]
        result = ungrounded_numbers(
            candidate="截止 2025-03 收益 +5.00%。",
            tool_results=tool_results,
        )
        self.assertEqual(result, [])

    def test_structural_numbers_not_flagged(self):
        tool_results = [{
            "tool": "get_window_summary",
            "content": json.dumps({"value": {"twr_pct": 5.0}, "currency": "SGD"}),
        }]
        result = ungrounded_numbers(
            candidate="1. 规模与资金\n2. 收益 +5.00%。",
            tool_results=tool_results,
        )
        self.assertEqual(result, [])
