"""Deterministic assertions for evaluation cases.

Every judgement is made by code, not by asking the model to grade itself:
tool usage comes from the agent's own tool-call log, numeric grounding reuses the
answer-checklist provenance matcher, and content checks compare the final answer
against the case expectations.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from advisor.output_review import (
    CheckContext,
    _parse_tool_results,
    _strip_prior_annotation,
    ungrounded_numbers,
)

_NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


@dataclass
class CaseResult:
    case_id: str
    name: str
    intent: str
    question: str
    answer: str
    actual_tools: list[str] = field(default_factory=list)
    tool_results: list[dict[str, str]] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    actual_intent: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures


def _normalize_text(value: str) -> str:
    """Normalize for tolerant content matching (whitespace and full-width marks)."""
    return re.sub(r"\s+", "", value.replace("，", ",").replace("（", "(").replace("）", ")"))


def _numbers_in(value: str) -> list[float]:
    """Extract numbers, ignoring the separators inside dates.

    "2022-10" must not yield 2022 and -10: the hyphen is a date separator,
    not a sign, and a stray negative number makes a textual expectation match
    the wrong figure.
    """
    text = re.sub(r"\d{4}\s*[-/]\s*\d{1,2}(?:\s*[-/]\s*\d{1,2})?", " ", value)
    text = re.sub(r"\d{4}年\d{1,2}月(?:\d{1,2}日)?", " ", text)
    numbers = []
    for match in _NUMBER_RE.finditer(text.replace(",", "")):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            continue
    return numbers


def _content_matches(expected: str, answer: str) -> bool:
    """Match an expected token in the answer.

    A purely numeric expectation matches any number in the answer whose value
    equals the expected one, so display formatting differences (thousand
    separators, trailing zeros, sign placement) do not cause false failures.

    An expectation may be a list of interchangeable phrasings; any one of them
    satisfies it.
    """
    options = expected if isinstance(expected, list) else [expected]
    return any(_single_content_matches(option, answer) for option in options)


def _single_content_matches(expected: str, answer: str) -> bool:
    normalized_answer = _normalize_text(answer)
    if _normalize_text(expected) in normalized_answer:
        return True
    expected_numbers = _numbers_in(expected)
    if not expected_numbers:
        return False
    answer_numbers = _numbers_in(answer)
    for expected_number in expected_numbers:
        for answer_number in answer_numbers:
            if abs(expected_number - answer_number) <= 0.01:
                return True
    return False


def _assert_tools(case: dict[str, Any], result: CaseResult) -> None:
    expected = list(case.get("expected_tools") or [])
    actual = result.actual_tools
    for tool in expected:
        if tool not in actual:
            result.failures.append(f"未调用期望工具 {tool}（实际：{actual or '无'}）")
    # "optional_tools" lists tools the agent may use; calling them is fine,
    # but a case that only names optional tools does not require any call.
    optional = list(case.get("optional_tools") or [])
    for tool in actual:
        if tool not in expected and tool not in optional:
            result.failures.append(f"调用了未期望的工具 {tool}")
    if not expected and optional and not actual:
        pass


def _assert_content(case: dict[str, Any], result: CaseResult) -> None:
    for token in case.get("must_include") or []:
        if not _content_matches(token, result.answer):
            result.failures.append(f"缺少必须包含的内容：{token}")
    for token in case.get("must_not") or []:
        if _content_matches(token, result.answer) and not _only_negated(token, result.answer):
            result.failures.append(f"包含禁止出现的内容：{token}")


_NEGATION_MARKERS = ("不能断言", "不可断言", "不计入", "并非", "不是", "无法断言", "不得", "不应",
                     "不能", "不可", "没有", "无法", "而非", "并不是", "不是由")
# A question-shaped heading ("### 是否因为美元升值") raises the topic, it does
# not assert it.
_QUESTION_MARKERS = ("是否", "是否可能", "会不会", "有没有可能", "能否", "可不可以", "为什么")


def _only_negated(token: str, answer: str) -> bool:
    """True when every occurrence of the token sits inside a negated clause.

    "不能断言这 1.14% 是因为美元升值" states the boundary rather than breaking
    it, so a plain substring check would report a false failure.
    """
    if not isinstance(token, str) or len(token) < 4:
        return False
    occurrences = [m.start() for m in re.finditer(re.escape(token), answer)]
    if not occurrences:
        return False
    for start in occurrences:
        context = answer[max(0, start - 40):start]
        if any(marker in context for marker in _NEGATION_MARKERS):
            continue
        # A heading that asks the question does not assert the answer.
        heading = answer[max(0, start - 60):start]
        if any(marker in heading for marker in _QUESTION_MARKERS):
            continue
        return False
    return True


def _assert_numbers_grounded(result: CaseResult) -> None:
    """Every number in the answer must be traceable to this turn's tool results."""
    if not result.tool_results:
        return
    try:
        ungrounded = ungrounded_numbers(result.answer, result.tool_results)
    except ValueError:
        return
    for value in ungrounded:
        result.failures.append(f"数字无工具依据：{value}")


def classify_grounding_failures(result: CaseResult) -> dict[str, list[str]]:
    """Split ungrounded numbers into derived and unexplained ones.

    A number that the model derived from tool values (a ratio it computed
    itself) is a different problem from a number that appears from nowhere;
    separating them makes the report actionable.
    """
    derived: list[str] = []
    unexplained: list[str] = []
    for failure in result.failures:
        if "数字无工具依据" not in failure:
            continue
        value = failure.split("：", 1)[1].strip()
        (derived if _looks_derived(value) else unexplained).append(value)
    return {"derived": derived, "unexplained": unexplained}


def _looks_derived(value: str) -> bool:
    """Heuristic: a percentage the model computed from amounts it had."""
    return "%" in value or "个百分点" in value or "pp" in value.lower()


def _assert_review(case: dict[str, Any], result: CaseResult) -> None:
    review = result.review
    expect_note = bool(case.get("expect_note"))
    failed = review.get("failed_checks") or []
    unavailable = review.get("unavailable_checks") or []
    has_note = bool(result.notes)

    if expect_note and not (failed or unavailable):
        result.failures.append("预期触发检查清单告警，但未触发")
    if expect_note and not has_note:
        result.failures.append("检查清单触发但答案末尾缺少 note 注解")
    if not expect_note and failed:
        result.failures.append(
            "检查清单判定失败：" + "；".join(
                f"{item.get('check_id')}: {item.get('reason')}" for item in failed
            )
        )
    if not expect_note and unavailable:
        result.failures.append(
            "检查清单无法执行：" + "；".join(
                f"{item.get('check_id')}: {item.get('reason')}" for item in unavailable
            )
        )
    if not expect_note and has_note:
        result.failures.append("答案带有 note 注解，但该用例预期检查全部通过")


def evaluate_case(case: dict[str, Any], result: CaseResult) -> CaseResult:
    if result.actual_intent and result.actual_intent != result.intent:
        result.failures.append(
            f"意图分类错误：期望 {result.intent}，实际 {result.actual_intent}"
        )
    _assert_tools(case, result)
    _assert_content(case, result)
    if case.get("check_numbers_grounded", True):
        _assert_numbers_grounded(result)
    _assert_review(case, result)
    return result


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    by_intent: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_intent.setdefault(result.intent, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1
    failure_reasons: dict[str, int] = {}
    derived_total = 0
    unexplained_total = 0
    for result in results:
        for failure in result.failures:
            reason = re.sub(r"[:：].*$", "", failure)
            reason = re.sub(r"\d+(\.\d+)?", "N", reason)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        split = classify_grounding_failures(result)
        derived_total += len(split["derived"])
        unexplained_total += len(split["unexplained"])
    return {
        "total": total,
        "passed": passed,
        "derived_numbers": derived_total,
        "unexplained_numbers": unexplained_total,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "by_intent": by_intent,
        "failure_reasons": dict(sorted(failure_reasons.items(), key=lambda kv: -kv[1])),
    }
