"""Deterministic assertions and result aggregation for evaluation cases.

Tool use, tool-derived facts, numeric grounding, and fixed replies are checked
in code. Semantic rubric results are supplied by the separate judge and cannot
override a deterministic failure.
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
from advisor.router import extract_fixed_reply, load_compose_template

_NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_GUARDED_NUMBER_RE = re.compile(
    r"(?<![\w])[-+−]?(?:\d{1,3}(?:[, \u00a0\u202f]\d{3})+|\d+)(?:\.\d+)?"
)
_GUARDED_NUMBER_CONTEXT = 50


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
    semantic_review: dict[str, Any] = field(default_factory=dict)
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


def _assert_fixed_reply(case: dict[str, Any], result: CaseResult) -> None:
    if not case.get("expect_fixed_reply"):
        return
    if result.actual_intent != "unsupported":
        result.failures.append(
            f"固定拒答用例未路由到 unsupported（实际：{result.actual_intent or '未记录'}）"
        )
    if result.actual_tools:
        result.failures.append(f"固定拒答用例调用了工具：{result.actual_tools}")
    expected = extract_fixed_reply(load_compose_template("unsupported"))
    if not expected or result.answer != expected:
        result.failures.append("回答与 unsupported 固定回复模板不一致")


def _assert_semantic_review(case: dict[str, Any], result: CaseResult) -> None:
    review = result.semantic_review
    if not review:
        if case.get("semantic_rubric"):
            result.failures.append("语义评审未执行")
        return
    if review.get("status") != "completed":
        result.failures.append(f"语义评审不可用：{review.get('error', '未知原因')}")
        return
    items = review.get("results")
    if not isinstance(items, list):
        result.failures.append("语义评审结果格式无效：缺少 results 列表")
        return
    expected_ids = {
        item.get("id") for item in case.get("semantic_rubric") or []
        if isinstance(item, dict)
    }
    actual_ids = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            result.failures.append("语义评审结果格式无效：结果项缺少字符串 ID")
            continue
        item_id = item["id"]
        if item_id not in expected_ids or item_id in actual_ids:
            result.failures.append(f"语义评审返回未知或重复 rubric ID：{item_id}")
            continue
        actual_ids.add(item_id)
        decision = item.get("decision")
        if decision not in {"pass", "fail", "uncertain"}:
            result.failures.append(f"语义评审返回非法判定：{item_id}={decision!r}")
            continue
        evidence = item.get("evidence", "")
        if not isinstance(evidence, str) or (evidence and evidence not in result.answer):
            result.failures.append(f"语义评审证据不是回答原文：{item_id}")
        elif decision == "pass" and not evidence:
            result.failures.append(f"语义评审通过判定缺少回答证据：{item_id}")
        if decision != "pass":
            result.failures.append(
                f"语义评审 {decision}：{item_id} — {item.get('reason', '')}"
            )
    if actual_ids != expected_ids:
        result.failures.append(
            "语义评审条目不完整：缺少 " + ", ".join(sorted(expected_ids - actual_ids))
        )


def _path_value(value: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(value, list) and part.isdigit():
            index = int(part)
            if index >= len(value):
                raise ValueError(f"工具结果缺少字段 {path}")
            value = value[index]
            continue
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"工具结果缺少字段 {path}")
        value = value[part]
    return value


def _tool_payloads(result: CaseResult, tool_name: str) -> list[Any]:
    payloads = []
    for tool_result in result.tool_results:
        if tool_result.get("tool") != tool_name:
            continue
        try:
            payloads.append(json.loads(tool_result["content"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法解析工具 {tool_name} 的结果：{exc}") from exc
    return payloads


def _numeric_value_matches(text_value: str, expected: float) -> bool:
    normalized = text_value.replace(",", "").replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    normalized = normalized.replace("−", "-")
    try:
        actual = float(normalized)
    except ValueError:
        return False
    decimals = len(normalized.rsplit(".", 1)[1]) if "." in normalized else 0
    return abs(actual - expected) <= (0.5 * (10 ** -decimals) + 1e-9)


def _tool_value_appears(value: Any, answer: str) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return any(
            _numeric_value_matches(match.group(), float(value))
            for match in _GUARDED_NUMBER_RE.finditer(answer)
        )
    if isinstance(value, str):
        return _normalize_text(value) in _normalize_text(answer)
    return False


def _tool_value_appears_near_label(value: Any, label: Any, answer: str) -> bool:
    if not isinstance(label, str) or not label or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    for match in re.finditer(re.escape(label), answer, re.IGNORECASE):
        start = max(0, match.start() - _GUARDED_NUMBER_CONTEXT)
        end = min(len(answer), match.end() + _GUARDED_NUMBER_CONTEXT)
        context = answer[start:end]
        relative_start = match.start() - start
        relative_end = match.end() - start
        candidates = [
            number for number in _GUARDED_NUMBER_RE.finditer(context)
            if number.end() <= relative_start or number.start() >= relative_end
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda number: (
                relative_start - number.end()
                if number.end() <= relative_start
                else number.start() - relative_end
            ),
        )
        if _numeric_value_matches(nearest.group(), float(value)):
            return True
    return False


def _assert_tool_value_assertions(case: dict[str, Any], result: CaseResult) -> None:
    for assertion in case.get("tool_value_assertions") or []:
        tool_name = assertion["tool"]
        try:
            payloads = _tool_payloads(result, tool_name)
            if not payloads:
                raise ValueError(f"未捕获工具 {tool_name} 的结果")
            matched = False
            observed = []
            for payload in payloads:
                try:
                    value = _path_value(payload, assertion["path"])
                    label_path = assertion.get("label_path")
                    label = assertion.get("label")
                    if label_path:
                        label = _path_value(payload, label_path)
                    observed.append((value, label))
                    if (
                        value is None
                        and assertion.get("allow_null")
                        and payload.get("status") != "error"
                    ):
                        matched = True
                        break
                    if label is not None:
                        matches = _tool_value_appears_near_label(value, label, result.answer)
                    else:
                        matches = _tool_value_appears(value, result.answer)
                    if matches:
                        matched = True
                        break
                except ValueError:
                    continue
            if not matched:
                label_detail = f"（标签 {observed[-1][1]!r}）" if observed and observed[-1][1] else ""
                value_detail = observed[-1][0] if observed else f"字段 {assertion['path']} 未返回"
                result.failures.append(
                    f"回答未包含工具 {tool_name} 字段 {assertion['path']} 的值"
                    f"{label_detail}：{value_detail!r}"
                )
        except (KeyError, ValueError) as exc:
            result.failures.append(f"无法执行工具事实断言：{exc}")


def _assert_tool_extrema(case: dict[str, Any], result: CaseResult) -> None:
    for assertion in case.get("tool_extrema_assertions") or []:
        try:
            payloads = _tool_payloads(result, assertion["tool"])
            if not payloads:
                raise ValueError(f"未捕获工具 {assertion['tool']} 的结果")
            rows = _path_value(payloads[0], assertion["list_path"])
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"工具字段 {assertion['list_path']} 不是非空列表")
            candidates = []
            for row in rows:
                exclusion = assertion.get("exclude_if")
                if exclusion and _path_value(row, exclusion["path"]) == exclusion["equals"]:
                    continue
                value = _path_value(row, assertion["value_path"])
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                candidates.append((float(value), _path_value(row, assertion["entity_path"])))
            if not candidates:
                raise ValueError(f"工具字段 {assertion['value_path']} 没有可比较的数字")
            for mode in assertion.get("modes", []):
                if mode not in {"max", "min"}:
                    raise ValueError(f"不支持的极值模式 {mode!r}")
                selected = (max if mode == "max" else min)(candidates, key=lambda item: item[0])
                if not _tool_value_appears(selected[1], result.answer):
                    result.failures.append(
                        f"回答未提及工具数据中的{mode}值对应实体：{selected[1]!r}"
                    )
                if not _tool_value_appears_near_label(
                    selected[0], assertion.get("entity_label", str(selected[1])), result.answer
                ):
                    result.failures.append(
                        f"回答未将工具数据中的{mode}值与实体配对："
                        f"{selected[1]!r} / {selected[0]:g}"
                    )
        except (KeyError, ValueError) as exc:
            result.failures.append(f"无法执行工具极值断言：{exc}")


def _assert_conditional_tool_value_guards(case: dict[str, Any], result: CaseResult) -> None:
    for guard in case.get("conditional_tool_value_guards") or []:
        tool_name = guard["tool"]
        try:
            payloads = _tool_payloads(result, tool_name)
            if not payloads:
                raise ValueError(f"未捕获工具 {tool_name} 的结果")
            for payload in payloads:
                condition = guard["when"]
                if _path_value(payload, condition["path"]) != condition["equals"]:
                    continue
                reported_paths: set[str] = set()
                for forbidden in guard["forbidden_values"]:
                    if forbidden["path"] in reported_paths:
                        continue
                    raw_value = _path_value(payload, forbidden["path"])
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise ValueError(f"工具字段 {forbidden['path']} 不是数字")
                    for label in forbidden["labels"]:
                        for label_match in re.finditer(re.escape(label), result.answer, re.IGNORECASE):
                            if _tool_value_appears_near_label(
                                raw_value, label, result.answer
                            ):
                                result.failures.append(
                                    f"归因数据不一致时仍引用类别贡献：{label} "
                                    f"{raw_value:g}（{forbidden['path']}）"
                                )
                                reported_paths.add(forbidden["path"])
                                break
        except (KeyError, ValueError) as exc:
            result.failures.append(f"无法执行条件工具值检查：{exc}")


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
        preceding_text = answer[:start]
        boundaries = list(re.finditer(r"[。！？!?;；\n]|(?<!\d)\.(?!\d)", preceding_text))
        sentence_start = boundaries[-1].end() if boundaries else 0
        context = answer[max(sentence_start, start - 40):start]
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
    _assert_fixed_reply(case, result)
    _assert_content(case, result)
    _assert_tool_value_assertions(case, result)
    _assert_tool_extrema(case, result)
    _assert_conditional_tool_value_guards(case, result)
    if case.get("check_numbers_grounded", True):
        _assert_numbers_grounded(result)
    _assert_review(case, result)
    _assert_semantic_review(case, result)
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
    semantic_summary = {"cases": 0, "pass": 0, "fail": 0, "uncertain": 0, "unavailable": 0}
    routing_summary = {"cases": 0, "correct": 0, "incorrect": 0, "unrecorded": 0}
    for result in results:
        for failure in result.failures:
            reason = re.sub(r"[:：].*$", "", failure)
            reason = re.sub(r"\d+(\.\d+)?", "N", reason)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        split = classify_grounding_failures(result)
        derived_total += len(split["derived"])
        unexplained_total += len(split["unexplained"])
        routing_summary["cases"] += 1
        if not result.actual_intent:
            routing_summary["unrecorded"] += 1
        elif result.actual_intent == result.intent:
            routing_summary["correct"] += 1
        else:
            routing_summary["incorrect"] += 1
        if result.semantic_review:
            semantic_summary["cases"] += 1
            if result.semantic_review.get("status") != "completed":
                semantic_summary["unavailable"] += 1
            else:
                for item in result.semantic_review.get("results", []):
                    decision = item.get("decision")
                    if decision in {"pass", "fail", "uncertain"}:
                        semantic_summary[decision] += 1
    return {
        "total": total,
        "passed": passed,
        "derived_numbers": derived_total,
        "unexplained_numbers": unexplained_total,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "by_intent": by_intent,
        "failure_reasons": dict(sorted(failure_reasons.items(), key=lambda kv: -kv[1])),
        "semantic_review": semantic_summary,
        "routing": routing_summary,
    }
