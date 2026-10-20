"""Checkers and helpers for validating generated answers before publication."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable

import httpx


@dataclass
class CheckContext:
    intent: str
    question: str
    candidate: str
    tool_results: list[dict[str, str]]
    logger: logging.Logger
    trace_id: str | None
    review_config: dict[str, str] | None = None


@dataclass
class CheckOutcome:
    passed: bool
    reason: str
    note: str = ""
    candidate: str | None = None


CheckPlugin = Callable[[CheckContext], CheckOutcome]

_CURRENCY = r"(?:USD|SGD|EUR|GBP|CAD|AUD|JPY|CNY|HKD|CHF|NZD|RMB)"
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
_DATE_RE = re.compile(
    r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?"
    r"|\d{4}年\d{1,2}月(?:\d{1,2}日)?"
    r"|(?<!\d)\d{8}(?!\d)"
    r"|(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日"
)
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)、]\s")
_YEAR_RE = re.compile(r"(?<![\d.])(19|20)\d{2}(?![\d.])")
_NUMERIC_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<before>{_CURRENCY})?\s*(?P<symbol>[$€£¥])?\s*"
    rf"(?P<number>{_NUMBER})\s*(?P<scale>万|亿|[kKmM])?\s*"
    rf"(?P<after>{_CURRENCY})?\s*(?P<unit>%|％|个百分点|个基点|bps)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_RATIO_KEYS = ("pct", "percent", "return", "rate", "weight", "volatility")
_PP_KEYS = ("_pp", "percentage_point")
_AMOUNT_KEYS = ("amount", "value", "nav", "profit", "pnl", "deposit", "withdrawal", "cash", "contribution")
_SCALE = {"万": Decimal(10_000), "亿": Decimal(100_000_000), "k": Decimal(1_000),
          "m": Decimal(1_000_000)}


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _number_category(path: str, currency: str | None) -> str:
    key = path.lower()
    if any(marker in key for marker in _PP_KEYS):
        return "pp"
    # A bare "pp" field name also denotes percentage points.
    if re.search(r"(?:^|[.\[])\s*pp\s*(?:[.\]]|$)", key):
        return "pp"
    if any(marker in key for marker in _RATIO_KEYS):
        return "ratio"
    if currency or any(marker in key for marker in _AMOUNT_KEYS):
        return "amount"
    return "plain"


def _collect_evidence(value: Any, tool: str, path: str = "$", inherited_currency: str | None = None):
    evidence = []
    if isinstance(value, dict):
        currency = value.get("currency", inherited_currency)
        for key, child in value.items():
            if key == "currency":
                continue
            evidence.extend(_collect_evidence(child, tool, f"{path}.{key}", currency))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            evidence.extend(_collect_evidence(child, tool, f"{path}[{index}]", inherited_currency))
    else:
        number = _decimal(value)
        if number is not None:
            evidence.append({
                "value": number,
                "currency": inherited_currency.upper() if isinstance(inherited_currency, str) else None,
                "category": _number_category(path, inherited_currency),
                "source": f"{tool}{path}",
            })
        elif isinstance(value, str):
            date_matches = list(_DATE_RE.finditer(value))
            for match in date_matches:
                evidence.append({
                    "date": _normalize_date(match.group()),
                    "source": f"{tool}{path}",
                })
            for match in _NUMERIC_RE.finditer(value):
                if any(
                    match.start() < date.end() and date.start() < match.end()
                    for date in date_matches
                ):
                    continue
                number, category, currency, _, _ = _parse_number(match)
                path_category = _number_category(path, inherited_currency)
                if category == "plain" and path_category != "plain":
                    category = path_category
                evidence.append({
                    "value": number,
                    "currency": currency or (
                        inherited_currency.upper() if isinstance(inherited_currency, str) else None
                    ),
                    "category": category,
                    "source": f"{tool}{path}",
                })
    return evidence


def _parse_tool_results(tool_results: list[dict[str, str]]) -> list[dict[str, Any]]:
    evidence = []
    for result in tool_results:
        try:
            value = json.loads(result["content"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"tool result from {result.get('tool', 'unknown')} is invalid JSON") from exc
        evidence.extend(_collect_evidence(value, result.get("tool", "tool")))
    return evidence


def _normalize_date(value: str) -> str:
    value = value.replace("/", "-")
    chinese_date = re.fullmatch(r"(\d{4})年(\d{1,2})月(?:\d{1,2}日)?", value)
    if chinese_date:
        return f"{chinese_date.group(1)}-{int(chinese_date.group(2)):02d}"
    compact = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", value)
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}-{compact.group(3)}"
    month_day = re.fullmatch(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", value)
    if month_day:
        return f"{int(month_day.group(1)):02d}-{int(month_day.group(2)):02d}"
    return value


def _date_matches(answer_date: str, evidence_date: str | None) -> bool:
    """Match a date written in the answer against a date found in tool output."""
    if not evidence_date:
        return False
    if answer_date == evidence_date:
        return True
    # "2025-03" may match "2025-03-31"; "10-09" may match "2026-10-09".
    if len(answer_date) == 7 and evidence_date.startswith(answer_date + "-"):
        return True
    if len(answer_date) == 5 and "-" in evidence_date:
        month_day = evidence_date.split("-", 1)[1]
        if month_day == answer_date:
            return True
    # A day-level answer ("10 月 9 日") may be covered by month-level evidence
    # ("2026-10") as long as the month agrees.
    if len(answer_date) == 5 and len(evidence_date) == 7:
        return evidence_date.endswith(answer_date)
    return False


def _parse_number(match: re.Match) -> tuple[Decimal, str, str | None, Decimal, bool]:
    number = Decimal(match.group("number").replace(",", ""))
    scale = (match.group("scale") or "").lower()
    number *= _SCALE.get(scale, Decimal(1))
    unit = (match.group("unit") or "").lower()
    currency = (match.group("before") or match.group("after") or "").upper() or None
    if match.group("symbol") == "$" and currency is None:
        currency = None
    category = "pp" if unit in {"个百分点", "个基点", "bps"} else "ratio" if unit in {"%", "％"} else "amount" if currency or scale or match.group("symbol") else "plain"
    if unit in {"个基点", "bps"}:
        number /= Decimal(100)
    decimals = len(match.group("number").partition(".")[2])
    significant_digits = len(match.group("number").replace(",", "").replace(".", "").lstrip("+-").lstrip("0")) or 1
    quantum = (
        Decimal(1).scaleb(number.copy_abs().adjusted() - significant_digits + 1)
        if scale and number
        else Decimal(1).scaleb(-decimals)
    )
    if unit in {"个基点", "bps"}:
        quantum /= Decimal(100)
    explicit_sign = match.group("number").startswith(("+", "-"))
    return number, category, currency, quantum, explicit_sign


def _same_magnitude(value: Decimal, number: Decimal, quantum: Decimal) -> bool:
    """Compare magnitudes at the precision the answer actually displayed."""
    if quantum > 0:
        return abs(value).quantize(quantum, rounding=ROUND_HALF_UP) == abs(number).quantize(
            quantum, rounding=ROUND_HALF_UP
        )
    return abs(value) == abs(number)


def _magnitude_matches(number: Decimal, category: str, currency: str | None, quantum: Decimal,
                       evidence: dict[str, Any]) -> bool:
    if "value" not in evidence:
        return False
    evidence_category = evidence["category"]
    if currency and evidence.get("currency") != currency:
        return False
    value = evidence["value"]
    if category == "amount" and currency and evidence.get("currency") is None:
        return False
    if category == "plain" and evidence_category != "plain":
        # A number written without a unit ("盈亏 107,932.36") carries no
        # category of its own; the surrounding prose decides the unit, so it
        # may ground on a tool value of any category with the same magnitude.
        if not _same_magnitude(value, number, quantum):
            return False
        if evidence_category == "amount" and evidence.get("currency") is None:
            return False
        return True
    if evidence_category != category:
        # Some tools report a percentage or ratio as a bare number, and the
        # currency of the account may even be inherited by dimensionless
        # fields. A "%" in the answer is a unit, not a different magnitude, so
        # a ratio written by the model may ground on such a tool value.
        if category == "ratio" and evidence_category in {"plain", "amount"} and _same_magnitude(value, number, quantum):
            return True
        return False
    if category in {"ratio", "pp", "amount"}:
        return _same_magnitude(value, number, quantum)
    return abs(value) == abs(number)


def _sign_conflicts(number: Decimal, evidence: dict[str, Any]) -> bool:
    """True when the magnitudes agree but the direction is opposite."""
    value = evidence.get("value")
    if not isinstance(value, Decimal) or value == 0 or number == 0:
        return False
    return (value > 0) != (number > 0)


def _matches_evidence(number: Decimal, category: str, currency: str | None, quantum: Decimal,
                      evidence: dict[str, Any], explicit_sign: bool) -> bool:
    if not _magnitude_matches(number, category, currency, quantum, evidence):
        return False
    if explicit_sign and _sign_conflicts(number, evidence):
        return False
    return True


_CITATION_BLOCK_RE = re.compile(r"\n\n数字来源：\n(?:-\s.*\n?)+")
_INLINE_CITATION_RE = re.compile(r"〔来源：[^〕]*〕")


def _strip_prior_annotation(candidate: str) -> str:
    """Remove annotations added by a previous review pass so re-checking is idempotent."""
    body = _CITATION_BLOCK_RE.sub("", candidate)
    body = _INLINE_CITATION_RE.sub("", body)
    return body.rstrip()


def _apply_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping span replacements left to right.

    Replacing right to left keeps the original offsets valid, but only when
    spans do not overlap. Sorting by start and applying left to right is safe
    for the non-overlapping spans produced here, and skips any span that
    overlaps an already applied one.
    """
    ordered = sorted(replacements, key=lambda item: item[0])
    result = text
    offset = 0
    last_end = -1
    for start, end, replacement in ordered:
        if start < last_end:
            continue
        result = result[:start + offset] + replacement + result[end + offset:]
        offset += len(replacement) - (end - start)
        last_end = end
    return result


def ungrounded_numbers(candidate: str, tool_results: list[dict[str, str]]) -> list[str]:
    """Return the numbers in the answer that no tool result supports.

    This is the shared implementation used by both the answer checklist and the
    evaluation runner, so the two always agree on what counts as grounded.
    """
    evidence = _parse_tool_results(tool_results)
    if not evidence:
        return []
    body = _strip_prior_annotation(candidate)
    dates = list(_DATE_RE.finditer(body))
    markers = list(_LIST_MARKER_RE.finditer(body))
    ungrounded: list[str] = []
    for match in _NUMERIC_RE.finditer(body):
        if any(match.start() < d.end() and d.start() < match.end() for d in dates):
            continue
        if any(match.start() >= m.start() and match.end() <= m.end() for m in markers):
            continue
        if _YEAR_RE.fullmatch(match.group("number")):
            continue
        number, category, currency, quantum, explicit_sign = _parse_number(match)
        if not any(
            _matches_evidence(number, category, currency, quantum, item, explicit_sign)
            for item in evidence
        ):
            ungrounded.append(match.group())
    return ungrounded


def check_numeric_provenance(context: CheckContext) -> CheckOutcome:
    evidence = _parse_tool_results(context.tool_results)
    body = _strip_prior_annotation(context.candidate)
    date_matches = list(_DATE_RE.finditer(body))
    list_markers = list(_LIST_MARKER_RE.finditer(body))
    matches = [
        match for match in _NUMERIC_RE.finditer(body)
        if not any(match.start() < date.end() and date.start() < match.end() for date in date_matches)
        and not any(match.start() >= marker.start() and match.end() <= marker.end() for marker in list_markers)
        and not _YEAR_RE.fullmatch(match.group("number"))
    ]
    date_matches = [
        match for match in date_matches
        if not any(match.start() >= marker.start() and match.end() <= marker.end() for marker in list_markers)
    ]
    citations = []
    replacements = []
    failed_values = []

    for match in matches:
        number, category, currency, quantum, explicit_sign = _parse_number(match)
        sources = [
            item["source"] for item in evidence
            if _matches_evidence(number, category, currency, quantum, item, explicit_sign)
        ]
        source_text = ", ".join(dict.fromkeys(sources)) if sources else "未在工具结果中匹配"
        replacements.append((match.start(), match.end(), f"{match.group()}〔来源：{source_text}〕"))
        citations.append(f"{match.group()}：{source_text}")
        if not sources:
            failed_values.append(match.group())

    for match in date_matches:
        date_value = _normalize_date(match.group())
        sources = [
            item["source"] for item in evidence
            if _date_matches(date_value, item.get("date"))
        ]
        source_text = ", ".join(dict.fromkeys(sources)) if sources else "未在工具结果中匹配"
        replacements.append((match.start(), match.end(), f"{match.group()}〔来源：{source_text}〕"))
        citations.append(f"{match.group()}：{source_text}")
        if not sources:
            failed_values.append(match.group())

    annotated = _apply_replacements(body, replacements)
    if citations:
        annotated += "\n\n数字来源：\n" + "\n".join(f"- {item}" for item in citations)
    return CheckOutcome(
        passed=not failed_values,
        reason="数字均可追溯至本轮工具结果" if not failed_values else
        "以下数字未在工具结果中找到依据：" + "、".join(failed_values),
        candidate=annotated,
    )


def check_product_policy(context: CheckContext) -> CheckOutcome:
    if context.review_config is None:
        raise ValueError("product-policy checker requires output-review templates")
    payload = {
        "model": os.environ["LLM_MODEL"],
        "messages": [
            {
                "role": "system",
                "content": context.review_config["template"] + "\n\n" + context.review_config["policy"],
            },
            {
                "role": "user",
                "content": json.dumps({
                    "question": context.question,
                    "candidate_answer": context.candidate,
                    "tool_results": context.tool_results,
                }, ensure_ascii=False),
            },
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    response = httpx.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = message.get("content")
    if not content:
        # Reasoning models may return the whole answer in reasoning_content.
        content = message.get("reasoning_content") or ""
    verdict = parse_review_verdict(content)
    return CheckOutcome(
        passed=verdict["decision"] == "pass",
        reason=verdict["reason"],
    )


CHECK_PLUGINS: dict[str, CheckPlugin] = {
    "numeric_provenance": check_numeric_provenance,
    "product_policy": check_product_policy,
}


def parse_review_verdict(raw: str) -> dict[str, str]:
    """Validate the reviewer's machine-readable verdict."""
    from advisor.router import _extract_json_object

    payload_text = _extract_json_object(raw) if raw else None
    if payload_text is None:
        raise ValueError("output reviewer returned no JSON object")
    try:
        result: Any = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError("output reviewer returned invalid JSON") from exc

    if not isinstance(result, dict) or set(result) != {"decision", "reason"}:
        raise ValueError("output reviewer response must contain decision and reason only")
    decision = result["decision"]
    reason = result["reason"]
    if not isinstance(decision, str) or decision not in {"pass", "fail", "uncertain"}:
        raise ValueError("output reviewer returned an unsupported decision")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("output reviewer must provide a non-empty reason")
    return {"decision": decision, "reason": reason.strip()}
