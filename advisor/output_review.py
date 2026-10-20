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
_DATE_RE = re.compile(r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|\d{4}年\d{1,2}月(?:\d{1,2}日)?")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)、]\s")
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
                number, category, currency, _ = _parse_number(match)
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
    return value


def _parse_number(match: re.Match) -> tuple[Decimal, str, str | None, Decimal]:
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
    return number, category, currency, quantum


def _matches_evidence(number: Decimal, category: str, currency: str | None, quantum: Decimal,
                      evidence: dict[str, Any]) -> bool:
    if "value" not in evidence or evidence["category"] != category:
        return False
    if currency and evidence.get("currency") != currency:
        return False
    value = evidence["value"]
    if category == "amount" and currency and evidence.get("currency") is None:
        return False
    if category in {"ratio", "pp"}:
        return abs(value).quantize(quantum, rounding=ROUND_HALF_UP) == abs(number).quantize(
            quantum, rounding=ROUND_HALF_UP
        )
    if category == "amount":
        return abs(value).quantize(quantum, rounding=ROUND_HALF_UP) == abs(number).quantize(
            quantum, rounding=ROUND_HALF_UP
        )
    return abs(value) == abs(number)


def check_numeric_provenance(context: CheckContext) -> CheckOutcome:
    evidence = _parse_tool_results(context.tool_results)
    date_matches = list(_DATE_RE.finditer(context.candidate))
    list_markers = list(_LIST_MARKER_RE.finditer(context.candidate))
    matches = [
        match for match in _NUMERIC_RE.finditer(context.candidate)
        if not any(match.start() < date.end() and date.start() < match.end() for date in date_matches)
        and not any(match.start() >= marker.start() and match.end() <= marker.end() for marker in list_markers)
    ]
    date_matches = [
        match for match in date_matches
        if not any(match.start() >= marker.start() and match.end() <= marker.end() for marker in list_markers)
    ]
    citations = []
    replacements = []
    failed_values = []

    for match in matches:
        number, category, currency, quantum = _parse_number(match)
        sources = [
            item["source"] for item in evidence
            if _matches_evidence(number, category, currency, quantum, item)
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
            if item.get("date") == date_value
            or (len(date_value) == 7 and item.get("date", "").startswith(date_value + "-"))
        ]
        source_text = ", ".join(dict.fromkeys(sources)) if sources else "未在工具结果中匹配"
        replacements.append((match.start(), match.end(), f"{match.group()}〔来源：{source_text}〕"))
        citations.append(f"{match.group()}：{source_text}")
        if not sources:
            failed_values.append(match.group())

    annotated = context.candidate
    for start, end, replacement in reversed(replacements):
        annotated = annotated[:start] + replacement + annotated[end:]
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
    content = response.json()["choices"][0]["message"].get("content") or ""
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
    try:
        result: Any = json.loads(raw)
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
