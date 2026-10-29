"""Checkers and helpers for validating generated answers before publication."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable

import pycountry
import httpx

from .llm_settings import DEFAULT_TIMEOUT, MAX_REVIEW_TOKENS


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

# Build a complete code/name registry from ISO 4217 data rather than a
# hand-maintained list of a few currencies. Historic codes are retained so
# older reports can still be checked.
_CURRENCY_ALIASES: dict[str, set[str]] = {}
for _currency_entry in pycountry.currencies:
    _currency_code = _currency_entry.alpha_3.upper()
    _CURRENCY_ALIASES.setdefault(_currency_code, set()).add(_currency_code)
    _CURRENCY_ALIASES.setdefault(_currency_entry.name.upper(), set()).add(_currency_code)
for _alias, _code in {
    "RMB": "CNY", "人民币": "CNY", "人民币元": "CNY", "美元": "USD",
    "美金": "USD", "新币": "SGD", "新元": "SGD", "新加坡元": "SGD",
    "欧元": "EUR", "英镑": "GBP", "日元": "JPY", "港币": "HKD",
    "港元": "HKD", "台币": "TWD", "新台币": "TWD", "韩元": "KRW",
    "加元": "CAD", "澳元": "AUD", "新西兰元": "NZD", "瑞郎": "CHF",
    "令吉": "MYR", "马币": "MYR", "泰铢": "THB", "印度卢比": "INR",
    "巴基斯坦卢比": "PKR",
}.items():
    _CURRENCY_ALIASES.setdefault(_alias, set()).add(_code)
_ALL_CURRENCY_CODES = {code for codes in _CURRENCY_ALIASES.values() for code in codes}
def _codes_named(fragment: str) -> set[str]:
    return {
        code for alias, codes in _CURRENCY_ALIASES.items()
        if fragment in alias for code in codes
    }
_CURRENCY = "(?:" + "|".join(
    re.escape(alias) for alias in sorted(_CURRENCY_ALIASES, key=len, reverse=True)
) + ")"
_CURRENCY_SYMBOL_CODES = {
    "€": {"EUR"}, "₹": {"INR"}, "₩": {"KRW"},
    "₽": {"RUB"}, "₺": {"TRY"}, "₫": {"VND"}, "฿": {"THB"},
    "₱": {"PHP"}, "₪": {"ILS"}, "₦": {"NGN"}, "₴": {"UAH"},
    "₵": {"GHS"}, "₡": {"CRC", "SVC"}, "₲": {"PYG"}, "₸": {"KZT"},
    "₭": {"LAK"}, "₮": {"MNT"}, "₼": {"AZN"}, "₾": {"GEL"},
    "¥": {"JPY", "CNY"}, "₨": _codes_named("RUPEE") | {"INR", "PKR", "NPR", "LKR"},
    "$": _codes_named("DOLLAR"),
    "£": _codes_named("POUND"),
}
# Models sometimes group digits with a narrow no-break space (U+202F) or
# other Unicode spaces instead of a comma, and separate date parts with a
# non-breaking hyphen (U+2011). Both would otherwise split a single figure
# into several unmatchable fragments.
_DATE_SEP = r"[-/]"
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
_DATE_RE = re.compile(
    rf"\d{{4}}{_DATE_SEP}\d{{1,2}}(?:{_DATE_SEP}\d{{1,2}})?"
    r"|\d{4}年\d{1,2}月(?:\d{1,2}日)?"
    r"|(?<!\d)\d{8}(?!\d)"
    r"|(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日"
    # English style: "31 Mar 2025", "30 Apr 2025".
    r"|(?<!\d)\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
)
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:\*{1,2})?\d+(?:\ufe0f\u20e3|[.)、])\s*")
# An index inside a path reference ("composition[2].pp") is structural, not a
# data claim.
_PATH_INDEX_RE = re.compile(r"\[[\d]+\]")
# Markdown headings ("### 2. 收益构成", "**5️⃣ 可选下一步**") and numbered section titles carry a
# structural number, not a data claim.
# The heading number may be followed by punctuation or an emoji decoration
# ("### 2. 收益构成", "### 2\ufe0f\u20e3 规模与资金", "**5️⃣ 可选下一步**").
_HEADING_MARKER_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*|\*{1,2})\d+(?:\.\d+)*(?:\ufe0f\u20e3|[.)])?[^\w\u4e00-\u9fff]*"
)
# Well-known index names carry a number that is part of the name, not a
# figure: "标普 500", "沪深 300".
_INDEX_NAME_RE = re.compile(
    r"(?:标普|标准普尔|道琼斯|纳斯达克|恒生|沪深|中证|日经|德国DAX|富时)"
    r"\s*\d{2,4}"
)
# Window labels ("YTD、1M、3M、6M、1Y、3Y、5Y") name a period, they are not
# figures.
_WINDOW_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,2}\s*[MY]|YTD)(?![A-Za-z0-9])", re.IGNORECASE)
# A quoted sample question ("“2023-01 到 2024-06 的收益是多少？”") illustrates
# what the user may ask; its numbers are not claims about the account.
_QUOTED_SPAN_RE = re.compile(r"[“「『][^”」』]{0,200}[”」』]")
_NUMERIC_RE = re.compile(
    rf"(?<![A-Za-z])(?P<before>{_CURRENCY})?(?![A-Za-z])\s*"
    rf"(?P<symbol>[$€£¥₹₩₽₺₫฿₱₪₦₴₵₡₲₸₭₮₼₾])?\s*"
    rf"(?P<number>{_NUMBER})\s*(?P<scale>万|亿|[kKmM])?\s*"
    rf"(?P<after>{_CURRENCY})?(?![A-Za-z])\s*"
    rf"(?P<symbol_after>[$€£¥₹₩₽₺₫฿₱₪₦₴₵₡₲₸₭₮₼₾])?\s*"
    rf"(?P<unit>%|％|个百分点|个基点|bps|pp|元|块)?(?![A-Za-z])",
    re.IGNORECASE,
)
# A grouped figure whose groups are separated by a Unicode space, such as
# "420<NBSP>878.99", is one number rather than two.
_GROUPED_NUMBER_RE = re.compile(
    r"\d{1,3}(?:[   ]\d{3})+(?:\.\d+)?"
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


def _collect_evidence(value: Any, tool: str, path: str = "$", inherited_currency: str | None = None,
                      base_currency: str | None = None):
    evidence = []
    if isinstance(value, dict):
        currency = value.get("currency", inherited_currency)
        for key, child in value.items():
            if key == "currency":
                continue
            child_path = f"{path}.{key}"
            # A "*_base" field is already expressed in the account's base
            # currency, whatever currency its sibling fields use.
            child_currency = base_currency if key.endswith("_base") else currency
            evidence.extend(_collect_evidence(child, tool, child_path, child_currency, base_currency))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            evidence.extend(_collect_evidence(child, tool, f"{path}[{index}]", inherited_currency, base_currency))
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
            for match in _NUMERIC_RE.finditer(value):
                number, category, currency, _, _ = _parse_number(match)
                path_category = _number_category(path, inherited_currency)
                if category == "plain" and path_category != "plain":
                    category = path_category
                if category not in {"ratio", "pp", "amount"}:
                    continue
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
        base_currency = value.get("currency") if isinstance(value, dict) else None
        evidence.extend(_collect_evidence(value, result.get("tool", "tool"), "$", None, base_currency))
    return evidence


def _parse_number(match: re.Match) -> tuple[Decimal, str, str | None, Decimal, bool]:
    number = Decimal(match.group("number").replace(",", ""))
    scale = (match.group("scale") or "").lower()
    number *= _SCALE.get(scale, Decimal(1))
    unit = (match.group("unit") or "").lower()
    currency_alias = match.group("before") or match.group("after") or ""
    currency_codes = (_CURRENCY_ALIASES.get(currency_alias.upper())
                      or _CURRENCY_ALIASES.get(currency_alias) or set())
    currency = next(iter(currency_codes)) if len(currency_codes) == 1 else None
    symbol = match.group("symbol") or match.group("symbol_after")
    if currency is None and symbol:
        symbol_codes = _CURRENCY_SYMBOL_CODES.get(symbol, set())
        currency = next(iter(symbol_codes)) if len(symbol_codes) == 1 else None
    category = (
        "pp" if unit in {"个百分点", "个基点", "bps", "pp"}
        else "ratio" if unit in {"%", "％"}
        else "amount" if currency_alias or symbol or unit in {"元", "块"}
        else "plain"
    )
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


def _possible_currencies(match: re.Match) -> set[str]:
    """Return the ISO currency codes allowed by an answer-side marker."""
    alias = match.group("before") or match.group("after") or ""
    if alias:
        return (_CURRENCY_ALIASES.get(alias.upper())
                or _CURRENCY_ALIASES.get(alias) or set())
    symbol = match.group("symbol") or match.group("symbol_after")
    if symbol:
        return _CURRENCY_SYMBOL_CODES.get(symbol, set())
    if match.group("unit") in {"元", "块"}:
        # These units are shared by multiple currencies. Let the current
        # tool results resolve them only when exactly one currency matches.
        return _ALL_CURRENCY_CODES
    return set()


def _resolve_currency(match: re.Match, number: Decimal, quantum: Decimal,
                      explicit_sign: bool, evidence: list[dict[str, Any]]) -> str | None:
    """Resolve aliases and context-dependent symbols against matching amounts."""
    _, _, currency, _, _ = _parse_number(match)
    if currency:
        return currency
    possible = _possible_currencies(match)
    if not possible:
        return None
    candidates = {
        item.get("currency") for item in evidence
        if item.get("category") == "amount"
        and item.get("currency") in possible
        and _magnitude_matches(number, "amount", item.get("currency"), quantum, item)
        and (not explicit_sign or not _sign_conflicts(number, item))
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _significant_digits(number: Decimal) -> int:
    """Number of significant digits in a value.

    ``normalize()`` drops leading zeros and trailing zeros alike, so the digit
    count of the normalized value is the number of significant digits.
    """
    digits = number.normalize().as_tuple().digits
    return len(digits) if digits else 1


def _same_magnitude(value: Decimal, number: Decimal, quantum: Decimal) -> bool:
    """Compare magnitudes at the precision the answer actually displayed.

    The answer may restate the tool value with fewer significant digits,
    and it may pad the result with trailing zeros ("420,879.00" and "4.50"
    for tool values of 420878.9896... and 4.4945...). The trailing zeros are
    formatting, so the comparison uses the answer's effective precision, that
    is the precision after removing those zeros.

    Two rules keep this from accepting a wrong figure:
    - the comparison is never coarser than the answer's effective precision,
      so -5.7% cannot match a tool value of -5.63% by collapsing to -6;
    - the answer may not claim more significant digits than the tool value
      carries, so 4.56 does not match a tool value of 4.4945.
    """
    if abs(value) == abs(number):
        return True
    answer_decimals = max(0, -quantum.as_tuple().exponent)
    if quantum > 0 and abs(value).quantize(quantum, rounding=ROUND_HALF_UP) == abs(number).quantize(
        quantum, rounding=ROUND_HALF_UP
    ):
        return True
    # Effective precision of the answer: drop the trailing zeros it padded.
    effective = abs(number).normalize()
    effective_decimals = max(0, -effective.as_tuple().exponent)
    if _significant_digits(number) > _significant_digits(effective):
        return False
    scale = Decimal(1).scaleb(-effective_decimals)
    return abs(value).quantize(scale, rounding=ROUND_HALF_UP) == abs(number).quantize(
        scale, rounding=ROUND_HALF_UP
    )


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
        # Tools report contributions in percentage points ("pp") while the
        # model may write the same figure with a "%" sign. The magnitude is
        # the same number, only the unit label differs, so a ratio in the
        # answer may ground on a pp value and the other way round.
        if {category, evidence_category} == {"ratio", "pp"} and _same_magnitude(value, number, quantum):
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


_CITATION_BLOCK_RE = re.compile(
    r"\n\n数字来源：\n(?:(?:\[\d+\]\s.*)|(?:-\s.*))\n?"
    r"(?:(?:\[\d+\]\s.*)|(?:-\s.*)\n?)*"
)
_INLINE_CITATION_RE = re.compile(r"〔来源：[^〕]*〕")
_INLINE_SOURCE_INDEX_RE = re.compile(r"[ \t]*(?:\[\d+\])+")


def _strip_prior_annotation(candidate: str) -> str:
    """Remove annotations added by a previous review pass so re-checking is idempotent."""
    body = _CITATION_BLOCK_RE.sub("", candidate)
    body = _INLINE_CITATION_RE.sub("", body)
    body = _INLINE_SOURCE_INDEX_RE.sub("", body)
    return body.rstrip()


_UNICODE_GROUP_SPACE_RE = re.compile(r"(?<=\d)[      ](?=\d{3}(?!\d))")
_UNICODE_HYPHENS = {
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-", "−": "-",
    "﹘": "-", "－": "-",
}


def _normalize_separators(text: str) -> str:
    """Rewrite Unicode digit/date separators as ASCII, preserving length.

    Models may group digits with a narrow no-break space ("420 878.99") or
    separate date parts with a non-breaking hyphen ("2025-03-28"). Left as
    is, a single figure is parsed as several fragments and never matches the
    tool value. Every replacement here is one character for one character, so
    the offsets used for annotating the answer stay valid.
    """
    normalized = _UNICODE_GROUP_SPACE_RE.sub(",", text)
    for source, target in _UNICODE_HYPHENS.items():
        normalized = normalized.replace(source, target)
    return normalized


def _structural_markers(text: str) -> list[re.Match]:
    """Spans whose numbers are structural (list ordinals, heading numbers).

    These carry no data claim, so they must not be treated as ungrounded
    numbers nor annotated with a source.
    """
    return (
        list(_LIST_MARKER_RE.finditer(text))
        + list(_HEADING_MARKER_RE.finditer(text))
        + list(_PATH_INDEX_RE.finditer(text))
    )


def _is_structural_number(match: re.Match, markers: list[re.Match]) -> bool:
    """True when the matched number is a list ordinal or heading number.

    The numeric pattern can absorb leading whitespace (for example the
    newline before "2. 收益构成"), so the match span may start before the
    marker. Compare the position of the digits themselves instead.
    """
    digits_start = match.start("number")
    digits_end = match.end("number")
    for marker in markers:
        marker_digits = re.search(r"\d+", marker.group())
        if marker_digits is None:
            continue
        marker_start = marker.start() + marker_digits.start()
        marker_end = marker.start() + marker_digits.end()
        if digits_start >= marker_start and digits_end <= marker_end:
            return True
    return False


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


def _excluded_number_spans(body: str) -> list[tuple[int, int]]:
    """Spans whose numbers are not data claims about the account."""
    spans = []
    spans.extend((m.start(), m.end()) for m in _INDEX_NAME_RE.finditer(body))
    spans.extend((m.start(), m.end()) for m in _WINDOW_LABEL_RE.finditer(body))
    spans.extend((m.start(), m.end()) for m in _QUOTED_SPAN_RE.finditer(body))
    return spans


def ungrounded_numbers(candidate: str, tool_results: list[dict[str, str]]) -> list[str]:
    """Return ratio and currency-amount figures that no tool result supports.

    This is the shared implementation used by both the answer checklist and the
    evaluation runner, so the two always agree on what counts as grounded.
    """
    evidence = _parse_tool_results(tool_results)
    if not evidence:
        return []
    body = _normalize_separators(_strip_prior_annotation(candidate))
    dates = list(_DATE_RE.finditer(body))
    markers = _structural_markers(body)
    excluded = _excluded_number_spans(body)
    ungrounded: list[str] = []
    for match in _NUMERIC_RE.finditer(body):
        if any(match.start() < d.end() and d.start() < match.end() for d in dates):
            continue
        if any(match.start() < end and start < match.end() for start, end in excluded):
            continue
        if _is_structural_number(match, markers):
            continue
        if any(match.start() < end and start < match.end() for start, end in excluded):
            continue
        number, category, currency, quantum, explicit_sign = _parse_number(match)
        if category not in {"ratio", "pp", "amount"}:
            continue
        if category == "amount":
            currency = _resolve_currency(match, number, quantum, explicit_sign, evidence)
            if currency is None:
                ungrounded.append(match.group())
                continue
        if not any(
            _matches_evidence(number, category, currency, quantum, item, explicit_sign)
            for item in evidence
        ):
            ungrounded.append(match.group())
    return ungrounded


def check_numeric_provenance(context: CheckContext) -> CheckOutcome:
    evidence = _parse_tool_results(context.tool_results)
    body = _normalize_separators(_strip_prior_annotation(context.candidate))
    date_matches = list(_DATE_RE.finditer(body))
    structural_markers = _structural_markers(body)
    excluded = _excluded_number_spans(body)
    matches = [
        match for match in _NUMERIC_RE.finditer(body)
        if not any(match.start() < date.end() and date.start() < match.end() for date in date_matches)
        and not _is_structural_number(match, structural_markers)
        and not any(match.start() < end and start < match.end() for start, end in excluded)
        and _parse_number(match)[1] in {"ratio", "pp", "amount"}
    ]
    sources_by_index: dict[str, int] = {}
    replacements = []
    failed_values = []

    for match in matches:
        number, category, currency, quantum, explicit_sign = _parse_number(match)
        if category == "amount":
            currency = _resolve_currency(match, number, quantum, explicit_sign, evidence)
        sources = [
            item["source"] for item in evidence
            if currency is not None or category != "amount"
            if _matches_evidence(number, category, currency, quantum, item, explicit_sign)
        ]
        unique_sources = list(dict.fromkeys(sources))
        if unique_sources:
            indexes = []
            for source in unique_sources:
                if source not in sources_by_index:
                    sources_by_index[source] = len(sources_by_index) + 1
                indexes.append(sources_by_index[source])
            reference = "".join(f"[{index}]" for index in indexes)
            replacements.append((match.start(), match.end(), f"{match.group()} {reference}"))
        else:
            replacements.append(
                (match.start(), match.end(), f"{match.group()}〔来源：未在工具结果中匹配〕")
            )
            failed_values.append(match.group())

    annotated = _apply_replacements(body, replacements)
    if sources_by_index:
        source_lines = [
            f"[{index}] {source}" for source, index in sources_by_index.items()
        ]
        annotated += "\n\n数字来源：\n" + "\n".join(source_lines)
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
        "max_tokens": MAX_REVIEW_TOKENS,
    }
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    response = httpx.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
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
