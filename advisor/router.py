"""Intent router node — classifies user questions before the LLM node.

Flow: START → router → llm (data_query / product_info)
                      → fixed_reply (unsupported)

The router calls the LLM once (no tools) and expects a JSON response with a
single ``intent`` field.  The JSON is validated with a strict schema; any
parse or validation failure maps to ``unsupported`` so the user always gets
a safe fixed reply instead of a crash or an ambiguous answer.

No keyword matching is used.  Classification is purely semantic via the LLM.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Intent profile registry
# ---------------------------------------------------------------------------
# Each profile declares:
#   version          – semver string; bump when the compose template changes
#   compose_template – filename under prompts/ loaded for the LLM system prompt
#   allowed_tools    – tool names the LLM may call (empty = no tools passed)
#   policy_checks    – symbolic tags consumed by _llm; reserved for future hooks
# ---------------------------------------------------------------------------

INTENT_PROFILES: dict[str, dict[str, Any]] = {
    "data_query": {
        "version": "1.0",
        "compose_template": "data_query.compose.md",
        "allowed_tools": [
            "get_capabilities",
            "get_portfolio_detail",
            "get_window_summary",
            "get_monthly_returns",
            "get_month_detail",
            "get_attribution",
            "get_yearly_table",
            "get_cash_interest",
            "get_top_contributors",
            "get_risk_measures",
            "get_open_positions",
        ],
        "policy_checks": ["require_tool_before_answer", "no_unverified_numbers"],
    },
    "product_info": {
        "version": "1.0",
        "compose_template": "product_info.compose.md",
        "allowed_tools": ["get_capabilities"],
        "policy_checks": [],
    },
    "unsupported": {
        "version": "1.0",
        "compose_template": "unsupported.compose.md",
        "allowed_tools": [],
        "policy_checks": [],
    },
}

# ---------------------------------------------------------------------------
# JSON schema for router LLM output
# ---------------------------------------------------------------------------
_VALID_INTENTS = set(INTENT_PROFILES)

_ROUTER_SYSTEM_PROMPT = """\
你是一个意图分类器，只负责把用户问题归类到以下三个意图之一，不做任何其他回答。

意图定义：
- data_query    用户询问与投资组合账户、收益、持仓、归因、历史数据等相关的问题，需要查询数据才能回答。
- product_info  用户询问本系统的功能范围、能回答什么、数据来源、工具能力等元问题，无需实际数据。
- unsupported   用户的问题与投资组合助手完全无关，或属于本系统明确不处理的范畴（如炒股建议、洗钱等）。

输出格式（严格 JSON，不含 Markdown 代码块，不含任何额外文字）：
{"intent": "<data_query|product_info|unsupported>", "reason": "<一句话说明分类理由>"}

规则：
- 直接输出结果，严禁输出任何思考过程、推理过程或思维链（Do not output any reasoning/thinking content）。
- intent 只能是 data_query、product_info、unsupported 三个值之一。
- reason 不超过 50 个汉字或英文单词。
- 不得在 JSON 以外输出任何内容。
"""


def _call_router_llm(user_text: str, logger: logging.Logger, trace_id: str | None) -> str:
    """Call the LLM with no tools and return raw response text."""
    payload = {
        "model": os.environ["LLM_MODEL"],
        "messages": [
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
        "max_tokens": 2000,
    }
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    logger.info(json.dumps(
        {"event": "router_llm_request", "trace_id": trace_id, "url": url, "payload": payload},
        ensure_ascii=False,
    ))
    response = None
    for attempt in range(2):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=30)
            logger.info(json.dumps(
                {"event": "router_llm_response", "trace_id": trace_id,
                 "attempt": attempt + 1, "status": response.status_code, "body": response.text},
                ensure_ascii=False,
            ))
            if response.status_code >= 500 and attempt == 0:
                time.sleep(0.5)
                continue
            response.raise_for_status()
            break
        except httpx.RequestError as exc:
            logger.warning(json.dumps(
                {"event": "router_llm_retryable_error", "trace_id": trace_id,
                 "attempt": attempt + 1, "error": str(exc)},
                ensure_ascii=False,
            ))
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
    if response is None:
        raise RuntimeError("Router LLM request did not produce a response")
    message = response.json()["choices"][0]["message"]
    content = message.get("content")
    if not content:
        # Fallback: if model only produced reasoning_content, check if JSON exists in it
        reasoning = message.get("reasoning_content") or ""
        if "{" in reasoning and "}" in reasoning:
            content = reasoning[reasoning.find("{"):reasoning.rfind("}") + 1]
        else:
            content = ""
    return content


def classify_intent(user_text: str, logger: logging.Logger, trace_id: str | None) -> str:
    """Return one of the INTENT_PROFILES keys.

    On any LLM error, JSON parse failure, or schema violation, returns
    ``"unsupported"`` so the caller always gets a safe fallback.
    """
    try:
        raw = _call_router_llm(user_text, logger, trace_id)
        # Strip accidental markdown fences if the model disobeys instructions
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("router response is not a JSON object")
        intent = data.get("intent")
        if intent not in _VALID_INTENTS:
            raise ValueError(f"unknown intent value: {intent!r}")
        logger.info(json.dumps(
            {"event": "router_classified", "trace_id": trace_id,
             "intent": intent, "reason": data.get("reason", "")},
            ensure_ascii=False,
        ))
        return intent
    except Exception as exc:
        logger.warning(json.dumps(
            {"event": "router_fallback_unsupported", "trace_id": trace_id, "error": str(exc)},
            ensure_ascii=False,
        ))
        return "unsupported"


def load_compose_template(intent: str) -> str:
    """Load the compose template for the given intent, returns empty string if missing."""
    profile = INTENT_PROFILES.get(intent, {})
    filename = profile.get("compose_template", "")
    if not filename:
        return ""
    path = Path(__file__).resolve().parent.parent / "prompts" / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

