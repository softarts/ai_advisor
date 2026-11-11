"""Intent router for sa_advisor — Strands-native implementation.

Uses Agent(structured_output_model=IntentResponse) instead of a raw HTTP call.
No agentic loop is triggered: structured_output bypasses tool-use and returns
a validated Pydantic object directly.

Public interface matches common.prompts / advisor.router so the rest of the
codebase can call classify_intent(text, logger, trace_id) -> str unchanged.
"""
from __future__ import annotations

import logging
import time

from pydantic import BaseModel
from strands import Agent
from strands.types.exceptions import StructuredOutputException
from langsmith import traceable

from common.prompts import INTENT_PROFILES, _prompt_path
from .model_provider import get_model

_VALID_INTENTS = set(INTENT_PROFILES)


# ---------------------------------------------------------------------------
# Schema for structured output
# ---------------------------------------------------------------------------

class IntentResponse(BaseModel):
    """Classification result returned by the router LLM."""
    intent: str
    reason: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@traceable(run_type="llm", name="router_llm")
def _call_router_llm(user_text: str, logger: logging.Logger, trace_id: str | None) -> IntentResponse | None:
    """Call the router LLM via Strands structured output; return None on failure."""
    system_prompt = _prompt_path("intent_router.md").read_text(encoding="utf-8")
    agent = Agent(
        model=get_model(),
        system_prompt=system_prompt,
        callback_handler=None,  # silence streaming output
    )
    try:
        result = agent(user_text, structured_output_model=IntentResponse)
        return result.structured_output
    except StructuredOutputException as exc:
        logger.warning({
            "event": "router_structured_output_failed",
            "trace_id": trace_id,
            "error": str(exc),
        })
        return None


def _normalize_intent(value: str | None) -> str | None:
    """Return a valid intent name or None."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if candidate in _VALID_INTENTS:
        return candidate
    # Guard against models copying the placeholder verbatim
    if "|" in candidate or candidate.startswith("<"):
        return None
    return None


def classify_intent(user_text: str, logger: logging.Logger, trace_id: str | None) -> str:
    """Return one of the INTENT_PROFILES keys.

    Falls back to 'unsupported' on any failure so the caller always gets a
    safe, valid intent string.
    """
    for attempt in range(2):
        try:
            response = _call_router_llm(user_text, logger, trace_id)
            if response is None:
                raise ValueError("structured output returned None")

            intent = _normalize_intent(response.intent)
            if intent is None:
                raise ValueError(f"unknown intent value: {response.intent!r}")

            logger.info({
                "event": "router_classified",
                "trace_id": trace_id,
                "intent": intent,
                "reason": response.reason,
            })
            return intent

        except Exception as exc:
            logger.warning({
                "event": "router_fallback_unsupported",
                "trace_id": trace_id,
                "attempt": attempt + 1,
                "error": str(exc),
            })
            if attempt == 0:
                time.sleep(0.5)
                continue
            return "unsupported"

    return "unsupported"
