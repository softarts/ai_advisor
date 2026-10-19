"""Generic parsing helpers for LLM-based final-answer review."""
from __future__ import annotations

import json
from typing import Any


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
