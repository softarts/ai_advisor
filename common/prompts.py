"""Intent profile registry and prompt-file helpers.

Agent-library-agnostic: pure file I/O, no LLM calls, no advisor/ imports.
Both the LangGraph advisor and sa_advisor read from this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Intent profile registry
# ---------------------------------------------------------------------------
# Each profile declares:
#   version          – semver string; bump when the compose template changes
#   compose_template – filename under prompts/ loaded for the LLM system prompt
#   allowed_tools    – tool names the LLM may call (empty = no tools passed)
# ---------------------------------------------------------------------------

INTENT_PROFILES: dict[str, dict[str, Any]] = {
    "data_query": {
        "version": "1.3",
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
    },
    "product_info": {
        "version": "1.1",
        "compose_template": "product_info.compose.md",
        "allowed_tools": ["get_capabilities"],
    },
    "unsupported": {
        "version": "1.0",
        "compose_template": "unsupported.compose.md",
        "allowed_tools": [],
    },
}

_VALID_INTENTS = set(INTENT_PROFILES)


def _prompt_path(filename: str) -> Path:
    """Resolve a filename relative to the top-level prompts/ directory."""
    return Path(__file__).resolve().parent.parent / "prompts" / filename


def load_compose_template(intent: str) -> str:
    """Load the compose template for the given intent; returns empty string if missing."""
    profile = INTENT_PROFILES.get(intent, {})
    filename = profile.get("compose_template", "")
    if not filename:
        return ""
    path = _prompt_path(filename)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def extract_fixed_reply(template: str) -> str:
    """Return the body of a fixed-reply template, excluding front matter and headings."""
    lines = template.splitlines()
    reply_lines = []
    in_front_matter = False
    past_front_matter = False
    for line in lines:
        if line.strip() == "---":
            if not past_front_matter:
                in_front_matter = not in_front_matter
                if not in_front_matter:
                    past_front_matter = True
            continue
        if not past_front_matter:
            continue
        if not reply_lines and (line.startswith("#") or not line.strip()):
            continue
        reply_lines.append(line)
    return "\n".join(reply_lines).strip()


def load_output_review_config(intent: str) -> dict[str, str] | None:
    """Load optional review and fail-closed templates using the intent convention."""
    profile = INTENT_PROFILES.get(intent)
    if not profile:
        return None
    review_path = _prompt_path(f"{intent}.review.md")
    failure_path = _prompt_path(f"{intent}.review_failed.md")
    if not review_path.exists() and not failure_path.exists():
        return None
    if not review_path.is_file() or not failure_path.is_file():
        raise FileNotFoundError(f"Incomplete output-review templates for intent {intent!r}")
    return {
        "template": review_path.read_text(encoding="utf-8"),
        "policy": _prompt_path(profile["compose_template"]).read_text(encoding="utf-8"),
        "failure_reply": failure_path.read_text(encoding="utf-8").strip(),
    }
