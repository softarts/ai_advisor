"""Prompt helpers — thin re-exports from advisor.router."""
from advisor.router import (
    load_compose_template,
    extract_fixed_reply,
    load_output_review_config,
    INTENT_PROFILES,
)

__all__ = [
    "load_compose_template",
    "extract_fixed_reply",
    "load_output_review_config",
    "INTENT_PROFILES",
]
