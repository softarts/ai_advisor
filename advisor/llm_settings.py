"""Shared LLM request settings.

Kept in one place so every call site uses the same token budget and timeouts.
"""
from __future__ import annotations

# Reasoning models spend hundreds of tokens on their internal reasoning before
# emitting the answer. A cap below that makes the server abort generation with
# 500 "Failed to generate completions", so all requests leave generous room.
MAX_COMPLETION_TOKENS = 2048

# The output reviewer reasons over a whole candidate answer, and a reasoning
# model may spend thousands of tokens before emitting its short JSON verdict.
# Truncating it leaves ``content`` empty and the verdict unparseable.
MAX_REVIEW_TOKENS = 8192

# Completion requests that carry tool schemas or long answers need more time
# than a small classification call.
DEFAULT_TIMEOUT = 120.0
