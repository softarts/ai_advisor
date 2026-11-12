"""Factory for the Strands OpenAI-compatible model.

Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from the environment.
"""
from __future__ import annotations
import logging
import os
import sys

# Bootstrap strands SDK before importing it
# This handles the case where the workspace has a local strands/ directory
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
existing = sys.modules.get("strands")
if existing is None or "site-packages" not in (getattr(existing, "__file__", "") or ""):
    def _is_workspace(p: str) -> bool:
        if not p:
            return True
        try:
            return os.path.realpath(p) == os.path.realpath(_WORKSPACE)
        except Exception:
            return False
    
    _orig_path = sys.path[:]
    sys.path = [p for p in sys.path if not _is_workspace(p)]
    _shadowed = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "strands" or k.startswith("strands.")}
    try:
        import strands as _sdk
        sys.modules["_strands_sdk"] = _sdk
    except Exception:
        sys.path = _orig_path
        for k, v in _shadowed.items():
            sys.modules[k] = v
        raise
    finally:
        sys.path = _orig_path

from strands.models.openai import OpenAIModel

# The SDK logs a WARNING for every reasoningContent block stripped from
# multi-turn history (expected behaviour with reasoning models on OpenAI-
# compatible endpoints).  Suppress it so the REPL stays clean.
logging.getLogger("strands.models.openai").setLevel(logging.ERROR)

# OpenAI client configuration
DEFAULT_TIMEOUT = 120.0
MAX_RETRIES = 3
CONNECT_TIMEOUT = 30.0


def get_model() -> OpenAIModel:
    """Return an OpenAIModel configured from environment variables."""
    import httpx

    return OpenAIModel(
        client_args={
            "api_key": os.environ["LLM_API_KEY"],
            "base_url": os.environ["LLM_BASE_URL"],
            "timeout": httpx.Timeout(DEFAULT_TIMEOUT, connect=CONNECT_TIMEOUT),
            "max_retries": MAX_RETRIES,
        },
        model_id=os.environ["LLM_MODEL"],
    )
