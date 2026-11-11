"""Factory for the Strands OpenAI-compatible model.

Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from the environment.
"""
from __future__ import annotations
import logging
import os
from strands.models.openai import OpenAIModel

# The SDK logs a WARNING for every reasoningContent block stripped from
# multi-turn history (expected behaviour with reasoning models on OpenAI-
# compatible endpoints).  Suppress it so the REPL stays clean.
logging.getLogger("strands.models.openai").setLevel(logging.ERROR)


def get_model() -> OpenAIModel:
    """Return an OpenAIModel configured from environment variables."""
    return OpenAIModel(
        client_args={
            "api_key": os.environ["LLM_API_KEY"],
            "base_url": os.environ["LLM_BASE_URL"],
        },
        model_id=os.environ["LLM_MODEL"],
    )
