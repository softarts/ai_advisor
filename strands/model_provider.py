"""Factory for the Strands OpenAI-compatible model.

Uses ``_strands_sdk`` (the real strands-agents SDK) rather than ``strands``
to avoid the name conflict with this package's own directory.
"""
from __future__ import annotations
import os
from _strands_sdk.models.openai import OpenAIModel


def get_model() -> OpenAIModel:
    """Return an OpenAIModel configured from environment variables.

    Requires: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
    """
    return OpenAIModel(
        client_args={
            "api_key": os.environ["LLM_API_KEY"],
            "base_url": os.environ["LLM_BASE_URL"],
        },
        model_id=os.environ["LLM_MODEL"],
    )
