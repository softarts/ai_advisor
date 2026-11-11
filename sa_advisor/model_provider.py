"""Factory for the Strands OpenAI-compatible model.

Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from the environment.
"""
from __future__ import annotations
import os
from strands.models.openai import OpenAIModel


def get_model() -> OpenAIModel:
    """Return an OpenAIModel configured from environment variables."""
    return OpenAIModel(
        client_args={
            "api_key": os.environ["LLM_API_KEY"],
            "base_url": os.environ["LLM_BASE_URL"],
        },
        model_id=os.environ["LLM_MODEL"],
    )
