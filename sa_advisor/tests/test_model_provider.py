import os
from unittest.mock import patch

from sa_advisor import model_provider


def test_get_model_uses_sdk_timeout_and_retry_settings():
    with (
        patch.dict(os.environ, {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.example/v1",
            "LLM_MODEL": "test-model",
        }),
        patch.object(model_provider, "OpenAIModel") as openai_model,
    ):
        model_provider.get_model()

    client_args = openai_model.call_args.kwargs["client_args"]
    assert client_args["api_key"] == "test-key"
    assert client_args["base_url"] == "https://llm.example/v1"
    timeout = client_args["timeout"]
    assert timeout.read == model_provider.DEFAULT_TIMEOUT
    assert timeout.connect == model_provider.CONNECT_TIMEOUT
    assert client_args["max_retries"] == model_provider.MAX_RETRIES
    assert "http_client" not in client_args
