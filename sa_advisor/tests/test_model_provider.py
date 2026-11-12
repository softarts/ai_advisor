import os
from unittest.mock import patch

from sa_advisor import model_provider


def test_get_model_uses_async_http_client():
    with (
        patch.dict(os.environ, {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.example/v1",
            "LLM_MODEL": "test-model",
        }),
        patch("httpx.AsyncClient") as async_client,
        patch.object(model_provider, "OpenAIModel") as openai_model,
    ):
        model_provider.get_model()

    client_args = openai_model.call_args.kwargs["client_args"]
    assert client_args["http_client"] is async_client.return_value
