import asyncio
from types import SimpleNamespace

import pytest

from core.config import Settings
from core.models import ChatCompletionRequest, ChatMessage
from infra.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    build_provider,
)


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.2,
        max_tokens=12,
    )


def test_build_provider_uses_provider_setting() -> None:
    provider = build_provider(Settings(provider="openai", openai_api_key="key"))

    assert isinstance(provider, OpenAIProvider)


def test_build_provider_requires_selected_cloud_key() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_provider(Settings(provider="anthropic"))


def test_build_provider_ollama_needs_no_api_key() -> None:
    provider = build_provider(
        Settings(provider="ollama", ollama_base_url="http://localhost:11435")
    )

    assert isinstance(provider, OllamaProvider)


def test_litellm_provider_normalizes_completion(monkeypatch) -> None:
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="chatcmpl-test",
            created=123,
            model="anthropic/test-model",
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content="provider response"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=2,
                completion_tokens=2,
                total_tokens=4,
            ),
        )

    monkeypatch.setattr(
        "infra.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = AnthropicProvider(api_key="anthropic-key")
    response = asyncio.run(provider.complete(_request()))

    assert calls[0]["model"] == "anthropic/test-model"
    assert calls[0]["api_key"] == "anthropic-key"
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert response.id == "chatcmpl-test"
    assert response.model == "test-model"
    assert response.choices[0].message.content == "provider response"
    assert response.usage.total_tokens == 4


def test_litellm_provider_streams_normalized_chunks(monkeypatch) -> None:
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)

        async def chunks():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]
            )

        return chunks()

    monkeypatch.setattr(
        "infra.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    async def collect() -> list[str]:
        provider = OllamaProvider(base_url="http://localhost:11435")
        return [chunk async for chunk in provider.stream(_request())]

    chunks = asyncio.run(collect())

    assert calls[0]["model"] == "ollama/test-model"
    assert calls[0]["api_base"] == "http://localhost:11435"
    assert calls[0]["stream"] is True
    assert chunks == ["hel", "lo"]
