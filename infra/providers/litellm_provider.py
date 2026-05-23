from __future__ import annotations

from collections.abc import AsyncIterator
from time import time
from typing import Any
from uuid import uuid4

from litellm import acompletion

from core.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceMessage,
    CompletionUsage,
)
from infra.providers.base import BaseProvider


class LiteLLMProvider(BaseProvider):
    litellm_provider: str
    model_prefix: str | None = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        response = await acompletion(**self._completion_kwargs(request, stream=False))
        return _normalize_completion_response(response, fallback_model=request.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        response = await acompletion(**self._completion_kwargs(request, stream=True))
        async for chunk in response:
            content = _extract_stream_content(chunk)
            if content:
                yield content

    def _completion_kwargs(
        self, request: ChatCompletionRequest, *, stream: bool
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._litellm_model(request.model),
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": stream,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url
        return kwargs

    def _litellm_model(self, model: str) -> str:
        if not self.model_prefix or model.startswith(f"{self.model_prefix}/"):
            return model
        return f"{self.model_prefix}/{model}"


class OpenAIProvider(LiteLLMProvider):
    litellm_provider = "openai"


class AnthropicProvider(LiteLLMProvider):
    litellm_provider = "anthropic"
    model_prefix = "anthropic"


class GeminiProvider(LiteLLMProvider):
    litellm_provider = "gemini"
    model_prefix = "gemini"


class OllamaProvider(LiteLLMProvider):
    litellm_provider = "ollama"
    model_prefix = "ollama"


def _normalize_completion_response(
    response: Any, *, fallback_model: str
) -> ChatCompletionResponse:
    choice = _first_choice(response)
    content = _get(_get(choice, "message", {}), "content", "") or ""
    prompt_tokens = int(_usage_value(response, "prompt_tokens", 0) or 0)
    completion_tokens = int(_usage_value(response, "completion_tokens", 0) or 0)
    total_tokens = int(
        _usage_value(
            response,
            "total_tokens",
            prompt_tokens + completion_tokens,
        )
        or 0
    )

    return ChatCompletionResponse(
        id=_get(response, "id", None) or f"chatcmpl-{uuid4().hex}",
        created=int(_get(response, "created", None) or time()),
        model=fallback_model,
        choices=[
            ChatCompletionChoice(
                index=int(_get(choice, "index", 0) or 0),
                message=ChoiceMessage(content=content),
                finish_reason=_get(choice, "finish_reason", None) or "stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def _extract_stream_content(chunk: Any) -> str | None:
    choice = _first_choice(chunk)
    delta = _get(choice, "delta", {})
    return _get(delta, "content", None)


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices", [])
    return choices[0] if choices else {}


def _usage_value(response: Any, key: str, default: int) -> int:
    return _get(_get(response, "usage", {}), key, default)


def _get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
