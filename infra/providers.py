from collections.abc import AsyncIterator
from typing import Protocol

from core.config import Settings
from core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    build_chat_completion_response,
)


class LLMProvider(Protocol):
    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Return a completion for one chat request."""


class StreamingLLMProvider(Protocol):
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Yield tokens one at a time for a chat request."""


class EchoProvider:
    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        prompt_tokens = sum(len(message.content.split()) for message in request.messages)
        last_user_message = next(
            (
                message.content
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            "",
        )
        content = f"Echo: {last_user_message}" if last_user_message else "Echo:"
        return build_chat_completion_response(
            model=request.model,
            content=content,
            prompt_tokens=prompt_tokens,
        )


class EchoStreamingProvider:
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        last_user_message = next(
            (
                message.content
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            "",
        )
        content = f"Echo: {last_user_message}" if last_user_message else "Echo:"
        for token in content.split():
            yield token


def build_provider(settings: Settings) -> LLMProvider:
    if settings.provider_mode == "echo":
        return EchoProvider()

    raise ValueError(f"Unsupported provider mode: {settings.provider_mode}")


def build_streaming_provider(settings: Settings) -> StreamingLLMProvider:
    if settings.provider_mode == "echo":
        return EchoStreamingProvider()

    raise ValueError(f"Unsupported provider mode: {settings.provider_mode}")
