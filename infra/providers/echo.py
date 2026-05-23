from collections.abc import AsyncIterator

from core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    build_chat_completion_response,
)
from infra.providers.base import BaseProvider


class EchoProvider(BaseProvider):
    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        prompt_tokens = sum(len(message.content.split()) for message in request.messages)
        content = _echo_content(request)
        return build_chat_completion_response(
            model=request.model,
            content=content,
            prompt_tokens=prompt_tokens,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        for token in _echo_content(request).split():
            yield token


class EchoStreamingProvider(EchoProvider):
    pass


def _echo_content(request: ChatCompletionRequest) -> str:
    last_user_message = next(
        (
            message.content
            for message in reversed(request.messages)
            if message.role == "user"
        ),
        "",
    )
    return f"Echo: {last_user_message}" if last_user_message else "Echo:"
