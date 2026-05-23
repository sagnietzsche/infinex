from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol

from core.models import ChatCompletionRequest, ChatCompletionResponse


class LLMProvider(Protocol):
    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Return a completion for one chat request."""


class StreamingLLMProvider(Protocol):
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Yield tokens one at a time for a chat request."""


class BaseProvider(ABC):
    @abstractmethod
    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Return a normalized chat completion response."""

    @abstractmethod
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Yield normalized text chunks for a streaming chat completion."""
