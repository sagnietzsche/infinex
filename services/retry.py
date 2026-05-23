from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
import asyncio
import json
import logging
import random
from typing import TypeVar

from core.models import ChatCompletionRequest, ChatCompletionResponse
from infra.providers import LLMProvider, StreamingLLMProvider


T = TypeVar("T")

logger = logging.getLogger(__name__)

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryExhaustedError(Exception):
    """Carries the provider error after retryable attempts are exhausted."""

    def __init__(self, original: BaseException, *, retries_attempted: int) -> None:
        super().__init__(str(original))
        self.original = original
        self.retries_attempted = retries_attempted


class RetryPolicy:
    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay_ms: int = 200,
        max_delay_ms: int = 5000,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
        random_float: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to zero")
        if base_delay_ms <= 0:
            raise ValueError("base_delay_ms must be greater than zero")
        if max_delay_ms <= 0:
            raise ValueError("max_delay_ms must be greater than zero")

        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self._sleep = sleep
        self._random_float = random_float

    async def call(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        operation_name: str,
    ) -> T:
        retries_attempted = 0

        while True:
            try:
                result = await operation()
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                if retries_attempted >= self.max_retries:
                    self._log_outcome(
                        operation_name=operation_name,
                        outcome="exhausted",
                        attempt=retries_attempted,
                        delay_ms=0,
                        error=exc,
                    )
                    raise RetryExhaustedError(
                        exc, retries_attempted=retries_attempted
                    ) from exc

                delay_seconds = self._delay_seconds(retries_attempted)
                retries_attempted += 1
                self._log_outcome(
                    operation_name=operation_name,
                    outcome="retrying",
                    attempt=retries_attempted,
                    delay_ms=round(delay_seconds * 1000, 3),
                    error=exc,
                )
                await self._sleep(delay_seconds)
                continue

            if retries_attempted:
                self._log_outcome(
                    operation_name=operation_name,
                    outcome="succeeded",
                    attempt=retries_attempted,
                    delay_ms=0,
                    error=None,
                )
            return result

    async def stream(
        self,
        stream_factory: Callable[[], AsyncIterator[str]],
        *,
        operation_name: str,
    ) -> AsyncIterator[str]:
        retries_attempted = 0

        while True:
            yielded = False
            try:
                async with aclosing(stream_factory()) as stream:
                    async for token in stream:
                        yielded = True
                        yield token
                if retries_attempted:
                    self._log_outcome(
                        operation_name=operation_name,
                        outcome="succeeded",
                        attempt=retries_attempted,
                        delay_ms=0,
                        error=None,
                    )
                return
            except Exception as exc:
                if yielded or not self._is_retryable(exc):
                    raise
                if retries_attempted >= self.max_retries:
                    self._log_outcome(
                        operation_name=operation_name,
                        outcome="exhausted",
                        attempt=retries_attempted,
                        delay_ms=0,
                        error=exc,
                    )
                    raise RetryExhaustedError(
                        exc, retries_attempted=retries_attempted
                    ) from exc

                delay_seconds = self._delay_seconds(retries_attempted)
                retries_attempted += 1
                self._log_outcome(
                    operation_name=operation_name,
                    outcome="retrying",
                    attempt=retries_attempted,
                    delay_ms=round(delay_seconds * 1000, 3),
                    error=exc,
                )
                await self._sleep(delay_seconds)

    def wrap_provider(self, provider: LLMProvider) -> LLMProvider:
        return _RetryingProvider(provider=provider, retry_policy=self)

    def wrap_streaming_provider(
        self, provider: StreamingLLMProvider
    ) -> StreamingLLMProvider:
        return _RetryingStreamingProvider(provider=provider, retry_policy=self)

    def _delay_seconds(self, attempt: int) -> float:
        cap_ms = min(self.max_delay_ms, self.base_delay_ms * (2**attempt))
        return self._random_float(0, cap_ms) / 1000

    def _is_retryable(self, exc: BaseException) -> bool:
        return provider_status_code(exc) in TRANSIENT_STATUS_CODES

    def _log_outcome(
        self,
        *,
        operation_name: str,
        outcome: str,
        attempt: int,
        delay_ms: float,
        error: BaseException | None,
    ) -> None:
        payload = {
            "event": "provider.retry",
            "operation": operation_name,
            "outcome": outcome,
            "attempt": attempt,
            "delay_ms": delay_ms,
        }
        if error is not None:
            payload["status_code"] = provider_status_code(error)
            payload["error"] = repr(error)
        logger.warning(json.dumps(payload, sort_keys=True, default=str))


class _RetryingProvider:
    def __init__(self, *, provider: LLMProvider, retry_policy: RetryPolicy) -> None:
        self._provider = provider
        self._retry_policy = retry_policy

    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        return await self._retry_policy.call(
            lambda: self._provider.complete(request),
            operation_name="provider.complete",
        )


class _RetryingStreamingProvider:
    def __init__(
        self, *, provider: StreamingLLMProvider, retry_policy: RetryPolicy
    ) -> None:
        self._provider = provider
        self._retry_policy = retry_policy

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        async for token in self._retry_policy.stream(
            lambda: self._provider.stream(request),
            operation_name="provider.stream",
        ):
            yield token


def provider_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None
