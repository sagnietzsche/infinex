from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import create_router
from core.models import (
    ChatCompletionRequest,
    ChatMessage,
    build_chat_completion_response,
)
from services.batcher import _StreamItem
from services.retry import RetryExhaustedError, RetryPolicy


class ProviderHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned {status_code}")
        self.status_code = status_code


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])


def test_retry_policy_retries_transient_errors_before_success(caplog) -> None:
    sleeps: list[float] = []
    calls = 0
    caplog.set_level(logging.WARNING, logger="services.retry")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async def complete():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderHTTPError(503)
        return build_chat_completion_response(
            model="test",
            content="ok",
            prompt_tokens=1,
        )

    policy = RetryPolicy(
        max_retries=3,
        base_delay_ms=200,
        max_delay_ms=5000,
        sleep=sleep,
        random_float=lambda low, high: high,
    )

    response = asyncio.run(
        policy.call(complete, operation_name="provider.complete")
    )

    assert response.choices[0].message.content == "ok"
    assert calls == 3
    assert sleeps == [0.2, 0.4]
    assert {record.levelno for record in caplog.records} == {logging.WARNING}
    assert '"outcome": "retrying"' in caplog.text
    assert '"outcome": "succeeded"' in caplog.text


def test_retry_policy_does_not_retry_client_errors() -> None:
    calls = 0

    async def complete():
        nonlocal calls
        calls += 1
        raise ProviderHTTPError(400)

    policy = RetryPolicy(
        max_retries=3,
        sleep=lambda delay: asyncio.sleep(0),
    )

    with pytest.raises(ProviderHTTPError):
        asyncio.run(policy.call(complete, operation_name="provider.complete"))

    assert calls == 1


def test_retry_policy_raises_retry_exhausted_with_attempt_count() -> None:
    calls = 0

    async def sleep(delay: float) -> None:
        pass

    async def complete():
        nonlocal calls
        calls += 1
        raise ProviderHTTPError(429)

    policy = RetryPolicy(max_retries=2, sleep=sleep)

    with pytest.raises(RetryExhaustedError) as raised:
        asyncio.run(policy.call(complete, operation_name="provider.complete"))

    assert calls == 3
    assert raised.value.retries_attempted == 2
    assert isinstance(raised.value.original, ProviderHTTPError)


def test_retry_policy_retries_stream_before_first_token() -> None:
    calls = 0

    async def sleep(delay: float) -> None:
        pass

    def stream_factory() -> AsyncIterator[str]:
        async def stream():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderHTTPError(500)
            yield "ok"

        return stream()

    async def collect() -> list[str]:
        policy = RetryPolicy(max_retries=1, sleep=sleep)
        return [
            chunk
            async for chunk in policy.stream(
                stream_factory,
                operation_name="provider.stream",
            )
        ]

    assert asyncio.run(collect()) == ["ok"]
    assert calls == 2


def test_retry_policy_does_not_retry_stream_after_first_token() -> None:
    calls = 0

    def stream_factory() -> AsyncIterator[str]:
        async def stream():
            nonlocal calls
            calls += 1
            yield "first"
            raise ProviderHTTPError(503)

        return stream()

    async def consume() -> None:
        policy = RetryPolicy(max_retries=3, sleep=lambda delay: asyncio.sleep(0))
        iterator = policy.stream(
            stream_factory,
            operation_name="provider.stream",
        )
        assert await anext(iterator) == "first"
        with pytest.raises(ProviderHTTPError):
            await anext(iterator)

    asyncio.run(consume())
    assert calls == 1


class FailingBatcher:
    async def submit(
        self, request: ChatCompletionRequest, trace_id: str | None = None
    ):
        raise RetryExhaustedError(
            ProviderHTTPError(503),
            retries_attempted=2,
        )


class FailingStreamingBatcher:
    async def submit(
        self, request: ChatCompletionRequest, trace_id: str | None = None
    ) -> _StreamItem:
        item = _StreamItem(
            request=request,
            trace_id=trace_id or "trace",
            enqueued_at=0,
        )
        await item.response_channel.put(
            RetryExhaustedError(
                ProviderHTTPError(504),
                retries_attempted=3,
            )
        )
        await item.response_channel.put(None)
        return item


def test_retry_exhausted_header_is_returned_for_non_streaming_failure() -> None:
    app = FastAPI()
    app.include_router(create_router(FailingBatcher(), FailingStreamingBatcher()))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 503
    assert response.headers["x-retries-attempted"] == "2"


def test_retry_exhausted_header_is_returned_for_streaming_pre_token_failure() -> None:
    app = FastAPI()
    app.include_router(create_router(FailingBatcher(), FailingStreamingBatcher()))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 504
    assert response.headers["x-retries-attempted"] == "3"
