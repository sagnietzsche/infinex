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
from services.circuit_breaker import (
    CircuitDecision,
    CircuitSnapshot,
    CircuitState,
)
from services.batcher import AsyncRequestBatcher, _StreamItem
from services.provider_router import (
    AllProvidersFailedError,
    ProviderRoute,
    ProviderRouter,
    ProviderStreamMetadata,
    provider_used,
)
from services.retry import RetryPolicy


class ProviderHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned {status_code}")
        self.status_code = status_code


class CircuitOpenError(Exception):
    circuit_open = True


class RecordingProvider:
    def __init__(
        self,
        *,
        name: str,
        content: str = "ok",
        failures: list[BaseException] | None = None,
        stream_tokens: list[str] | None = None,
    ) -> None:
        self.name = name
        self.content = content
        self.failures = failures or []
        self.stream_tokens = stream_tokens or [content]
        self.requests: list[ChatCompletionRequest] = []

    async def complete(
        self, request: ChatCompletionRequest
    ):
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return build_chat_completion_response(
            model=request.model,
            content=self.content,
            prompt_tokens=1,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        for token in self.stream_tokens:
            yield token


class ProviderSelectiveCircuitBreaker:
    def __init__(self, open_provider: str) -> None:
        self.open_provider = open_provider
        self.successes: list[str] = []
        self.failures: list[str] = []

    async def before_request(self, provider: str) -> CircuitDecision:
        if provider == self.open_provider:
            return CircuitDecision(allowed=False, state=CircuitState.OPEN)
        return CircuitDecision(allowed=True, state=CircuitState.CLOSED)

    async def record_success(self, provider: str) -> None:
        self.successes.append(provider)

    async def record_failure(self, provider: str) -> None:
        self.failures.append(provider)


class SnapshotCircuitBreaker:
    async def snapshots(self) -> dict[str, CircuitSnapshot]:
        return {
            "openai": CircuitSnapshot(
                provider="openai",
                state=CircuitState.OPEN,
                error_rate=0.75,
                errors=3,
                total=4,
            )
        }


def _request(model: str = "gpt-4o") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="hello")],
    )


def _router(
    primary: RecordingProvider,
    secondary: RecordingProvider,
) -> ProviderRouter:
    return ProviderRouter(
        routes=[
            ProviderRoute(name="openai", provider=primary),
            ProviderRoute(name="anthropic", provider=secondary),
        ],
        model_mapping={"openai/gpt-4o": "anthropic/claude-opus-4-5"},
    )


def test_router_fails_over_after_primary_retries_are_exhausted(caplog) -> None:
    async def sleep(delay: float) -> None:
        pass

    primary = RecordingProvider(
        name="openai",
        failures=[ProviderHTTPError(503), ProviderHTTPError(503)],
    )
    retrying_primary = RetryPolicy(max_retries=1, sleep=sleep).wrap_provider(primary)
    secondary = RecordingProvider(name="anthropic", content="fallback")
    router = _router(retrying_primary, secondary)

    with caplog.at_level(logging.ERROR, logger="services.provider_router"):
        response = asyncio.run(router.complete(_request()))

    assert response.choices[0].message.content == "fallback"
    assert provider_used(response) == "anthropic"
    assert [request.model for request in primary.requests] == ["gpt-4o", "gpt-4o"]
    assert [request.model for request in secondary.requests] == [
        "claude-opus-4-5"
    ]
    assert '"event": "provider.failover"' in caplog.text


def test_router_skips_open_circuit_without_retrying() -> None:
    primary = RecordingProvider(
        name="openai",
        failures=[CircuitOpenError("circuit open")],
    )
    retrying_primary = RetryPolicy(
        max_retries=3,
        sleep=lambda delay: asyncio.sleep(0),
    ).wrap_provider(primary)
    secondary = RecordingProvider(name="anthropic", content="fallback")
    router = _router(retrying_primary, secondary)

    response = asyncio.run(router.complete(_request()))

    assert response.choices[0].message.content == "fallback"
    assert len(primary.requests) == 1


def test_router_consults_circuit_breaker_before_dispatch() -> None:
    primary = RecordingProvider(name="openai")
    secondary = RecordingProvider(name="anthropic", content="fallback")
    circuit_breaker = ProviderSelectiveCircuitBreaker(open_provider="openai")
    router = ProviderRouter(
        routes=[
            ProviderRoute(name="openai", provider=primary),
            ProviderRoute(name="anthropic", provider=secondary),
        ],
        model_mapping={"openai/gpt-4o": "anthropic/claude-opus-4-5"},
        circuit_breaker=circuit_breaker,
    )

    response = asyncio.run(router.complete(_request()))

    assert response.choices[0].message.content == "fallback"
    assert primary.requests == []
    assert secondary.requests[0].model == "claude-opus-4-5"
    assert circuit_breaker.successes == ["anthropic"]


def test_router_raises_structured_error_when_all_providers_fail() -> None:
    router = _router(
        RecordingProvider(name="openai", failures=[ProviderHTTPError(503)]),
        RecordingProvider(name="anthropic", failures=[ProviderHTTPError(504)]),
    )

    with pytest.raises(AllProvidersFailedError) as raised:
        asyncio.run(router.complete(_request()))

    assert [failure.provider for failure in raised.value.failures] == [
        "openai",
        "anthropic",
    ]
    assert [failure.status_code for failure in raised.value.failures] == [503, 504]


def test_router_stream_emits_provider_metadata_before_tokens() -> None:
    primary = RecordingProvider(
        name="openai",
        failures=[ProviderHTTPError(503)],
    )
    secondary = RecordingProvider(
        name="anthropic",
        stream_tokens=["hel", "lo"],
    )
    router = _router(primary, secondary)

    async def collect() -> list[object]:
        return [chunk async for chunk in router.stream(_request())]

    events = asyncio.run(collect())

    assert events == [
        ProviderStreamMetadata(provider="anthropic", model="claude-opus-4-5"),
        "hel",
        "lo",
    ]
    assert secondary.requests[0].model == "claude-opus-4-5"


class UnusedStreamingBatcher:
    async def submit(
        self, request: ChatCompletionRequest, trace_id: str | None = None
    ) -> _StreamItem:
        raise AssertionError("streaming batcher should not be used")


class MetadataStreamingBatcher:
    async def submit(
        self, request: ChatCompletionRequest, trace_id: str | None = None
    ) -> _StreamItem:
        item = _StreamItem(
            request=request,
            trace_id=trace_id or "trace",
            enqueued_at=0,
        )
        await item.response_channel.put(
            ProviderStreamMetadata(
                provider="anthropic",
                model="claude-opus-4-5",
            )
        )
        await item.response_channel.put("hel")
        await item.response_channel.put("lo")
        await item.response_channel.put(None)
        return item


def test_chat_completion_response_includes_provider_used_header() -> None:
    router = _router(
        RecordingProvider(name="openai", failures=[ProviderHTTPError(503)]),
        RecordingProvider(name="anthropic", content="fallback"),
    )
    app = FastAPI()
    app.include_router(
        create_router(
            AsyncRequestBatcher(
                provider=router,
                max_batch_size=1,
                max_wait_ms=1,
            ),
            UnusedStreamingBatcher(),
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["x-provider-used"] == "anthropic"
    assert response.json()["choices"][0]["message"]["content"] == "fallback"


def test_health_response_exposes_circuit_state() -> None:
    app = FastAPI()
    app.include_router(
        create_router(
            AsyncRequestBatcher(
                provider=RecordingProvider(name="unused"),
                max_batch_size=1,
                max_wait_ms=1,
            ),
            UnusedStreamingBatcher(),
            circuit_breaker=SnapshotCircuitBreaker(),
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "circuits": {
            "openai": {
                "state": "OPEN",
                "error_rate": 0.75,
                "errors": 3,
                "total": 4,
            }
        },
    }


def test_streaming_response_includes_provider_used_header_and_served_model() -> None:
    app = FastAPI()
    app.include_router(
        create_router(
            AsyncRequestBatcher(
                provider=RecordingProvider(name="unused"),
                max_batch_size=1,
                max_wait_ms=1,
            ),
            MetadataStreamingBatcher(),
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["x-provider-used"] == "anthropic"
    assert '"model": "claude-opus-4-5"' in response.text


def test_chat_completion_returns_503_when_all_providers_fail() -> None:
    router = _router(
        RecordingProvider(name="openai", failures=[ProviderHTTPError(503)]),
        RecordingProvider(name="anthropic", failures=[ProviderHTTPError(504)]),
    )
    app = FastAPI()
    app.include_router(
        create_router(
            AsyncRequestBatcher(
                provider=router,
                max_batch_size=1,
                max_wait_ms=1,
            ),
            UnusedStreamingBatcher(),
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    body = response.json()
    assert response.status_code == 503
    assert body["error"]["code"] == "providers_unavailable"
    assert [failure["provider"] for failure in body["error"]["failures"]] == [
        "openai",
        "anthropic",
    ]
