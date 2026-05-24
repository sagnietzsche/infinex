from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.health import create_health_router
from main import _ProbeAccessLogFilter
from services.batcher import BatcherStats, DynamicBatcherStats
from services.circuit_breaker import CircuitSnapshot, CircuitState


class StubRedisProbe:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def ping(self) -> None:
        if self.fail:
            raise asyncio.TimeoutError


class StubBatcher:
    def __init__(self, *, queued: int = 0, max_queue_size: int = 10) -> None:
        self.queued = queued
        self.max_queue_size = max_queue_size

    def stats(self) -> BatcherStats:
        return BatcherStats(
            queued_requests=self.queued,
            processed_requests=0,
            processed_batches=0,
            largest_batch_size=0,
            max_batch_size=1,
            max_wait_ms=1,
            max_queue_size=self.max_queue_size,
        )


class StubStreamingBatcher(StubBatcher):
    def stats(self) -> DynamicBatcherStats:
        return DynamicBatcherStats(
            queued_requests=self.queued,
            processed_requests=0,
            processed_batches=0,
            largest_batch_size=0,
            max_batch_size=1,
            max_wait_ms=1,
            max_queue_size=self.max_queue_size,
        )


class StubCircuitBreaker:
    def __init__(
        self,
        states: dict[str, CircuitState] | None = None,
    ) -> None:
        self.states = states or {
            "openai": CircuitState.CLOSED,
            "anthropic": CircuitState.OPEN,
        }
        self.providers = tuple(self.states)

    async def snapshots(self) -> dict[str, CircuitSnapshot]:
        return {
            provider: CircuitSnapshot(
                provider=provider,
                state=state,
                error_rate=0.0,
                errors=0,
                total=0,
            )
            for provider, state in self.states.items()
        }


class SlowCircuitBreaker:
    providers = ("openai", "anthropic")

    async def snapshots(self) -> dict[str, CircuitSnapshot]:
        await asyncio.sleep(1)
        return {}


def _app(
    *,
    redis_probe: StubRedisProbe | None = None,
    batcher: StubBatcher | None = None,
    streaming_batcher: StubStreamingBatcher | None = None,
    circuit_breaker: StubCircuitBreaker | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.started_at_monotonic = time.monotonic() - 5
    app.include_router(
        create_health_router(
            redis_probe=redis_probe or StubRedisProbe(),
            batcher=batcher or StubBatcher(),
            streaming_batcher=streaming_batcher or StubStreamingBatcher(),
            circuit_breaker=circuit_breaker or StubCircuitBreaker(),  # type: ignore[arg-type]
        )
    )
    return app


def test_health_returns_process_liveness_and_uptime() -> None:
    with TestClient(_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 5


def test_ready_returns_200_when_all_dependencies_pass() -> None:
    with TestClient(_app()) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "redis": "ok",
        "queue": "ok",
        "providers": {
            "openai": "closed",
            "anthropic": "open",
        },
    }


def test_ready_returns_503_when_redis_ping_fails() -> None:
    with TestClient(_app(redis_probe=StubRedisProbe(fail=True))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["redis"] == "error"
    assert response.json()["providers"] == {
        "openai": "unknown",
        "anthropic": "unknown",
    }


def test_ready_returns_503_when_queue_is_at_capacity() -> None:
    with TestClient(_app(batcher=StubBatcher(queued=10, max_queue_size=10))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["queue"] == "full"


def test_ready_returns_503_when_no_provider_circuit_is_closed() -> None:
    circuit_breaker = StubCircuitBreaker(
        {
            "openai": CircuitState.OPEN,
            "anthropic": CircuitState.HALF_OPEN,
        }
    )
    with TestClient(_app(circuit_breaker=circuit_breaker)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["providers"] == {
        "openai": "open",
        "anthropic": "half_open",
    }


def test_ready_times_out_provider_circuit_lookup() -> None:
    started_at = time.monotonic()
    with TestClient(_app(circuit_breaker=SlowCircuitBreaker())) as client:  # type: ignore[arg-type]
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["providers"] == {
        "openai": "unknown",
        "anthropic": "unknown",
    }
    assert time.monotonic() - started_at < 0.5


def test_probe_access_log_filter_excludes_probe_paths() -> None:
    access_filter = _ProbeAccessLogFilter()
    health_record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "GET", "/health", "1.1", 200),
        None,
    )
    ready_record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "GET", "/ready?full=1", "1.1", 503),
        None,
    )
    app_record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "GET", "/stats", "1.1", 200),
        None,
    )

    assert not access_filter.filter(health_record)
    assert not access_filter.filter(ready_record)
    assert access_filter.filter(app_record)
