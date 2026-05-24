from __future__ import annotations

import asyncio
import time
from typing import Protocol

import redis.asyncio as aioredis
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from services.batcher import BatcherStats, DynamicBatcherStats
from services.circuit_breaker import CircuitBreaker, CircuitSnapshot
from services.shutdown import ShutdownDrain


REDIS_READY_TIMEOUT_SECONDS = 0.2


class RedisReadinessProbe:
    def __init__(
        self,
        *,
        redis_url: str,
        timeout_seconds: float = REDIS_READY_TIMEOUT_SECONDS,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )

    async def ping(self) -> None:
        await asyncio.wait_for(
            self._client.ping(),
            timeout=self._timeout_seconds,
        )

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except (OSError, RedisError, TypeError, ValueError):
            return


class _Batcher(Protocol):
    def stats(self) -> BatcherStats | DynamicBatcherStats:
        ...


class _RedisProbe(Protocol):
    async def ping(self) -> None:
        ...


def create_health_router(
    *,
    redis_probe: _RedisProbe,
    batcher: _Batcher,
    streaming_batcher: _Batcher,
    circuit_breaker: CircuitBreaker,
    shutdown_drain: ShutdownDrain | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health(request: Request) -> dict[str, object]:
        started_at = getattr(request.app.state, "started_at_monotonic", None)
        uptime_seconds = 0.0
        if isinstance(started_at, int | float):
            uptime_seconds = max(0.0, time.monotonic() - float(started_at))
        return {
            "status": "ok",
            "uptime_seconds": uptime_seconds,
        }

    @router.get("/ready")
    async def ready() -> JSONResponse:
        ready_status = True
        redis_ready = True
        body: dict[str, object] = {"status": "ok"}

        try:
            await redis_probe.ping()
            body["redis"] = "ok"
        except (asyncio.TimeoutError, OSError, RedisError, TypeError, ValueError):
            body["redis"] = "error"
            redis_ready = False
            ready_status = False

        if _queue_is_ready(batcher.stats(), streaming_batcher.stats()):
            body["queue"] = "ok"
        else:
            body["queue"] = "full"
            ready_status = False

        provider_states = (
            await _provider_states(circuit_breaker)
            if redis_ready
            else _unknown_provider_states(circuit_breaker)
        )
        body["providers"] = provider_states
        if not any(state == "closed" for state in provider_states.values()):
            ready_status = False

        if shutdown_drain is not None and shutdown_drain.is_shutting_down:
            body["shutdown"] = "draining"
            ready_status = False

        if not ready_status:
            body["status"] = "unavailable"

        return JSONResponse(
            body,
            status_code=200 if ready_status else 503,
        )

    return router


def _queue_is_ready(
    batcher_stats: BatcherStats | DynamicBatcherStats,
    streaming_stats: BatcherStats | DynamicBatcherStats,
) -> bool:
    return (
        batcher_stats.queued_requests < batcher_stats.max_queue_size
        and streaming_stats.queued_requests < streaming_stats.max_queue_size
    )


async def _provider_states(
    circuit_breaker: CircuitBreaker,
) -> dict[str, str]:
    try:
        snapshots = await asyncio.wait_for(
            circuit_breaker.snapshots(),
            timeout=REDIS_READY_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, OSError, RedisError, TypeError, ValueError):
        return {
            provider: "unknown"
            for provider in getattr(circuit_breaker, "providers", ())
        }
    return {
        provider: _readiness_state(snapshot)
        for provider, snapshot in snapshots.items()
    }


def _unknown_provider_states(circuit_breaker: CircuitBreaker) -> dict[str, str]:
    return {
        provider: "unknown"
        for provider in getattr(circuit_breaker, "providers", ())
    }


def _readiness_state(snapshot: CircuitSnapshot) -> str:
    return snapshot.state.value.lower()
