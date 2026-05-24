from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import time
from uuid import uuid4

import redis.asyncio as aioredis
from redis.exceptions import RedisError


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    circuit_open = True

    def __init__(self, provider: str, state: CircuitState) -> None:
        super().__init__(f"circuit for provider {provider} is {state.value}")
        self.provider = provider
        self.state = state


@dataclass(frozen=True)
class CircuitDecision:
    allowed: bool
    state: CircuitState
    probe: bool = False


@dataclass(frozen=True)
class CircuitSnapshot:
    provider: str
    state: CircuitState
    error_rate: float
    errors: int
    total: int


CHECK_CIRCUIT_LUA = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])

local state = redis.call("HGET", key, "state")
if state == false or state == "CLOSED" then
    return {1, "CLOSED", 0}
end

if state == "OPEN" then
    local opened_until_ms = tonumber(
        redis.call("HGET", key, "opened_until_ms") or "0"
    )
    if now_ms >= opened_until_ms then
        redis.call("HSET", key, "state", "HALF_OPEN", "probe_in_flight", "1")
        return {1, "HALF_OPEN", 1}
    end
    return {0, "OPEN", 0}
end

if state == "HALF_OPEN" then
    local probe_in_flight = redis.call("HGET", key, "probe_in_flight")
    if probe_in_flight == false or probe_in_flight == "0" then
        redis.call("HSET", key, "probe_in_flight", "1")
        return {1, "HALF_OPEN", 1}
    end
    return {0, "HALF_OPEN", 0}
end

return {1, "CLOSED", 0}
""".strip()


class CircuitBreaker:
    def __init__(
        self,
        *,
        redis_url: str,
        providers: tuple[str, ...],
        error_threshold: float = 0.5,
        window_seconds: int = 60,
        cooldown_seconds: int = 30,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        if error_threshold <= 0 or error_threshold > 1:
            raise ValueError(
                "error_threshold must be greater than zero and at most one"
            )
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be greater than zero")

        self._client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._providers = providers
        self._error_threshold = error_threshold
        self._window_seconds = window_seconds
        self._cooldown_seconds = cooldown_seconds

    @property
    def providers(self) -> tuple[str, ...]:
        return self._providers

    async def before_request(self, provider: str) -> CircuitDecision:
        try:
            allowed, state, probe = await self._client.eval(
                CHECK_CIRCUIT_LUA,
                1,
                self._state_key(provider),
                self._now_ms(),
            )
        except (OSError, RedisError, TypeError, ValueError):
            return CircuitDecision(allowed=True, state=CircuitState.CLOSED)
        return CircuitDecision(
            allowed=bool(int(allowed)),
            state=CircuitState(str(state)),
            probe=bool(int(probe)),
        )

    async def record_success(self, provider: str) -> None:
        try:
            await self._record_total(provider)
            state = await self._state(provider)
            if state == CircuitState.HALF_OPEN:
                await self._close(provider)
        except (OSError, RedisError, TypeError, ValueError):
            return

    async def record_failure(self, provider: str) -> None:
        try:
            await self._record_total(provider, error=True)
            state = await self._state(provider)
            if state in {CircuitState.OPEN, CircuitState.HALF_OPEN}:
                await self._open(provider)
                return

            snapshot = await self.snapshot(provider)
            if snapshot.total > 0 and snapshot.error_rate >= self._error_threshold:
                await self._open(provider)
        except (OSError, RedisError, TypeError, ValueError):
            return

    async def snapshot(self, provider: str) -> CircuitSnapshot:
        try:
            state = await self._state(provider)
            total, errors = await self._window_counts(provider)
        except (OSError, RedisError, TypeError, ValueError):
            state = CircuitState.CLOSED
            total = 0
            errors = 0
        error_rate = errors / total if total else 0.0
        return CircuitSnapshot(
            provider=provider,
            state=state,
            error_rate=error_rate,
            errors=errors,
            total=total,
        )

    async def snapshots(self) -> dict[str, CircuitSnapshot]:
        snapshots = await asyncio.gather(
            *(self.snapshot(provider) for provider in self._providers)
        )
        return {
            snapshot.provider: snapshot
            for snapshot in snapshots
        }

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except (OSError, RedisError, TypeError, ValueError):
            return

    async def _record_total(self, provider: str, *, error: bool = False) -> None:
        now_ms = self._now_ms()
        member = f"{now_ms}:{uuid4().hex}"
        pipe = self._client.pipeline()
        pipe.zadd(self._total_key(provider), {member: now_ms})
        pipe.expire(self._total_key(provider), self._window_seconds)
        if error:
            pipe.zadd(self._errors_key(provider), {member: now_ms})
            pipe.expire(self._errors_key(provider), self._window_seconds)
        await pipe.execute()

    async def _window_counts(self, provider: str) -> tuple[int, int]:
        now_ms = self._now_ms()
        start_ms = now_ms - (self._window_seconds * 1000)
        total, errors = await self._client.zcount(
            self._total_key(provider), start_ms, now_ms
        ), await self._client.zcount(
            self._errors_key(provider), start_ms, now_ms
        )
        return int(total), int(errors)

    async def _state(self, provider: str) -> CircuitState:
        raw = await self._client.hgetall(self._state_key(provider))
        state_value = raw.get("state") if raw else None
        if state_value == CircuitState.OPEN.value:
            opened_until_ms = int(raw.get("opened_until_ms") or "0")
            if self._now_ms() >= opened_until_ms:
                await self._client.hset(
                    self._state_key(provider),
                    mapping={
                        "state": CircuitState.HALF_OPEN.value,
                        "probe_in_flight": "0",
                    },
                )
                return CircuitState.HALF_OPEN
        try:
            return CircuitState(str(state_value))
        except ValueError:
            pass
        return CircuitState.CLOSED

    async def _open(self, provider: str) -> None:
        await self._client.hset(
            self._state_key(provider),
            mapping={
                "state": CircuitState.OPEN.value,
                "opened_until_ms": str(
                    self._now_ms() + (self._cooldown_seconds * 1000)
                ),
                "probe_in_flight": "0",
            },
        )

    async def _close(self, provider: str) -> None:
        await self._client.delete(
            self._state_key(provider),
            self._errors_key(provider),
            self._total_key(provider),
        )

    def _state_key(self, provider: str) -> str:
        return f"cb:{provider}:state"

    def _errors_key(self, provider: str) -> str:
        return f"cb:{provider}:errors"

    def _total_key(self, provider: str) -> str:
        return f"cb:{provider}:total"

    def _now_ms(self) -> int:
        return int(time.time() * 1000)
