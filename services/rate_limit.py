from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import redis.asyncio as aioredis


TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_ms = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])

local redis_time = redis.call("TIME")
local now_ms = (tonumber(redis_time[1]) * 1000) + math.floor(tonumber(redis_time[2]) / 1000)

local bucket = redis.call("HMGET", key, "tokens", "updated_at")
local tokens = tonumber(bucket[1])
local updated_at = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
end

if updated_at == nil then
    updated_at = now_ms
end

local elapsed_ms = math.max(0, now_ms - updated_at)
tokens = math.min(capacity, tokens + (elapsed_ms * refill_per_ms))

local allowed = 0
local retry_after_ms = 0

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    local missing = requested - tokens
    retry_after_ms = math.ceil(missing / refill_per_ms)
end

redis.call("HSET", key, "tokens", tokens, "updated_at", now_ms)
redis.call("PEXPIRE", key, ttl_ms)

return { allowed, retry_after_ms }
""".strip()


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int


class RedisTokenBucketRateLimiter:
    def __init__(
        self,
        *,
        redis_url: str,
        capacity: int,
        refill_rate_per_second: float,
        key_prefix: str = "llm-gateway:rate-limit:v1:",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if refill_rate_per_second <= 0:
            raise ValueError("refill_rate_per_second must be greater than zero")

        self._client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._capacity = capacity
        self._refill_per_ms = refill_rate_per_second / 1000
        self._ttl_ms = max(
            1000,
            math.ceil((capacity / refill_rate_per_second) * 2000),
        )
        self._key_prefix = key_prefix

    async def check(
        self,
        api_key: str,
        *,
        capacity: int | None = None,
        refill_rate_per_second: float | None = None,
    ) -> RateLimitResult:
        capacity = capacity or self._capacity
        refill_per_ms = (
            self._refill_per_ms
            if refill_rate_per_second is None
            else refill_rate_per_second / 1000
        )
        ttl_ms = max(
            1000,
            math.ceil((capacity / (refill_per_ms * 1000)) * 2000),
        )
        allowed, retry_after_ms = await self._client.eval(
            TOKEN_BUCKET_LUA,
            1,
            self._redis_key(api_key),
            capacity,
            refill_per_ms,
            1,
            ttl_ms,
        )
        is_allowed = bool(int(allowed))
        return RateLimitResult(
            allowed=is_allowed,
            retry_after_seconds=(
                0
                if is_allowed
                else max(1, math.ceil(int(retry_after_ms) / 1000))
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _redis_key(self, api_key: str) -> str:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return self._key_prefix + digest
