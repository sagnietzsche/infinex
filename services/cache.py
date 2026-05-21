from __future__ import annotations

import hashlib
import json

import redis.asyncio as aioredis

from core.models import ChatCompletionRequest

_KEY_PREFIX = "llm-gateway:v1:"


def make_cache_key(request: ChatCompletionRequest) -> str:
    payload = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return _KEY_PREFIX + digest


class ResponseCache:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._ttl = ttl_seconds

    async def get(self, key: str) -> list[str] | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, chunks: list[str]) -> None:
        await self._client.set(key, json.dumps(chunks), ex=self._ttl)

    async def close(self) -> None:
        await self._client.aclose()
