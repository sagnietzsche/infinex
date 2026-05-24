from __future__ import annotations

import asyncio
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from time import time
from typing import Any, Protocol

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from core.models import ChatCompletionRequest

_KEY_PREFIX = "llm-gateway:v1:"
_SEMANTIC_KEY_PREFIX = "llm-gateway:semantic:v1:"
_SEMANTIC_INDEX = "llm-gateway:semantic:v1:index"
_SEMANTIC_KNN_LIMIT = 10

logger = logging.getLogger(__name__)


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


def user_facing_prompt(request: ChatCompletionRequest) -> str:
    return "\n".join(
        message.content
        for message in request.messages
        if message.role == "user" and message.content
    )


def make_semantic_request_signature(request: ChatCompletionRequest) -> str:
    payload = {
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...


class LiteLLMEmbeddingProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    async def embed(self, text: str) -> list[float]:
        from litellm import aembedding

        kwargs: dict[str, Any] = {"model": self._model, "input": [text]}
        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = await aembedding(**kwargs)
        data = _get(response, "data", [])
        if not data:
            raise ValueError("embedding response did not include data")

        embedding = _get(data[0], "embedding", None)
        if embedding is None:
            raise ValueError("embedding response did not include an embedding")
        return [float(value) for value in embedding]


@dataclass(frozen=True)
class SemanticCacheLookup:
    prompt: str
    embedding: list[float] | None = None
    chunks: list[str] | None = None
    similarity: float | None = None

    @property
    def hit(self) -> bool:
        return self.chunks is not None


class ResponseCache:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        *,
        semantic_enabled: bool = False,
        semantic_ttl_seconds: int = 3600,
        semantic_threshold: float = 0.95,
        semantic_embedding_model: str = "text-embedding-3-small",
        semantic_embedding_dimension: int = 1536,
        openai_api_key: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._ttl = ttl_seconds
        self._semantic_enabled = semantic_enabled
        self._semantic_ttl = semantic_ttl_seconds
        self._semantic_threshold = semantic_threshold
        self._semantic_dimension = semantic_embedding_dimension
        self._semantic_available = semantic_enabled
        self._semantic_index_ready = False
        self._semantic_index_lock = asyncio.Lock()
        self._semantic_client: aioredis.Redis | None = None
        self._embedding_provider = embedding_provider or LiteLLMEmbeddingProvider(
            model=semantic_embedding_model,
            api_key=openai_api_key,
        )

        if semantic_enabled:
            self._semantic_client = aioredis.from_url(
                redis_url, decode_responses=False
            )

    async def get(self, key: str) -> list[str] | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, chunks: list[str]) -> None:
        await self._client.set(key, json.dumps(chunks), ex=self._ttl)

    async def lookup_semantic(
        self, request: ChatCompletionRequest
    ) -> SemanticCacheLookup:
        prompt = user_facing_prompt(request)
        if not self._semantic_enabled or not prompt:
            return SemanticCacheLookup(prompt=prompt)

        embedding = await self._embed(prompt)
        if embedding is None:
            return SemanticCacheLookup(prompt=prompt)

        if not await self._ensure_semantic_index():
            return SemanticCacheLookup(prompt=prompt, embedding=embedding)

        assert self._semantic_client is not None
        vector = _vector_blob(embedding)
        now = int(time())
        query = (
            f"(@expires_at:[{now} +inf])=>"
            f"[KNN {_SEMANTIC_KNN_LIMIT} @embedding $embedding AS score]"
        )
        try:
            raw = await self._semantic_client.execute_command(
                "FT.SEARCH",
                _SEMANTIC_INDEX,
                query,
                "PARAMS",
                2,
                "embedding",
                vector,
                "SORTBY",
                "score",
                "RETURN",
                4,
                "chunks",
                "score",
                "expires_at",
                "request_signature",
                "DIALECT",
                2,
            )
        except Exception as exc:
            self._semantic_available = False
            logger.warning(
                "semantic cache lookup disabled after Redis error: %r", exc
            )
            return SemanticCacheLookup(prompt=prompt, embedding=embedding)

        signature = make_semantic_request_signature(request)
        for fields in _search_fields(raw):
            if _decode(fields.get("request_signature")) != signature:
                continue
            if _expired(fields.get("expires_at"), now=now):
                continue

            distance = _float_value(fields.get("score"))
            if distance is None:
                continue
            similarity = 1 - distance
            if similarity < self._semantic_threshold:
                continue

            chunks = _json_list(fields.get("chunks"))
            if chunks is not None:
                return SemanticCacheLookup(
                    prompt=prompt,
                    embedding=embedding,
                    chunks=chunks,
                    similarity=similarity,
                )

        return SemanticCacheLookup(prompt=prompt, embedding=embedding)

    async def set_semantic(
        self,
        key: str,
        request: ChatCompletionRequest,
        chunks: list[str],
        *,
        prompt: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> None:
        if not self._semantic_enabled or not chunks:
            return

        prompt = user_facing_prompt(request) if prompt is None else prompt
        if not prompt:
            return

        if embedding is None:
            generated_embedding = await self._embed(prompt)
            if generated_embedding is None:
                return
            embedding = generated_embedding

        if not await self._ensure_semantic_index():
            return

        assert self._semantic_client is not None
        semantic_key = _semantic_cache_key(
            key=key,
            prompt=prompt,
            signature=make_semantic_request_signature(request),
        )
        expires_at = int(time()) + self._semantic_ttl
        try:
            await self._semantic_client.hset(
                semantic_key,
                mapping={
                    "prompt": prompt,
                    "cache_key": key,
                    "chunks": json.dumps(chunks),
                    "request_signature": make_semantic_request_signature(request),
                    "expires_at": str(expires_at),
                    "embedding": _vector_blob(embedding),
                },
            )
            await self._semantic_client.expire(semantic_key, self._semantic_ttl)
        except Exception as exc:
            self._semantic_available = False
            logger.warning(
                "semantic cache store disabled after Redis error: %r", exc
            )

    async def close(self) -> None:
        await self._client.aclose()
        if self._semantic_client is not None:
            await self._semantic_client.aclose()

    async def _embed(self, prompt: str) -> list[float] | None:
        try:
            embedding = await self._embedding_provider.embed(prompt)
        except Exception as exc:
            logger.warning("semantic cache embedding failed: %r", exc)
            return None

        if len(embedding) != self._semantic_dimension:
            logger.warning(
                "semantic cache embedding dimension mismatch: expected %s, got %s",
                self._semantic_dimension,
                len(embedding),
            )
            return None
        return embedding

    async def _ensure_semantic_index(self) -> bool:
        if not self._semantic_enabled or not self._semantic_available:
            return False
        if self._semantic_index_ready:
            return True

        async with self._semantic_index_lock:
            if self._semantic_index_ready:
                return True
            assert self._semantic_client is not None
            try:
                await self._semantic_client.execute_command(
                    "FT.CREATE",
                    _SEMANTIC_INDEX,
                    "ON",
                    "HASH",
                    "PREFIX",
                    1,
                    _SEMANTIC_KEY_PREFIX,
                    "SCHEMA",
                    "prompt",
                    "TEXT",
                    "NOINDEX",
                    "chunks",
                    "TEXT",
                    "NOINDEX",
                    "cache_key",
                    "TAG",
                    "request_signature",
                    "TAG",
                    "expires_at",
                    "NUMERIC",
                    "embedding",
                    "VECTOR",
                    "HNSW",
                    6,
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    self._semantic_dimension,
                    "DISTANCE_METRIC",
                    "COSINE",
                )
            except ResponseError as exc:
                if "exists" not in str(exc).lower():
                    self._semantic_available = False
                    logger.warning(
                        "semantic cache index creation failed: %r", exc
                    )
                    return False
            except Exception as exc:
                self._semantic_available = False
                logger.warning("semantic cache index creation failed: %r", exc)
                return False

            self._semantic_index_ready = True
            return True


def _semantic_cache_key(*, key: str, prompt: str, signature: str) -> str:
    payload = {"cache_key": key, "prompt": prompt, "signature": signature}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return _SEMANTIC_KEY_PREFIX + digest


def _vector_blob(values: Sequence[float]) -> bytes:
    return array("f", values).tobytes()


def _search_fields(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list) or len(raw) < 2:
        return []

    documents: list[dict[str, object]] = []
    for index in range(2, len(raw), 2):
        field_values = raw[index]
        if not isinstance(field_values, list):
            continue
        fields: dict[str, object] = {}
        for field_index in range(0, len(field_values), 2):
            name = field_values[field_index]
            if not isinstance(name, bytes) or field_index + 1 >= len(field_values):
                continue
            fields[name.decode()] = field_values[field_index + 1]
        documents.append(fields)
    return documents


def _decode(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _float_value(value: object) -> float | None:
    decoded = _decode(value)
    if decoded is None:
        return None
    try:
        return float(decoded)
    except ValueError:
        return None


def _expired(value: object, *, now: int) -> bool:
    decoded = _decode(value)
    if decoded is None:
        return True
    try:
        return int(float(decoded)) <= now
    except ValueError:
        return True


def _json_list(value: object) -> list[str] | None:
    decoded = _decode(value)
    if decoded is None:
        return None
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        return None
    return parsed


def _get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
