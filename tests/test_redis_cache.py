from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import ChatCompletionRequest, ChatMessage
from services.cache import (
    ResponseCache,
    SemanticCacheLookup,
    make_cache_key,
    make_semantic_request_signature,
    user_facing_prompt,
)


def _request(prompt: str, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=prompt)], **kwargs
    )


class MakeCacheKeyTests(unittest.TestCase):
    def test_same_request_produces_same_key(self) -> None:
        req = _request("hello")
        self.assertEqual(make_cache_key(req), make_cache_key(req))

    def test_different_prompts_produce_different_keys(self) -> None:
        self.assertNotEqual(make_cache_key(_request("hello")), make_cache_key(_request("world")))

    def test_stream_flag_excluded_from_key(self) -> None:
        streaming = _request("hello", stream=True)
        non_streaming = _request("hello", stream=False)
        self.assertEqual(make_cache_key(streaming), make_cache_key(non_streaming))

    def test_model_included_in_key(self) -> None:
        req_a = ChatCompletionRequest(
            model="gpt-4", messages=[ChatMessage(role="user", content="hi")]
        )
        req_b = ChatCompletionRequest(
            model="gpt-3.5", messages=[ChatMessage(role="user", content="hi")]
        )
        self.assertNotEqual(make_cache_key(req_a), make_cache_key(req_b))

    def test_temperature_included_in_key(self) -> None:
        self.assertNotEqual(
            make_cache_key(_request("hi", temperature=0.0)),
            make_cache_key(_request("hi", temperature=1.0)),
        )

    def test_key_has_expected_prefix(self) -> None:
        key = make_cache_key(_request("hello"))
        self.assertTrue(key.startswith("llm-gateway:v1:"))

    def test_user_facing_prompt_excludes_system_messages(self) -> None:
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="answer like a pirate"),
                ChatMessage(role="user", content="hello"),
                ChatMessage(role="assistant", content="old answer"),
                ChatMessage(role="user", content="world"),
            ]
        )

        self.assertEqual(user_facing_prompt(req), "hello\nworld")


class StaticEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class ResponseCacheTests(unittest.IsolatedAsyncioTestCase):
    def _make_cache(self) -> tuple[ResponseCache, MagicMock]:
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=None)
        mock_client.set = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch("services.cache.aioredis.from_url", return_value=mock_client):
            cache = ResponseCache(redis_url="redis://localhost:6379", ttl_seconds=60)

        return cache, mock_client

    async def test_get_returns_none_on_cache_miss(self) -> None:
        cache, mock_client = self._make_cache()
        mock_client.get.return_value = None

        result = await cache.get("some-key")
        self.assertIsNone(result)
        mock_client.get.assert_called_once_with("some-key")

    async def test_get_returns_deserialized_chunks_on_hit(self) -> None:
        cache, mock_client = self._make_cache()
        mock_client.get.return_value = '["hello", " ", "world"]'

        result = await cache.get("some-key")
        self.assertEqual(result, ["hello", " ", "world"])

    async def test_set_serializes_and_stores_with_ttl(self) -> None:
        cache, mock_client = self._make_cache()

        await cache.set("some-key", ["tok1", "tok2"])

        mock_client.set.assert_called_once_with(
            "some-key", '["tok1", "tok2"]', ex=60
        )

    async def test_close_calls_aclose(self) -> None:
        cache, mock_client = self._make_cache()
        await cache.close()
        mock_client.aclose.assert_called_once()

    async def test_flush_deletes_exact_and_semantic_cache_keys(self) -> None:
        cache, mock_client = self._make_cache()
        mock_client.delete = AsyncMock(side_effect=[2, 1])
        calls: list[tuple[str | None, int | None]] = []

        async def scan_iter(match: str | None = None, count: int | None = None):
            calls.append((match, count))
            if match == "llm-gateway:v1:*":
                yield "llm-gateway:v1:a"
                yield "llm-gateway:v1:b"
            if match == "llm-gateway:semantic:v1:*":
                yield "llm-gateway:semantic:v1:a"

        mock_client.scan_iter = scan_iter

        deleted = await cache.flush()

        self.assertEqual(deleted, 3)
        self.assertEqual(
            calls,
            [
                ("llm-gateway:v1:*", 500),
                ("llm-gateway:semantic:v1:*", 500),
            ],
        )
        mock_client.delete.assert_any_await(
            "llm-gateway:v1:a",
            "llm-gateway:v1:b",
        )
        mock_client.delete.assert_any_await("llm-gateway:semantic:v1:a")

    async def test_flush_with_prefix_deletes_only_matching_keys(self) -> None:
        cache, mock_client = self._make_cache()
        mock_client.delete = AsyncMock(return_value=1)
        calls: list[tuple[str | None, int | None]] = []

        async def scan_iter(match: str | None = None, count: int | None = None):
            calls.append((match, count))
            yield "llm-gateway:v1:custom:a"

        mock_client.scan_iter = scan_iter

        deleted = await cache.flush(prefix="llm-gateway:v1:custom:")

        self.assertEqual(deleted, 1)
        self.assertEqual(calls, [("llm-gateway:v1:custom:*", 500)])
        mock_client.delete.assert_awaited_once_with("llm-gateway:v1:custom:a")

    async def test_flush_rejects_non_cache_prefix(self) -> None:
        cache, _ = self._make_cache()

        with self.assertRaises(ValueError):
            await cache.flush(prefix="admin:key:")

    def _make_semantic_cache(self) -> tuple[ResponseCache, MagicMock, MagicMock]:
        exact_client = MagicMock()
        exact_client.get = AsyncMock(return_value=None)
        exact_client.set = AsyncMock()
        exact_client.aclose = AsyncMock()

        semantic_client = MagicMock()
        semantic_client.execute_command = AsyncMock()
        semantic_client.hset = AsyncMock()
        semantic_client.expire = AsyncMock()
        semantic_client.aclose = AsyncMock()

        with patch(
            "services.cache.aioredis.from_url",
            side_effect=[exact_client, semantic_client],
        ):
            cache = ResponseCache(
                redis_url="redis://localhost:6379",
                ttl_seconds=60,
                semantic_enabled=True,
                semantic_ttl_seconds=120,
                semantic_threshold=0.95,
                semantic_embedding_dimension=3,
                embedding_provider=StaticEmbeddingProvider(),
            )

        return cache, exact_client, semantic_client

    async def test_semantic_lookup_returns_hit_above_threshold(self) -> None:
        cache, _, semantic_client = self._make_semantic_cache()
        req = _request("hello")
        signature = make_semantic_request_signature(req)
        semantic_client.execute_command.side_effect = [
            b"OK",
            [
                1,
                b"llm-gateway:semantic:v1:item",
                [
                    b"chunks",
                    b'["Echo: hello"]',
                    b"score",
                    b"0.02",
                    b"expires_at",
                    b"9999999999",
                    b"request_signature",
                    signature.encode(),
                ],
            ],
        ]

        result = await cache.lookup_semantic(req)

        self.assertTrue(result.hit)
        self.assertEqual(result.chunks, ["Echo: hello"])
        self.assertAlmostEqual(result.similarity or 0, 0.98)

    async def test_semantic_lookup_misses_below_threshold(self) -> None:
        cache, _, semantic_client = self._make_semantic_cache()
        req = _request("hello")
        signature = make_semantic_request_signature(req)
        semantic_client.execute_command.side_effect = [
            b"OK",
            [
                1,
                b"llm-gateway:semantic:v1:item",
                [
                    b"chunks",
                    b'["Echo: hello"]',
                    b"score",
                    b"0.20",
                    b"expires_at",
                    b"9999999999",
                    b"request_signature",
                    signature.encode(),
                ],
            ],
        ]

        result = await cache.lookup_semantic(req)

        self.assertFalse(result.hit)
        self.assertEqual(result.embedding, [1.0, 0.0, 0.0])

    async def test_set_semantic_stores_hash_with_ttl(self) -> None:
        cache, _, semantic_client = self._make_semantic_cache()
        semantic_client.execute_command.return_value = b"OK"

        await cache.set_semantic(
            "exact-key",
            _request("hello"),
            ["Echo: hello"],
            prompt="hello",
            embedding=[1.0, 0.0, 0.0],
        )

        semantic_client.hset.assert_called_once()
        mapping = semantic_client.hset.call_args.kwargs["mapping"]
        self.assertEqual(mapping["prompt"], "hello")
        self.assertEqual(mapping["chunks"], '["Echo: hello"]')
        self.assertEqual(len(mapping["embedding"]), 12)
        semantic_client.expire.assert_called_once_with(
            semantic_client.hset.call_args.args[0], 120
        )


class CacheIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end route tests verifying cache hit/miss behaviour."""

    def _make_mock_cache(
        self, cached_value: list[str] | None
    ) -> ResponseCache:
        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            return_value=None if cached_value is None
            else __import__("json").dumps(cached_value)
        )
        mock_client.set = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch("services.cache.aioredis.from_url", return_value=mock_client):
            cache = ResponseCache(redis_url="redis://localhost:6379", ttl_seconds=60)
        cache._mock_client = mock_client  # type: ignore[attr-defined]
        return cache

    async def test_non_streaming_cache_hit_skips_batcher(self) -> None:
        from core.models import ChatCompletionResponse
        from services.batcher import AsyncRequestBatcher
        from api.routes import create_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        cache = self._make_mock_cache(["Echo: hello"])

        mock_batcher = MagicMock(spec=AsyncRequestBatcher)
        mock_batcher.submit = AsyncMock()
        mock_batcher.stats = MagicMock(
            return_value=MagicMock(
                queued_requests=0,
                processed_requests=0,
                processed_batches=0,
                largest_batch_size=0,
                max_batch_size=8,
                max_wait_ms=25,
            )
        )

        mock_streaming_batcher = MagicMock()

        app = FastAPI()
        app.include_router(create_router(mock_batcher, mock_streaming_batcher, cache))

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gateway-echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["x-cache"], "HIT-EXACT")
        data = resp.json()
        self.assertEqual(data["choices"][0]["message"]["content"], "Echo: hello")
        mock_batcher.submit.assert_not_called()

    async def test_non_streaming_semantic_cache_hit_skips_batcher(self) -> None:
        from api.routes import create_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from services.batcher import AsyncRequestBatcher

        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.lookup_semantic = AsyncMock(
            return_value=SemanticCacheLookup(
                prompt="hi",
                embedding=[1.0, 0.0, 0.0],
                chunks=["Echo: hello"],
                similarity=0.98,
            )
        )
        cache.set = AsyncMock()
        cache.set_semantic = AsyncMock()

        mock_batcher = MagicMock(spec=AsyncRequestBatcher)
        mock_batcher.submit = AsyncMock()
        mock_batcher.stats = MagicMock(
            return_value=MagicMock(
                queued_requests=0,
                processed_requests=0,
                processed_batches=0,
                largest_batch_size=0,
                max_batch_size=8,
                max_wait_ms=25,
            )
        )
        mock_streaming_batcher = MagicMock()

        app = FastAPI()
        app.include_router(create_router(mock_batcher, mock_streaming_batcher, cache))

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gateway-echo",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["x-cache"], "HIT-SEMANTIC")
        self.assertEqual(
            resp.json()["choices"][0]["message"]["content"], "Echo: hello"
        )
        mock_batcher.submit.assert_not_called()
        cache.set.assert_called_once()
        cache.set_semantic.assert_called_once()

    async def test_non_streaming_cache_miss_calls_batcher_and_stores(self) -> None:
        import json as _json
        from core.models import ChatMessage
        from services.batcher import AsyncRequestBatcher
        from api.routes import create_router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from core.models import build_chat_completion_response

        cache = self._make_mock_cache(None)

        batcher_response = build_chat_completion_response(
            model="gateway-echo", content="Echo: hello", prompt_tokens=1
        )
        mock_batcher = MagicMock(spec=AsyncRequestBatcher)
        mock_batcher.submit = AsyncMock(return_value=batcher_response)
        mock_batcher.stats = MagicMock(
            return_value=MagicMock(
                queued_requests=0,
                processed_requests=0,
                processed_batches=0,
                largest_batch_size=0,
                max_batch_size=8,
                max_wait_ms=25,
            )
        )
        mock_streaming_batcher = MagicMock()

        app = FastAPI()
        app.include_router(create_router(mock_batcher, mock_streaming_batcher, cache))

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gateway-echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["x-cache"], "MISS")
        mock_batcher.submit.assert_called_once()
        cache._mock_client.set.assert_called_once()
        stored = _json.loads(cache._mock_client.set.call_args[0][1])
        self.assertEqual(stored, ["Echo: hello"])
