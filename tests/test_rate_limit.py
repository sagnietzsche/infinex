from __future__ import annotations

import hashlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from core.config import Settings
from main import create_app
from services.rate_limit import (
    RedisTokenBucketRateLimiter,
    TOKEN_BUCKET_LUA,
)


class RedisTokenBucketRateLimiterTests(unittest.IsolatedAsyncioTestCase):
    def _make_limiter(self) -> tuple[RedisTokenBucketRateLimiter, MagicMock]:
        mock_client = MagicMock()
        mock_client.eval = AsyncMock(return_value=[1, 0])
        mock_client.aclose = AsyncMock()

        with patch("services.rate_limit.aioredis.from_url", return_value=mock_client):
            limiter = RedisTokenBucketRateLimiter(
                redis_url="redis://localhost:6379",
                capacity=10,
                refill_rate_per_second=2,
            )

        return limiter, mock_client

    async def test_check_runs_token_bucket_lua_script_atomically(self) -> None:
        limiter, mock_client = self._make_limiter()

        result = await limiter.check("secret-key")

        self.assertTrue(result.allowed)
        self.assertEqual(result.retry_after_seconds, 0)
        expected_key = (
            "llm-gateway:rate-limit:v1:"
            + hashlib.sha256(b"secret-key").hexdigest()
        )
        mock_client.eval.assert_awaited_once_with(
            TOKEN_BUCKET_LUA,
            1,
            expected_key,
            10,
            0.002,
            1,
            10000,
        )

    async def test_denied_result_converts_retry_delay_to_seconds(self) -> None:
        limiter, mock_client = self._make_limiter()
        mock_client.eval.return_value = [0, 1200]

        result = await limiter.check("secret-key")

        self.assertFalse(result.allowed)
        self.assertEqual(result.retry_after_seconds, 2)

    async def test_close_closes_redis_client(self) -> None:
        limiter, mock_client = self._make_limiter()

        await limiter.close()

        mock_client.aclose.assert_awaited_once()

    def test_lua_script_uses_token_bucket_primitives(self) -> None:
        self.assertIn('redis.call("HMGET"', TOKEN_BUCKET_LUA)
        self.assertIn('redis.call("HSET"', TOKEN_BUCKET_LUA)
        self.assertIn('redis.call("PEXPIRE"', TOKEN_BUCKET_LUA)
        self.assertIn("tokens = math.min(capacity", TOKEN_BUCKET_LUA)
        self.assertIn("retry_after_ms = math.ceil", TOKEN_BUCKET_LUA)


class APIKeyRateLimitMiddlewareTests(unittest.TestCase):
    def _settings(self) -> Settings:
        return Settings(
            cache_enabled=False,
            allowed_api_keys=("secret-key",),
            rate_limit_capacity=2,
            rate_limit_refill_per_second=1,
        )

    def _mock_redis_client(self, eval_result: list[int]) -> MagicMock:
        mock_client = MagicMock()
        mock_client.eval = AsyncMock(return_value=eval_result)
        mock_client.aclose = AsyncMock()
        return mock_client

    def test_missing_api_key_returns_401(self) -> None:
        mock_client = self._mock_redis_client([1, 0])
        with patch("services.rate_limit.aioredis.from_url", return_value=mock_client):
            app = create_app(self._settings())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Invalid or missing API key"})
        mock_client.eval.assert_not_called()

    def test_invalid_api_key_returns_401(self) -> None:
        mock_client = self._mock_redis_client([1, 0])
        with patch("services.rate_limit.aioredis.from_url", return_value=mock_client):
            app = create_app(self._settings())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "wrong-key"},
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Invalid or missing API key"})
        mock_client.eval.assert_not_called()

    def test_rate_limited_request_returns_429_with_retry_after(self) -> None:
        mock_client = self._mock_redis_client([0, 1500])
        with patch("services.rate_limit.aioredis.from_url", return_value=mock_client):
            app = create_app(self._settings())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "secret-key"},
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"detail": "Too Many Requests"})
        self.assertEqual(response.headers["Retry-After"], "2")

    def test_valid_api_key_with_available_token_reaches_route(self) -> None:
        mock_client = self._mock_redis_client([1, 0])
        with patch("services.rate_limit.aioredis.from_url", return_value=mock_client):
            app = create_app(self._settings())
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "secret-key"},
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "Echo: hello",
        )

    def test_health_check_does_not_require_api_key(self) -> None:
        mock_client = self._mock_redis_client([1, 0])
        with patch("services.rate_limit.aioredis.from_url", return_value=mock_client):
            app = create_app(self._settings())
            with TestClient(app) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        mock_client.eval.assert_not_called()
