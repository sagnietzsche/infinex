from __future__ import annotations

import asyncio
import hashlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.usage import TokenCounter, UsageTotals, UsageTracker


class UsageTrackerTests(unittest.IsolatedAsyncioTestCase):
    def _make_tracker(
        self, *, token_counter: object | None = None
    ) -> tuple[UsageTracker, MagicMock]:
        mock_client = MagicMock()
        mock_client.hincrby = AsyncMock()
        mock_client.hincrbyfloat = AsyncMock()
        mock_client.hgetall = AsyncMock(return_value={})
        mock_client.aclose = AsyncMock()

        with patch("services.usage.aioredis.from_url", return_value=mock_client):
            tracker = UsageTracker(
                redis_url="redis://localhost:6379",
                pricing_table={"test-model": {"input": 0.1, "output": 0.2}},
                token_counter=token_counter,  # type: ignore[arg-type]
            )

        return tracker, mock_client

    async def test_record_request_logs_and_accumulates_usage_by_hashed_key(self) -> None:
        class FakeTokenCounter:
            async def count_text_tokens(self, *, model: str, text: str) -> int:
                return 4

        tracker, mock_client = self._make_tracker(token_counter=FakeTokenCounter())

        usage = await tracker.record_request(
            api_key="secret-key",
            model="test-model",
            prompt_tokens=10,
            completion_text="completion text",
            completion_usage=None,
            log_fields={"trace_id": "trace-1", "stream": False, "cache_hit": False},
        )

        redis_key = (
            "llm-gateway:usage:v1:"
            + hashlib.sha256(b"secret-key").hexdigest()
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 4)
        self.assertEqual(usage.total_tokens, 14)
        self.assertEqual(usage.estimated_cost_usd, 0.0018)
        mock_client.hincrby.assert_any_await(redis_key, "prompt_tokens", 10)
        mock_client.hincrby.assert_any_await(redis_key, "completion_tokens", 4)
        mock_client.hincrby.assert_any_await(redis_key, "total_tokens", 14)
        mock_client.hincrby.assert_any_await(redis_key, "request_count", 1)
        mock_client.hincrbyfloat.assert_awaited_once_with(
            redis_key, "estimated_cost_usd", 0.0018
        )

    async def test_get_usage_returns_zeroes_for_missing_hash_fields(self) -> None:
        tracker, mock_client = self._make_tracker()
        mock_client.hgetall.return_value = {
            "prompt_tokens": "2",
            "completion_tokens": "3",
            "total_tokens": "5",
            "estimated_cost_usd": "0.25",
            "request_count": "4",
        }

        usage = await tracker.get_usage("secret-key")

        self.assertEqual(
            usage,
            UsageTotals(
                prompt_tokens=2,
                completion_tokens=3,
                total_tokens=5,
                estimated_cost_usd=0.25,
                request_count=4,
            ),
        )

    async def test_prompt_token_task_is_awaited_when_recording(self) -> None:
        tracker, _ = self._make_tracker()
        prompt_task = asyncio.create_task(asyncio.sleep(0, result=12))

        usage = await tracker.record_request(
            api_key=None,
            model="test-model",
            prompt_tokens=prompt_task,
            completion_text="",
            completion_usage=None,
            log_fields={"trace_id": "trace-2", "stream": False, "cache_hit": False},
        )

        self.assertEqual(usage.prompt_tokens, 12)


class TokenCounterTests(unittest.TestCase):
    def test_unknown_model_uses_character_heuristic(self) -> None:
        counter = TokenCounter()

        tokens = counter.count_text_tokens_sync(
            model="provider/not-a-real-model",
            text="1234567",
        )

        self.assertEqual(tokens, 2)
