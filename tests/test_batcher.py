import asyncio
import unittest

from core.models import (
    ChatCompletionRequest,
    ChatMessage,
    build_chat_completion_response,
)
from services.batcher import AsyncRequestBatcher
from services.queue import QueueFullError


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest):
        self.requests.append(request)
        user_message = next(
            message.content
            for message in reversed(request.messages)
            if message.role == "user"
        )
        return build_chat_completion_response(
            model=request.model,
            content=f"done: {user_message}",
            prompt_tokens=len(user_message.split()),
        )


def request_with_prompt(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=prompt)]
    )


class AsyncRequestBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_priority_defaults_to_normal(self) -> None:
        self.assertEqual(request_with_prompt("default").priority, "normal")

    async def test_flushes_when_max_batch_size_is_reached(self) -> None:
        provider = RecordingProvider()
        batcher = AsyncRequestBatcher(
            provider=provider,
            max_batch_size=2,
            max_wait_ms=1000,
        )

        first = asyncio.create_task(batcher.submit(request_with_prompt("one")))
        await asyncio.sleep(0)
        second = asyncio.create_task(batcher.submit(request_with_prompt("two")))

        responses = await asyncio.gather(first, second)

        self.assertEqual([response.choices[0].message.content for response in responses], [
            "done: one",
            "done: two",
        ])
        self.assertEqual([request.messages[0].content for request in provider.requests], [
            "one",
            "two",
        ])
        stats = batcher.stats()
        self.assertEqual(stats.processed_batches, 1)
        self.assertEqual(stats.processed_requests, 2)
        self.assertEqual(stats.largest_batch_size, 2)

    async def test_flushes_when_wait_window_expires(self) -> None:
        provider = RecordingProvider()
        batcher = AsyncRequestBatcher(
            provider=provider,
            max_batch_size=10,
            max_wait_ms=5,
        )

        response = await batcher.submit(request_with_prompt("solo"))

        self.assertEqual(response.choices[0].message.content, "done: solo")
        stats = batcher.stats()
        self.assertEqual(stats.processed_batches, 1)
        self.assertEqual(stats.processed_requests, 1)
        self.assertEqual(stats.largest_batch_size, 1)

    async def test_rejects_when_queue_is_full(self) -> None:
        provider = RecordingProvider()
        batcher = AsyncRequestBatcher(
            provider=provider,
            max_batch_size=10,
            max_wait_ms=1000,
            max_queue_size=1,
        )

        first = asyncio.create_task(batcher.submit(request_with_prompt("one")))
        await asyncio.sleep(0)

        with self.assertRaises(QueueFullError):
            await batcher.submit(request_with_prompt("two"))

        await batcher.close()
        self.assertEqual(
            (await first).choices[0].message.content,
            "done: one",
        )

    async def test_dispatches_high_priority_before_lower_priority_backlog(self) -> None:
        provider = RecordingProvider()
        batcher = AsyncRequestBatcher(
            provider=provider,
            max_batch_size=3,
            max_wait_ms=1000,
        )

        low = asyncio.create_task(
            batcher.submit(
                ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="low")],
                    priority="low",
                )
            )
        )
        await asyncio.sleep(0)
        high = asyncio.create_task(
            batcher.submit(
                ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="high")],
                    priority="high",
                )
            )
        )
        normal = asyncio.create_task(batcher.submit(request_with_prompt("normal")))

        await asyncio.gather(low, high, normal)

        self.assertEqual(
            [request.messages[0].content for request in provider.requests],
            ["high", "normal", "low"],
        )
