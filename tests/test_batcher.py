import asyncio
import unittest

from core.models import (
    ChatCompletionRequest,
    ChatMessage,
    build_chat_completion_response,
)
from services.batcher import AsyncRequestBatcher


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
