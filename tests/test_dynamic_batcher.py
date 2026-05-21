import asyncio
import unittest

from core.models import ChatCompletionRequest, ChatMessage
from infra.providers import EchoStreamingProvider
from services.batcher import DynamicBatcher


def _request(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)])


async def _collect(channel: asyncio.Queue) -> list[str]:
    tokens: list[str] = []
    while True:
        value = await channel.get()
        if value is None:
            break
        if isinstance(value, BaseException):
            raise value
        tokens.append(value)
    return tokens


class DynamicBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_flushes_when_batch_size_reached(self) -> None:
        batcher = DynamicBatcher(
            provider=EchoStreamingProvider(),
            max_batch_size=2,
            max_wait_ms=1000,
        )
        batcher.start()

        item1 = await batcher.submit(_request("hello"))
        item2 = await batcher.submit(_request("world"))

        tokens1, tokens2 = await asyncio.gather(
            _collect(item1.response_channel), _collect(item2.response_channel)
        )

        self.assertEqual(tokens1, ["Echo:", "hello"])
        self.assertEqual(tokens2, ["Echo:", "world"])

        stats = batcher.stats()
        self.assertEqual(stats.processed_batches, 1)
        self.assertEqual(stats.processed_requests, 2)
        self.assertEqual(stats.largest_batch_size, 2)

        await batcher.close()

    async def test_flushes_after_wait_window_expires(self) -> None:
        batcher = DynamicBatcher(
            provider=EchoStreamingProvider(),
            max_batch_size=10,
            max_wait_ms=5,
        )
        batcher.start()

        item = await batcher.submit(_request("solo"))
        tokens = await _collect(item.response_channel)

        self.assertEqual(tokens, ["Echo:", "solo"])

        stats = batcher.stats()
        self.assertEqual(stats.processed_batches, 1)
        self.assertEqual(stats.processed_requests, 1)
        self.assertEqual(stats.largest_batch_size, 1)

        await batcher.close()

    async def test_each_request_gets_its_own_channel(self) -> None:
        batcher = DynamicBatcher(
            provider=EchoStreamingProvider(),
            max_batch_size=3,
            max_wait_ms=5,
        )
        batcher.start()

        items = [await batcher.submit(_request(f"msg{i}")) for i in range(3)]
        results = await asyncio.gather(*(_collect(it.response_channel) for it in items))

        for i, tokens in enumerate(results):
            self.assertIn(f"msg{i}", tokens)

        await batcher.close()

    async def test_provider_failure_does_not_crash_batcher(self) -> None:
        class FailingProvider:
            async def stream(self, request):
                raise RuntimeError("provider down")
                # make it a proper async generator
                yield  # pragma: no cover

        batcher = DynamicBatcher(
            provider=FailingProvider(),
            max_batch_size=1,
            max_wait_ms=50,
        )
        batcher.start()

        stream_item = await batcher.submit(_request("test"))
        first = await stream_item.response_channel.get()
        self.assertIsInstance(first, RuntimeError)
        sentinel = await stream_item.response_channel.get()
        self.assertIsNone(sentinel)

        # batcher loop is still alive for the next request
        self.assertFalse(batcher._task.done())

        await batcher.close()

    async def test_tokens_routed_to_correct_channel_across_batches(self) -> None:
        batcher = DynamicBatcher(
            provider=EchoStreamingProvider(),
            max_batch_size=1,
            max_wait_ms=5,
        )
        batcher.start()

        item1 = await batcher.submit(_request("first"))
        tokens1 = await _collect(item1.response_channel)

        item2 = await batcher.submit(_request("second"))
        tokens2 = await _collect(item2.response_channel)

        self.assertIn("first", tokens1)
        self.assertIn("second", tokens2)
        self.assertEqual(batcher.stats().processed_batches, 2)

        await batcher.close()
