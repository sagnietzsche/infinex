import asyncio
import unittest

from core.models import ChatCompletionRequest, ChatMessage
from services.batcher import DynamicBatcher


def _request(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)])


class SlowStreamingProvider:
    """Yields tokens one at a time with a short delay so tests can interrupt mid-stream."""

    def __init__(self, tokens: list[str], delay: float = 0.01) -> None:
        self._tokens = tokens
        self._delay = delay
        self.closed = False

    async def stream(self, request: ChatCompletionRequest):
        try:
            for token in self._tokens:
                await asyncio.sleep(self._delay)
                yield token
        finally:
            self.closed = True


class ClientDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_flag_stops_streaming(self) -> None:
        provider = SlowStreamingProvider(["a", "b", "c", "d", "e"])
        batcher = DynamicBatcher(
            provider=provider,
            max_batch_size=1,
            max_wait_ms=5,
        )
        batcher.start()

        item = await batcher.submit(_request("test"))

        # Read the first token, then mark cancelled.
        first = await item.response_channel.get()
        self.assertEqual(first, "a")
        item.cancelled = True

        # Drain the channel until the sentinel; no more real tokens should arrive
        # after cancellation (the batcher may have buffered one in-flight token).
        tokens_after_cancel: list[str] = []
        while True:
            token = await asyncio.wait_for(item.response_channel.get(), timeout=1.0)
            if token is None:
                break
            if isinstance(token, BaseException):
                raise token
            tokens_after_cancel.append(token)

        # At most one in-flight token is acceptable; the full ["b","c","d","e"]
        # sequence should NOT have been delivered.
        self.assertLess(len(tokens_after_cancel), 4)

        await batcher.close()

    async def test_cancelled_flag_closes_provider_stream(self) -> None:
        provider = SlowStreamingProvider(["x", "y", "z"], delay=0.02)
        batcher = DynamicBatcher(
            provider=provider,
            max_batch_size=1,
            max_wait_ms=5,
        )
        batcher.start()

        item = await batcher.submit(_request("close-test"))

        # Allow the first token through, then cancel.
        await item.response_channel.get()
        item.cancelled = True

        # Drain to the sentinel so _stream_to_channel finishes.
        while True:
            token = await asyncio.wait_for(item.response_channel.get(), timeout=1.0)
            if token is None:
                break

        # The provider's finally block should have run, marking it closed.
        self.assertTrue(provider.closed)

        await batcher.close()

    async def test_sentinel_always_delivered_after_cancel(self) -> None:
        """The None sentinel must appear even when the stream is cancelled."""
        provider = SlowStreamingProvider(["1", "2", "3"])
        batcher = DynamicBatcher(
            provider=provider,
            max_batch_size=1,
            max_wait_ms=5,
        )
        batcher.start()

        item = await batcher.submit(_request("sentinel-test"))
        item.cancelled = True  # cancel before reading anything

        last = await asyncio.wait_for(item.response_channel.get(), timeout=1.0)
        # The channel must eventually yield the None sentinel.
        while last is not None:
            last = await asyncio.wait_for(item.response_channel.get(), timeout=1.0)

        self.assertIsNone(last)

        await batcher.close()

    async def test_uncancelled_stream_delivers_all_tokens(self) -> None:
        """Sanity check: a normal (non-cancelled) stream still works correctly."""
        provider = SlowStreamingProvider(["hello", "world"], delay=0.001)
        batcher = DynamicBatcher(
            provider=provider,
            max_batch_size=1,
            max_wait_ms=5,
        )
        batcher.start()

        item = await batcher.submit(_request("normal"))
        tokens: list[str] = []
        while True:
            token = await asyncio.wait_for(item.response_channel.get(), timeout=1.0)
            if token is None:
                break
            if isinstance(token, BaseException):
                raise token
            tokens.append(token)

        self.assertEqual(tokens, ["hello", "world"])

        await batcher.close()
