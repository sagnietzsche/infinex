from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from core.models import ChatCompletionRequest, ChatCompletionResponse
from infra.providers import LLMProvider, StreamingLLMProvider


@dataclass
class BatcherStats:
    queued_requests: int
    processed_requests: int
    processed_batches: int
    largest_batch_size: int
    max_batch_size: int
    max_wait_ms: int


@dataclass
class _QueuedRequest:
    request: ChatCompletionRequest
    future: asyncio.Future[ChatCompletionResponse]


class AsyncRequestBatcher:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        max_batch_size: int,
        max_wait_ms: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be greater than zero")

        self._provider = provider
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._max_wait_seconds = max_wait_ms / 1000
        self._queue: list[_QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._flush_handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._processed_requests = 0
        self._processed_batches = 0
        self._largest_batch_size = 0

    async def submit(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ChatCompletionResponse] = loop.create_future()
        should_flush = False

        async with self._lock:
            if self._closed:
                raise RuntimeError("batcher is closed")

            self._queue.append(_QueuedRequest(request=request, future=future))

            if len(self._queue) == 1:
                self._flush_handle = loop.call_later(
                    self._max_wait_seconds, self._schedule_flush
                )

            if len(self._queue) >= self._max_batch_size:
                should_flush = True

        if should_flush:
            await self.flush()

        return await future

    async def flush(self) -> None:
        async with self._lock:
            batch = self._queue
            self._queue = []

            if self._flush_handle is not None:
                self._flush_handle.cancel()
                self._flush_handle = None

        if not batch:
            return

        self._processed_batches += 1
        self._processed_requests += len(batch)
        self._largest_batch_size = max(self._largest_batch_size, len(batch))

        results = await asyncio.gather(
            *(self._provider.complete(item.request) for item in batch),
            return_exceptions=True,
        )

        for item, result in zip(batch, results, strict=True):
            if item.future.done():
                continue

            if isinstance(result, Exception):
                item.future.set_exception(result)
            else:
                item.future.set_result(result)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

        await self.flush()

    def stats(self) -> BatcherStats:
        return BatcherStats(
            queued_requests=len(self._queue),
            processed_requests=self._processed_requests,
            processed_batches=self._processed_batches,
            largest_batch_size=self._largest_batch_size,
            max_batch_size=self._max_batch_size,
            max_wait_ms=self._max_wait_ms,
        )

    def _schedule_flush(self) -> None:
        asyncio.create_task(self.flush())


# ---------------------------------------------------------------------------
# Dynamic streaming batcher
# ---------------------------------------------------------------------------

@dataclass
class DynamicBatcherStats:
    queued_requests: int
    processed_requests: int
    processed_batches: int
    largest_batch_size: int
    max_batch_size: int
    max_wait_ms: int


@dataclass
class _StreamItem:
    request: ChatCompletionRequest
    response_channel: asyncio.Queue[str | BaseException | None] = field(
        default_factory=asyncio.Queue
    )


class DynamicBatcher:
    """Collects requests into batches, then streams tokens back per-request.

    Call ``start()`` to launch the background worker before submitting requests.
    Each call to ``submit`` returns an ``asyncio.Queue`` that receives string
    tokens as they arrive, followed by ``None`` as an end-of-stream sentinel.
    If the provider raises, the exception instance is placed in the queue
    before the ``None`` sentinel so callers can propagate it.
    """

    def __init__(
        self,
        *,
        provider: StreamingLLMProvider,
        max_batch_size: int,
        max_wait_ms: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be greater than zero")

        self._provider = provider
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._max_wait_seconds = max_wait_ms / 1000
        self._input_queue: asyncio.Queue[_StreamItem] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._processed_requests = 0
        self._processed_batches = 0
        self._largest_batch_size = 0

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="dynamic-batcher")

    async def submit(
        self, request: ChatCompletionRequest
    ) -> asyncio.Queue[str | BaseException | None]:
        if self._closed:
            raise RuntimeError("batcher is closed")
        item = _StreamItem(request=request)
        await self._input_queue.put(item)
        return item.response_channel

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Signal closure to any requests still waiting in the input queue.
        while not self._input_queue.empty():
            try:
                item = self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await item.response_channel.put(RuntimeError("batcher is closed"))
            await item.response_channel.put(None)

    def stats(self) -> DynamicBatcherStats:
        return DynamicBatcherStats(
            queued_requests=self._input_queue.qsize(),
            processed_requests=self._processed_requests,
            processed_batches=self._processed_batches,
            largest_batch_size=self._largest_batch_size,
            max_batch_size=self._max_batch_size,
            max_wait_ms=self._max_wait_ms,
        )

    async def _run(self) -> None:
        while True:
            # Block until the first request of the next batch arrives.
            first = await self._input_queue.get()
            batch: list[_StreamItem] = [first]

            # Open a MAX_WAIT_MS window to accumulate additional requests.
            deadline = asyncio.get_running_loop().time() + self._max_wait_seconds
            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._input_queue.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            self._processed_batches += 1
            self._processed_requests += len(batch)
            self._largest_batch_size = max(self._largest_batch_size, len(batch))

            # Fire all provider calls in parallel; a failure in one must not
            # propagate and crash the loop — return_exceptions=True ensures this.
            await asyncio.gather(
                *(self._stream_to_channel(item) for item in batch),
                return_exceptions=True,
            )

    async def _stream_to_channel(self, item: _StreamItem) -> None:
        try:
            async for token in self._provider.stream(item.request):
                await item.response_channel.put(token)
        except Exception as exc:
            # Route the exception to the caller via the response channel.
            await item.response_channel.put(exc)
        finally:
            # Always close the stream with the None sentinel.
            await item.response_channel.put(None)
