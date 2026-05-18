from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.models import ChatCompletionRequest, ChatCompletionResponse
from infra.providers import LLMProvider


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
