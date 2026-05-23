from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
import logging
from uuid import uuid4

from core.models import ChatCompletionRequest, ChatCompletionResponse
from infra.providers import LLMProvider, StreamingLLMProvider
from services.observability import (
    log_event,
    observe_batch,
    observe_latency,
    set_queue_depth,
)
from services.queue import QueueFullError


logger = logging.getLogger(__name__)


@dataclass
class BatcherStats:
    queued_requests: int
    processed_requests: int
    processed_batches: int
    largest_batch_size: int
    max_batch_size: int
    max_wait_ms: int
    max_queue_size: int


@dataclass
class _QueuedRequest:
    request: ChatCompletionRequest
    future: asyncio.Future[ChatCompletionResponse]
    trace_id: str
    enqueued_at: float


class AsyncRequestBatcher:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        max_batch_size: int,
        max_wait_ms: int,
        max_queue_size: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be greater than zero")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than zero")

        self._provider = provider
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._max_queue_size = max_queue_size
        self._max_wait_seconds = max_wait_ms / 1000
        self._queue: list[_QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._flush_handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._processed_requests = 0
        self._processed_batches = 0
        self._largest_batch_size = 0
        self._batcher_id = f"async-{uuid4().hex[:8]}"
        self._metrics_queue_name = "async_batcher"
        set_queue_depth(queue=self._metrics_queue_name, depth=0)

    async def submit(
        self, request: ChatCompletionRequest, trace_id: str | None = None
    ) -> ChatCompletionResponse:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ChatCompletionResponse] = loop.create_future()
        trace_id = trace_id or uuid4().hex
        should_flush = False

        async with self._lock:
            if self._closed:
                raise RuntimeError("batcher is closed")

            if len(self._queue) >= self._max_queue_size:
                future.cancel()
                set_queue_depth(
                    queue=self._metrics_queue_name,
                    depth=len(self._queue),
                )
                log_event(
                    logger,
                    "batcher.enqueue_rejected",
                    trace_id=trace_id,
                    batcher_id=self._batcher_id,
                    queue_depth=len(self._queue),
                    max_queue_size=self._max_queue_size,
                    stream=False,
                )
                raise QueueFullError("request queue is full")

            self._queue.append(
                _QueuedRequest(
                    request=request,
                    future=future,
                    trace_id=trace_id,
                    enqueued_at=loop.time(),
                )
            )
            queue_depth = len(self._queue)
            set_queue_depth(queue=self._metrics_queue_name, depth=queue_depth)
            log_event(
                logger,
                "batcher.enqueue",
                trace_id=trace_id,
                batcher_id=self._batcher_id,
                queue_depth=queue_depth,
                stream=False,
            )

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
            set_queue_depth(queue=self._metrics_queue_name, depth=0)

            if self._flush_handle is not None:
                self._flush_handle.cancel()
                self._flush_handle = None

        if not batch:
            return

        loop = asyncio.get_running_loop()
        batch_id = f"batch-{uuid4().hex[:8]}"
        self._processed_batches += 1
        self._processed_requests += len(batch)
        self._largest_batch_size = max(self._largest_batch_size, len(batch))
        observe_batch(
            batcher=self._batcher_id,
            size=len(batch),
            max_size=self._max_batch_size,
        )
        for item in batch:
            observe_latency(
                operation="batcher_queue_wait",
                seconds=loop.time() - item.enqueued_at,
            )
        log_event(
            logger,
            "batcher.flush",
            batcher_id=self._batcher_id,
            batch_id=batch_id,
            batch_size=len(batch),
            trace_ids=[item.trace_id for item in batch],
            stream=False,
        )

        provider_started_at = loop.time()
        results = await asyncio.gather(
            *(self._provider.complete(item.request) for item in batch),
            return_exceptions=True,
        )
        observe_latency(
            operation="batcher_provider_batch",
            seconds=loop.time() - provider_started_at,
        )

        for item, result in zip(batch, results, strict=True):
            if item.future.done():
                continue

            observe_latency(
                operation="batcher_request_total",
                seconds=loop.time() - item.enqueued_at,
            )
            if isinstance(result, Exception):
                item.future.set_exception(result)
                log_event(
                    logger,
                    "batcher.request_failed",
                    trace_id=item.trace_id,
                    batcher_id=self._batcher_id,
                    batch_id=batch_id,
                    error=repr(result),
                    stream=False,
                )
            else:
                item.future.set_result(result)
                log_event(
                    logger,
                    "batcher.request_completed",
                    trace_id=item.trace_id,
                    batcher_id=self._batcher_id,
                    batch_id=batch_id,
                    stream=False,
                )

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
            max_queue_size=self._max_queue_size,
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
    max_queue_size: int


@dataclass
class _StreamItem:
    request: ChatCompletionRequest
    trace_id: str
    enqueued_at: float
    cancelled: bool = False
    response_channel: asyncio.Queue[object] = field(
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
        max_queue_size: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be greater than zero")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than zero")

        self._provider = provider
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._max_queue_size = max_queue_size
        self._max_wait_seconds = max_wait_ms / 1000
        self._input_queue: asyncio.Queue[_StreamItem] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._processed_requests = 0
        self._processed_batches = 0
        self._largest_batch_size = 0
        self._batcher_id = f"dynamic-{uuid4().hex[:8]}"
        self._metrics_queue_name = "dynamic_batcher"
        set_queue_depth(queue=self._metrics_queue_name, depth=0)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="dynamic-batcher")

    async def submit(
        self, request: ChatCompletionRequest, trace_id: str | None = None
    ) -> _StreamItem:
        if self._closed:
            raise RuntimeError("batcher is closed")
        loop = asyncio.get_running_loop()
        trace_id = trace_id or uuid4().hex
        item = _StreamItem(
            request=request,
            trace_id=trace_id,
            enqueued_at=loop.time(),
        )
        try:
            self._input_queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            set_queue_depth(
                queue=self._metrics_queue_name,
                depth=self._input_queue.qsize(),
            )
            log_event(
                logger,
                "batcher.enqueue_rejected",
                trace_id=trace_id,
                batcher_id=self._batcher_id,
                queue_depth=self._input_queue.qsize(),
                max_queue_size=self._max_queue_size,
                stream=True,
            )
            raise QueueFullError("request queue is full") from exc
        queue_depth = self._input_queue.qsize()
        set_queue_depth(queue=self._metrics_queue_name, depth=queue_depth)
        log_event(
            logger,
            "batcher.enqueue",
            trace_id=trace_id,
            batcher_id=self._batcher_id,
            queue_depth=queue_depth,
            stream=True,
        )
        return item

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
            set_queue_depth(
                queue=self._metrics_queue_name,
                depth=self._input_queue.qsize(),
            )
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
            max_queue_size=self._max_queue_size,
        )

    async def _run(self) -> None:
        while True:
            # Block until the first request of the next batch arrives.
            first = await self._input_queue.get()
            set_queue_depth(
                queue=self._metrics_queue_name,
                depth=self._input_queue.qsize(),
            )
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
                    set_queue_depth(
                        queue=self._metrics_queue_name,
                        depth=self._input_queue.qsize(),
                    )
                except asyncio.TimeoutError:
                    break

            loop = asyncio.get_running_loop()
            batch_id = f"batch-{uuid4().hex[:8]}"
            self._processed_batches += 1
            self._processed_requests += len(batch)
            self._largest_batch_size = max(self._largest_batch_size, len(batch))
            observe_batch(
                batcher=self._batcher_id,
                size=len(batch),
                max_size=self._max_batch_size,
            )
            for item in batch:
                observe_latency(
                    operation="dynamic_batcher_queue_wait",
                    seconds=loop.time() - item.enqueued_at,
                )
            log_event(
                logger,
                "batcher.flush",
                batcher_id=self._batcher_id,
                batch_id=batch_id,
                batch_size=len(batch),
                trace_ids=[item.trace_id for item in batch],
                stream=True,
            )

            # Fire all provider calls in parallel; a failure in one must not
            # propagate and crash the loop — return_exceptions=True ensures this.
            await asyncio.gather(
                *(self._stream_to_channel(item, batch_id=batch_id) for item in batch),
                return_exceptions=True,
            )

    async def _stream_to_channel(self, item: _StreamItem, *, batch_id: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            # aclosing ensures aclose() is awaited when we exit early, so the
            # underlying provider connection (e.g. httpx stream) is closed
            # synchronously rather than waiting for GC.
            async with aclosing(self._provider.stream(item.request)) as stream:
                async for token in stream:
                    if item.cancelled:
                        log_event(
                            logger,
                            "batcher.stream_cancelled",
                            trace_id=item.trace_id,
                            batcher_id=self._batcher_id,
                            batch_id=batch_id,
                            stream=True,
                        )
                        return
                    await item.response_channel.put(token)
        except Exception as exc:
            if not item.cancelled:
                await item.response_channel.put(exc)
                log_event(
                    logger,
                    "batcher.request_failed",
                    trace_id=item.trace_id,
                    batcher_id=self._batcher_id,
                    batch_id=batch_id,
                    error=repr(exc),
                    stream=True,
                )
        finally:
            observe_latency(
                operation="dynamic_batcher_request_total",
                seconds=loop.time() - item.enqueued_at,
            )
            if not item.cancelled:
                log_event(
                    logger,
                    "batcher.request_completed",
                    trace_id=item.trace_id,
                    batcher_id=self._batcher_id,
                    batch_id=batch_id,
                    stream=True,
                )
            await item.response_channel.put(None)
