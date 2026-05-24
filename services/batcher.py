from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
import heapq
import logging
from uuid import uuid4

from core.models import ChatCompletionRequest, ChatCompletionResponse
from core.priority import PRIORITY_LEVELS, PriorityLevel, priority_rank
from infra.providers import LLMProvider, StreamingLLMProvider
from services.observability import (
    log_event,
    observe_batch,
    observe_latency,
    set_queue_depth,
    set_queue_depth_by_priority,
)
from services.queue import QueueFullError


logger = logging.getLogger(__name__)


def _set_priority_queue_depths(
    queue_name: str, priorities: list[PriorityLevel]
) -> None:
    set_queue_depth(queue=queue_name, depth=len(priorities))
    for priority in PRIORITY_LEVELS:
        set_queue_depth_by_priority(
            queue=queue_name,
            priority=priority,
            depth=priorities.count(priority),
        )


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
    priority: PriorityLevel = "normal"


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
        self._queue: list[tuple[int, float, int, _QueuedRequest]] = []
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._flush_handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._processed_requests = 0
        self._processed_batches = 0
        self._largest_batch_size = 0
        self._batcher_id = f"async-{uuid4().hex[:8]}"
        self._metrics_queue_name = "async_batcher"
        _set_priority_queue_depths(self._metrics_queue_name, [])

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
                _set_priority_queue_depths(
                    self._metrics_queue_name,
                    [entry[3].priority for entry in self._queue],
                )
                log_event(
                    logger,
                    "batcher.enqueue_rejected",
                    trace_id=trace_id,
                    batcher_id=self._batcher_id,
                    queue_depth=len(self._queue),
                    max_queue_size=self._max_queue_size,
                    priority=request.priority,
                    stream=False,
                )
                raise QueueFullError("request queue is full")

            enqueued_at = loop.time()
            queued_request = _QueuedRequest(
                request=request,
                future=future,
                trace_id=trace_id,
                enqueued_at=enqueued_at,
                priority=request.priority,
            )
            heapq.heappush(
                self._queue,
                (
                    priority_rank(request.priority),
                    enqueued_at,
                    self._sequence,
                    queued_request,
                ),
            )
            self._sequence += 1
            queue_depth = len(self._queue)
            _set_priority_queue_depths(
                self._metrics_queue_name,
                [entry[3].priority for entry in self._queue],
            )
            log_event(
                logger,
                "batcher.enqueue",
                trace_id=trace_id,
                batcher_id=self._batcher_id,
                queue_depth=queue_depth,
                priority=request.priority,
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
            batch = [
                heapq.heappop(self._queue)[3] for _ in range(len(self._queue))
            ]
            self._queue = []
            _set_priority_queue_depths(self._metrics_queue_name, [])

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
    priority: PriorityLevel = "normal"
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
        self._input_queue: list[tuple[int, float, int, _StreamItem]] = []
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._processed_requests = 0
        self._processed_batches = 0
        self._largest_batch_size = 0
        self._batcher_id = f"dynamic-{uuid4().hex[:8]}"
        self._metrics_queue_name = "dynamic_batcher"
        _set_priority_queue_depths(self._metrics_queue_name, [])

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
            priority=request.priority,
        )
        async with self._not_empty:
            if len(self._input_queue) >= self._max_queue_size:
                _set_priority_queue_depths(
                    self._metrics_queue_name,
                    [entry[3].priority for entry in self._input_queue],
                )
                log_event(
                    logger,
                    "batcher.enqueue_rejected",
                    trace_id=trace_id,
                    batcher_id=self._batcher_id,
                    queue_depth=len(self._input_queue),
                    max_queue_size=self._max_queue_size,
                    priority=request.priority,
                    stream=True,
                )
                raise QueueFullError("request queue is full")

            heapq.heappush(
                self._input_queue,
                (
                    priority_rank(request.priority),
                    item.enqueued_at,
                    self._sequence,
                    item,
                ),
            )
            self._sequence += 1
            queue_depth = len(self._input_queue)
            _set_priority_queue_depths(
                self._metrics_queue_name,
                [entry[3].priority for entry in self._input_queue],
            )
            self._not_empty.notify()

        log_event(
            logger,
            "batcher.enqueue",
            trace_id=trace_id,
            batcher_id=self._batcher_id,
            queue_depth=queue_depth,
            priority=request.priority,
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
        pending: list[_StreamItem] = []
        async with self._not_empty:
            while self._input_queue:
                pending.append(heapq.heappop(self._input_queue)[3])
            _set_priority_queue_depths(self._metrics_queue_name, [])
            self._not_empty.notify_all()

        # Signal closure to any requests still waiting in the input queue.
        for item in pending:
            await item.response_channel.put(RuntimeError("batcher is closed"))
            await item.response_channel.put(None)

    def stats(self) -> DynamicBatcherStats:
        return DynamicBatcherStats(
            queued_requests=len(self._input_queue),
            processed_requests=self._processed_requests,
            processed_batches=self._processed_batches,
            largest_batch_size=self._largest_batch_size,
            max_batch_size=self._max_batch_size,
            max_wait_ms=self._max_wait_ms,
            max_queue_size=self._max_queue_size,
        )

    async def _get_next_item(self, *, timeout: float | None = None) -> _StreamItem:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout if timeout is not None else None

        async with self._not_empty:
            while not self._input_queue:
                if deadline is None:
                    await self._not_empty.wait()
                    continue

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError

                try:
                    await asyncio.wait_for(
                        self._not_empty.wait(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    raise

            item = heapq.heappop(self._input_queue)[3]
            _set_priority_queue_depths(
                self._metrics_queue_name,
                [entry[3].priority for entry in self._input_queue],
            )
            return item

    async def _run(self) -> None:
        while True:
            # Block until the first request of the next batch arrives.
            first = await self._get_next_item()
            batch: list[_StreamItem] = [first]

            # Open a MAX_WAIT_MS window to accumulate additional requests.
            deadline = asyncio.get_running_loop().time() + self._max_wait_seconds
            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await self._get_next_item(timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            batch.sort(
                key=lambda item: (priority_rank(item.priority), item.enqueued_at)
            )

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
