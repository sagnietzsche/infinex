from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
import logging
from typing import Any

from services.observability import log_event


logger = logging.getLogger(__name__)


class ShutdownDrain:
    def __init__(self, *, timeout_seconds: float) -> None:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be greater than or equal to zero")

        self.timeout_seconds = timeout_seconds
        self.shutdown_event = asyncio.Event()
        self._condition = asyncio.Condition()
        self._active_requests = 0
        self._active_streams = 0
        self._task_refcounts: dict[asyncio.Task[Any], int] = {}
        self._drain_task: asyncio.Task[None] | None = None

    @property
    def is_shutting_down(self) -> bool:
        return self.shutdown_event.is_set()

    @property
    def drain_task(self) -> asyncio.Task[None] | None:
        return self._drain_task

    async def snapshot(self) -> dict[str, int]:
        async with self._condition:
            return self._snapshot_locked()

    def initiate(self) -> None:
        if self.shutdown_event.is_set():
            return

        self.shutdown_event.set()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._drain_task = loop.create_task(
            self._drain(), name="shutdown-drain"
        )

    async def wait_until_complete(self) -> None:
        if self._drain_task is not None:
            await asyncio.shield(self._drain_task)

    @asynccontextmanager
    async def track_request(self) -> AsyncIterator[None]:
        await self._increment(stream=False, track_task=True)
        try:
            yield
        finally:
            await self._decrement(stream=False, track_task=True)

    async def track_stream(
        self, stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        await self._increment(stream=True, track_task=False)

        async def tracked() -> AsyncIterator[str]:
            task_registered = False
            try:
                await self._register_current_task()
                task_registered = True
                async with aclosing(stream):
                    async for chunk in stream:
                        yield chunk
            finally:
                if task_registered:
                    await self._unregister_current_task()
                await self._decrement(stream=True, track_task=False)

        return tracked()

    async def _drain(self) -> None:
        snapshot = await self.snapshot()
        log_event(
            logger,
            "shutdown.drain_started",
            timeout_seconds=self.timeout_seconds,
            **snapshot,
        )

        try:
            await asyncio.wait_for(
                self._wait_for_no_active_requests(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            aborted = await self._abort_remaining()
            log_event(
                logger,
                "shutdown.drain_forced",
                level=logging.WARNING,
                aborted_requests=aborted,
            )
            log_event(
                logger,
                "shutdown.drain_completed",
                outcome="forced",
                aborted_requests=aborted,
                **await self.snapshot(),
            )
            return

        log_event(
            logger,
            "shutdown.drain_completed",
            outcome="clean",
            aborted_requests=0,
            **await self.snapshot(),
        )

    async def _increment(self, *, stream: bool, track_task: bool) -> None:
        async with self._condition:
            if stream:
                self._active_streams += 1
            else:
                self._active_requests += 1
            if track_task:
                self._register_current_task_locked()

    async def _decrement(self, *, stream: bool, track_task: bool) -> None:
        async with self._condition:
            if stream:
                self._active_streams = max(0, self._active_streams - 1)
            else:
                self._active_requests = max(0, self._active_requests - 1)
            if track_task:
                self._unregister_current_task_locked()
            if self._active_total_locked() == 0:
                self._condition.notify_all()

    async def _register_current_task(self) -> None:
        async with self._condition:
            self._register_current_task_locked()

    async def _unregister_current_task(self) -> None:
        async with self._condition:
            self._unregister_current_task_locked()

    def _register_current_task_locked(self) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._task_refcounts[task] = self._task_refcounts.get(task, 0) + 1

    def _unregister_current_task_locked(self) -> None:
        task = asyncio.current_task()
        if task is None or task not in self._task_refcounts:
            return

        remaining = self._task_refcounts[task] - 1
        if remaining <= 0:
            self._task_refcounts.pop(task, None)
        else:
            self._task_refcounts[task] = remaining

    async def _wait_for_no_active_requests(self) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._active_total_locked() == 0
            )

    async def _abort_remaining(self) -> int:
        async with self._condition:
            aborted = self._active_total_locked()
            tasks = list(self._task_refcounts)

        for task in tasks:
            if not task.done():
                task.cancel()

        return aborted

    def _active_total_locked(self) -> int:
        return self._active_requests + self._active_streams

    def _snapshot_locked(self) -> dict[str, int]:
        return {
            "active_requests": self._active_requests,
            "active_streaming_responses": self._active_streams,
            "active_total": self._active_total_locked(),
        }
