from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
import logging

from fastapi.testclient import TestClient

from core.config import Settings
from main import create_app
from services.shutdown import ShutdownDrain


def test_gateway_returns_503_after_shutdown_begins() -> None:
    app = create_app(Settings(cache_enabled=False))

    with TestClient(app) as client:
        app.state.shutdown_drain.initiate()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Gateway is shutting down"}


def test_stream_counter_decrements_when_generator_is_closed() -> None:
    async def run() -> None:
        drain = ShutdownDrain(timeout_seconds=1)

        async def stream() -> AsyncIterator[str]:
            yield "first"
            await asyncio.sleep(60)

        tracked = await drain.track_stream(stream())
        assert (await drain.snapshot())["active_streaming_responses"] == 1

        assert await anext(tracked) == "first"
        await tracked.aclose()

        snapshot = await drain.snapshot()
        assert snapshot["active_streaming_responses"] == 0
        assert snapshot["active_total"] == 0

    asyncio.run(run())


def test_forced_drain_cancels_active_requests_and_logs_warning(caplog) -> None:
    async def run() -> None:
        drain = ShutdownDrain(timeout_seconds=0.01)
        never = asyncio.Event()

        async def hold_request() -> None:
            async with drain.track_request():
                await never.wait()

        task = asyncio.create_task(hold_request())
        while (await drain.snapshot())["active_total"] == 0:
            await asyncio.sleep(0)

        caplog.set_level(logging.INFO, logger="services.shutdown")
        drain.initiate()
        await drain.wait_until_complete()

        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert '"event": "shutdown.drain_started"' in caplog.text
    assert '"event": "shutdown.drain_forced"' in caplog.text
    assert '"aborted_requests": 1' in caplog.text
    assert '"outcome": "forced"' in caplog.text
