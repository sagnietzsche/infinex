"""Phase 7 tests: streaming API."""

import pytest


@pytest.mark.asyncio
async def test_complete_returns_full_response(model):
    """Non-streaming completion: text, finish_reason, token counts."""
    raise NotImplementedError


@pytest.mark.asyncio
async def test_stream_yields_incremental_chunks(model):
    """Streamed chunks concatenate to the non-streaming text (same seed)."""
    raise NotImplementedError


@pytest.mark.asyncio
async def test_concurrent_requests_are_isolated(model):
    """Two interleaved streams each receive only their own tokens."""
    raise NotImplementedError
