from __future__ import annotations

import asyncio
import json
from time import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.models import ChatCompletionRequest, ChatCompletionResponse
from services.batcher import AsyncRequestBatcher, BatcherStats, DynamicBatcher


def create_router(
    batcher: AsyncRequestBatcher,
    streaming_batcher: DynamicBatcher,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/stats", response_model=BatcherStats)
    async def stats() -> BatcherStats:
        return batcher.stats()

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
    ) -> StreamingResponse | ChatCompletionResponse:
        if not body.stream:
            return await batcher.submit(body)

        item = await streaming_batcher.submit(body)
        stream_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        item.cancelled = True
                        return

                    try:
                        token = await asyncio.wait_for(
                            item.response_channel.get(), timeout=0.05
                        )
                    except asyncio.TimeoutError:
                        continue

                    if token is None:
                        break

                    if isinstance(token, BaseException):
                        raise token

                    chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                yield "data: [DONE]\n\n"
            finally:
                # Ensure cleanup even if the parent task is cancelled.
                item.cancelled = True

        return StreamingResponse(generate(), media_type="text/event-stream")

    return router
