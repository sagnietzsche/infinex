from __future__ import annotations

import asyncio
import json
from time import time
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    build_chat_completion_response,
)
from services.batcher import AsyncRequestBatcher, BatcherStats, DynamicBatcher
from services.cache import ResponseCache, make_cache_key


def create_router(
    batcher: AsyncRequestBatcher,
    streaming_batcher: DynamicBatcher,
    cache: ResponseCache | None = None,
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
            cache_key = make_cache_key(body) if cache is not None else None

            if cache_key is not None:
                cached = await cache.get(cache_key)
                if cached is not None:
                    content = "".join(cached)
                    prompt_tokens = sum(
                        len(m.content.split()) for m in body.messages
                    )
                    return build_chat_completion_response(
                        model=body.model,
                        content=content,
                        prompt_tokens=prompt_tokens,
                    )

            result = await batcher.submit(body)

            if cache_key is not None:
                await cache.set(cache_key, [result.choices[0].message.content])

            return result

        # Streaming path — check cache before submitting to the batcher.
        cache_key = make_cache_key(body) if cache is not None else None

        if cache_key is not None:
            cached = await cache.get(cache_key)
            if cached is not None:
                stream_id = f"chatcmpl-{uuid4().hex}"
                created = int(time())

                async def generate_cached():
                    for chunk_content in cached:
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_cached(), media_type="text/event-stream"
                )

        item = await streaming_batcher.submit(body)
        stream_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())

        async def generate():
            collected: list[str] = []
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

                    collected.append(token)
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

                if cache_key is not None and collected:
                    await cache.set(cache_key, collected)

                yield "data: [DONE]\n\n"
            finally:
                # Ensure cleanup even if the parent task is cancelled.
                item.cancelled = True

        return StreamingResponse(generate(), media_type="text/event-stream")

    return router
