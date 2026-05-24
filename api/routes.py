from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter, time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    build_chat_completion_response,
)
from services.batcher import AsyncRequestBatcher, BatcherStats, DynamicBatcher
from services.cache import ResponseCache, make_cache_key
from services.circuit_breaker import CircuitBreaker
from services.observability import (
    log_event,
    observe_latency,
    record_cache_lookup,
    record_request,
)
from services.provider_router import (
    AllProvidersFailedError,
    ProviderStreamMetadata,
    all_providers_error_body,
    provider_used,
)
from services.queue import QueueFullError
from services.retry import RetryExhaustedError, provider_status_code
from services.usage import UsageTotals, UsageTracker


logger = logging.getLogger(__name__)
_RETRY_ATTEMPTED_HEADER = "X-Retries-Attempted"
_PROVIDER_USED_HEADER = "X-Provider-Used"
_NO_STREAM_ITEM = object()


def _zero_usage() -> dict[str, int | float]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def _start_prompt_count(
    usage_tracker: UsageTracker | None,
    body: ChatCompletionRequest,
) -> asyncio.Task[int] | int:
    if usage_tracker is None:
        return 0
    task = asyncio.create_task(
        usage_tracker.count_prompt_tokens(body),
        name="usage-prompt-token-count",
    )
    task.add_done_callback(_consume_task_exception)
    return task


def _schedule_usage_record(
    usage_tracker: UsageTracker | None,
    *,
    api_key: str | None,
    model: str,
    prompt_tokens: int | asyncio.Task[int],
    completion_text: str,
    completion_usage: CompletionUsage | None,
    log_fields: dict[str, object],
) -> None:
    if usage_tracker is None:
        log_event(logger, "request.completed", **log_fields, **_zero_usage())
        return

    task = asyncio.create_task(
        usage_tracker.record_request(
            api_key=api_key,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_text=completion_text,
            completion_usage=completion_usage,
            log_fields=log_fields,
        ),
        name="usage-record-request",
    )
    task.add_done_callback(_consume_task_exception)


def _usage_response(usage: UsageTotals) -> dict[str, int | float]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
    }


def _consume_task_exception(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("background task failed: %r", exc)


def _raise_retry_exhausted(exc: RetryExhaustedError, *, trace_id: str) -> None:
    headers = {
        _RETRY_ATTEMPTED_HEADER: str(exc.retries_attempted),
        "x-trace-id": trace_id,
    }
    original = exc.original

    if isinstance(original, HTTPException):
        merged_headers = {**(original.headers or {}), **headers}
        raise HTTPException(
            status_code=original.status_code,
            detail=original.detail,
            headers=merged_headers,
        ) from exc

    raise HTTPException(
        status_code=provider_status_code(original) or 500,
        detail=str(original),
        headers=headers,
    ) from exc


def create_router(
    batcher: AsyncRequestBatcher,
    streaming_batcher: DynamicBatcher,
    cache: ResponseCache | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    usage_tracker: UsageTracker | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, object]:
        body: dict[str, object] = {"status": "ok"}
        if circuit_breaker is not None:
            snapshots = await circuit_breaker.snapshots()
            body["circuits"] = {
                provider: {
                    "state": snapshot.state.value,
                    "error_rate": snapshot.error_rate,
                    "errors": snapshot.errors,
                    "total": snapshot.total,
                }
                for provider, snapshot in snapshots.items()
            }
        return body

    @router.get("/stats", response_model=BatcherStats)
    async def stats() -> BatcherStats:
        return batcher.stats()

    @router.get("/admin/keys/{key}/usage")
    async def key_usage(key: str) -> dict[str, int | float]:
        if usage_tracker is None:
            return _zero_usage()
        return _usage_response(await usage_tracker.get_usage(key))

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        response: Response,
    ) -> StreamingResponse | ChatCompletionResponse:
        trace_id = request.headers.get("x-trace-id") or uuid4().hex
        response.headers["x-trace-id"] = trace_id
        started_at = perf_counter()
        api_key = request.headers.get("x-api-key")
        prompt_tokens = _start_prompt_count(usage_tracker, body)
        record_request(endpoint="chat_completions", stream=body.stream)
        log_event(
            logger,
            "request.accepted",
            trace_id=trace_id,
            path=str(request.url.path),
            stream=body.stream,
        )

        if not body.stream:
            cache_key = make_cache_key(body) if cache is not None else None

            if cache_key is not None:
                cached = await cache.get(cache_key)
                record_cache_lookup(hit=cached is not None)
                if cached is not None:
                    content = "".join(cached)
                    response_prompt_tokens = sum(
                        len(message.content.split()) for message in body.messages
                    )
                    observe_latency(
                        operation="http_chat_completion",
                        seconds=perf_counter() - started_at,
                    )
                    _schedule_usage_record(
                        usage_tracker,
                        api_key=api_key,
                        model=body.model,
                        prompt_tokens=prompt_tokens,
                        completion_text=content,
                        completion_usage=None,
                        log_fields={
                            "trace_id": trace_id,
                            "cache_hit": True,
                            "stream": False,
                        },
                    )
                    return build_chat_completion_response(
                        model=body.model,
                        content=content,
                        prompt_tokens=response_prompt_tokens,
                    )

            try:
                result = await batcher.submit(body, trace_id=trace_id)
            except QueueFullError as exc:
                observe_latency(
                    operation="http_chat_completion",
                    seconds=perf_counter() - started_at,
                )
                log_event(
                    logger,
                    "request.rejected",
                    trace_id=trace_id,
                    cache_hit=False,
                    error=repr(exc),
                    stream=False,
                    status_code=503,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Request queue is full",
                ) from exc
            except RetryExhaustedError as exc:
                observe_latency(
                    operation="http_chat_completion",
                    seconds=perf_counter() - started_at,
                )
                log_event(
                    logger,
                    "request.failed",
                    trace_id=trace_id,
                    cache_hit=False,
                    error=repr(exc.original),
                    retries_attempted=exc.retries_attempted,
                    stream=False,
                )
                _raise_retry_exhausted(exc, trace_id=trace_id)
            except AllProvidersFailedError as exc:
                observe_latency(
                    operation="http_chat_completion",
                    seconds=perf_counter() - started_at,
                )
                log_event(
                    logger,
                    "request.failed",
                    trace_id=trace_id,
                    cache_hit=False,
                    error=repr(exc),
                    stream=False,
                    status_code=503,
                )
                return JSONResponse(
                    all_providers_error_body(exc),
                    status_code=503,
                    headers={"x-trace-id": trace_id},
                )
            except Exception as exc:
                observe_latency(
                    operation="http_chat_completion",
                    seconds=perf_counter() - started_at,
                )
                log_event(
                    logger,
                    "request.failed",
                    trace_id=trace_id,
                    cache_hit=False,
                    error=repr(exc),
                    stream=False,
                )
                raise

            if cache_key is not None:
                await cache.set(cache_key, [result.choices[0].message.content])

            provider_name = provider_used(result)
            if provider_name is not None:
                response.headers[_PROVIDER_USED_HEADER] = provider_name

            observe_latency(
                operation="http_chat_completion",
                seconds=perf_counter() - started_at,
            )
            _schedule_usage_record(
                usage_tracker,
                api_key=api_key,
                model=result.model,
                prompt_tokens=prompt_tokens,
                completion_text=result.choices[0].message.content,
                completion_usage=result.usage,
                log_fields={
                    "trace_id": trace_id,
                    "cache_hit": False,
                    "stream": False,
                },
            )
            return result

        # Streaming path — check cache before submitting to the batcher.
        cache_key = make_cache_key(body) if cache is not None else None

        if cache_key is not None:
            cached = await cache.get(cache_key)
            record_cache_lookup(hit=cached is not None)
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
                    observe_latency(
                        operation="http_chat_completion_stream",
                        seconds=perf_counter() - started_at,
                    )
                    _schedule_usage_record(
                        usage_tracker,
                        api_key=api_key,
                        model=body.model,
                        prompt_tokens=prompt_tokens,
                        completion_text="".join(cached),
                        completion_usage=None,
                        log_fields={
                            "trace_id": trace_id,
                            "cache_hit": True,
                            "stream": True,
                        },
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_cached(),
                    media_type="text/event-stream",
                    headers={"x-trace-id": trace_id},
                )

        try:
            item = await streaming_batcher.submit(body, trace_id=trace_id)
        except QueueFullError as exc:
            observe_latency(
                operation="http_chat_completion_stream",
                seconds=perf_counter() - started_at,
            )
            log_event(
                logger,
                "request.rejected",
                trace_id=trace_id,
                cache_hit=False,
                error=repr(exc),
                stream=True,
                status_code=503,
            )
            raise HTTPException(
                status_code=503,
                detail="Request queue is full",
            ) from exc
        except Exception as exc:
            observe_latency(
                operation="http_chat_completion_stream",
                seconds=perf_counter() - started_at,
            )
            log_event(
                logger,
                "request.failed",
                trace_id=trace_id,
                cache_hit=False,
                error=repr(exc),
                stream=True,
            )
            raise
        stream_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())

        while True:
            if await request.is_disconnected():
                item.cancelled = True
                observe_latency(
                    operation="http_chat_completion_stream",
                    seconds=perf_counter() - started_at,
                )
                log_event(
                    logger,
                    "request.disconnected",
                    trace_id=trace_id,
                    cache_hit=False,
                    stream=True,
                )
                raise HTTPException(status_code=499, detail="Client disconnected")

            try:
                first_token = await asyncio.wait_for(
                    item.response_channel.get(), timeout=0.05
                )
                break
            except asyncio.TimeoutError:
                continue

        provider_name: str | None = None
        stream_model = body.model

        while isinstance(first_token, ProviderStreamMetadata):
            provider_name = first_token.provider
            stream_model = first_token.model
            while True:
                if await request.is_disconnected():
                    item.cancelled = True
                    observe_latency(
                        operation="http_chat_completion_stream",
                        seconds=perf_counter() - started_at,
                    )
                    log_event(
                        logger,
                        "request.disconnected",
                        trace_id=trace_id,
                        cache_hit=False,
                        stream=True,
                    )
                    raise HTTPException(status_code=499, detail="Client disconnected")

                try:
                    first_token = await asyncio.wait_for(
                        item.response_channel.get(), timeout=0.05
                    )
                    break
                except asyncio.TimeoutError:
                    continue

        if isinstance(first_token, AllProvidersFailedError):
            observe_latency(
                operation="http_chat_completion_stream",
                seconds=perf_counter() - started_at,
            )
            log_event(
                logger,
                "request.failed",
                trace_id=trace_id,
                cache_hit=False,
                error=repr(first_token),
                stream=True,
                status_code=503,
            )
            return JSONResponse(
                all_providers_error_body(first_token),
                status_code=503,
                headers={"x-trace-id": trace_id},
            )

        if isinstance(first_token, RetryExhaustedError):
            observe_latency(
                operation="http_chat_completion_stream",
                seconds=perf_counter() - started_at,
            )
            log_event(
                logger,
                "request.failed",
                trace_id=trace_id,
                cache_hit=False,
                error=repr(first_token.original),
                retries_attempted=first_token.retries_attempted,
                stream=True,
            )
            _raise_retry_exhausted(first_token, trace_id=trace_id)

        if isinstance(first_token, BaseException):
            observe_latency(
                operation="http_chat_completion_stream",
                seconds=perf_counter() - started_at,
            )
            log_event(
                logger,
                "request.failed",
                trace_id=trace_id,
                cache_hit=False,
                error=repr(first_token),
                stream=True,
            )
            raise first_token

        async def generate():
            collected: list[str] = []
            token: object = first_token
            try:
                while True:
                    if token is _NO_STREAM_ITEM:
                        if await request.is_disconnected():
                            item.cancelled = True
                            observe_latency(
                                operation="http_chat_completion_stream",
                                seconds=perf_counter() - started_at,
                            )
                            log_event(
                                logger,
                                "request.disconnected",
                                trace_id=trace_id,
                                cache_hit=False,
                                stream=True,
                            )
                            return

                        try:
                            token = await asyncio.wait_for(
                                item.response_channel.get(), timeout=0.05
                            )
                        except asyncio.TimeoutError:
                            continue

                    if token is None:
                        break

                    if isinstance(token, ProviderStreamMetadata):
                        token = _NO_STREAM_ITEM
                        continue

                    if isinstance(token, BaseException):
                        observe_latency(
                            operation="http_chat_completion_stream",
                            seconds=perf_counter() - started_at,
                        )
                        log_event(
                            logger,
                            "request.failed",
                            trace_id=trace_id,
                            cache_hit=False,
                            error=repr(token),
                            stream=True,
                        )
                        raise token

                    assert isinstance(token, str)
                    collected.append(token)
                    chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": stream_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    token = _NO_STREAM_ITEM

                if cache_key is not None and collected:
                    await cache.set(cache_key, collected)

                observe_latency(
                    operation="http_chat_completion_stream",
                    seconds=perf_counter() - started_at,
                )
                _schedule_usage_record(
                    usage_tracker,
                    api_key=api_key,
                    model=stream_model,
                    prompt_tokens=prompt_tokens,
                    completion_text="".join(collected),
                    completion_usage=None,
                    log_fields={
                        "trace_id": trace_id,
                        "cache_hit": False,
                        "stream": True,
                    },
                )
                yield "data: [DONE]\n\n"
            finally:
                # Ensure cleanup even if the parent task is cancelled.
                item.cancelled = True

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                **({"x-trace-id": trace_id}),
                **(
                    {_PROVIDER_USED_HEADER: provider_name}
                    if provider_name is not None
                    else {}
                ),
            },
        )

    return router
