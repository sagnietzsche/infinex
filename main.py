from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hmac
import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from api.routes import create_router
from core.config import Settings, load_settings
from infra.providers import (
    build_provider_for_name,
    build_streaming_provider_for_name,
)
from services.batcher import AsyncRequestBatcher, DynamicBatcher
from services.cache import ResponseCache
from services.circuit_breaker import CircuitBreaker
from services.provider_router import ProviderRoute, ProviderRouter
from services.rate_limit import RedisTokenBucketRateLimiter
from services.retry import RetryPolicy
from services.usage import UsageTracker


def create_app(settings: Settings | None = None) -> FastAPI:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(message)s",
    )
    settings = settings or load_settings()
    retry_policy = RetryPolicy(
        max_retries=settings.max_retries,
        base_delay_ms=settings.retry_base_delay_ms,
        max_delay_ms=settings.retry_max_delay_ms,
    )
    provider_routes = []
    for provider_name in settings.provider_fallback_chain:
        provider_routes.append(
            ProviderRoute(
                name=provider_name,
                provider=retry_policy.wrap_provider(
                    build_provider_for_name(settings, provider_name)
                ),
                streaming_provider=retry_policy.wrap_streaming_provider(
                    build_streaming_provider_for_name(settings, provider_name)
                ),
            )
        )
    circuit_breaker = CircuitBreaker(
        redis_url=settings.redis_url,
        providers=settings.provider_fallback_chain,
        error_threshold=settings.cb_error_threshold,
        window_seconds=settings.cb_window_seconds,
        cooldown_seconds=settings.cb_cooldown_seconds,
    )
    provider_router = ProviderRouter(
        routes=provider_routes,
        model_mapping=settings.provider_model_mapping,
        circuit_breaker=circuit_breaker,
    )
    batcher = AsyncRequestBatcher(
        provider=provider_router,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
        max_queue_size=settings.batch_queue_max_size,
    )
    streaming_batcher = DynamicBatcher(
        provider=provider_router,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
        max_queue_size=settings.batch_queue_max_size,
    )
    cache = (
        ResponseCache(
            redis_url=settings.redis_url,
            ttl_seconds=settings.cache_ttl_seconds,
            semantic_enabled=settings.semantic_cache_enabled,
            semantic_ttl_seconds=settings.semantic_cache_ttl_seconds,
            semantic_threshold=settings.semantic_cache_threshold,
            semantic_embedding_model=settings.semantic_cache_embedding_model,
            semantic_embedding_dimension=(
                settings.semantic_cache_embedding_dimension
            ),
            openai_api_key=settings.openai_api_key,
        )
        if settings.cache_enabled
        else None
    )
    rate_limiter = (
        RedisTokenBucketRateLimiter(
            redis_url=settings.redis_url,
            capacity=settings.rate_limit_capacity,
            refill_rate_per_second=settings.rate_limit_refill_per_second,
        )
        if settings.allowed_api_keys
        else None
    )
    usage_tracker = UsageTracker(redis_url=settings.redis_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        streaming_batcher.start()
        yield
        await batcher.close()
        await streaming_batcher.close()
        if cache is not None:
            await cache.close()
        if rate_limiter is not None:
            await rate_limiter.close()
        await circuit_breaker.close()
        await usage_tracker.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.batcher = batcher
    app.state.streaming_batcher = streaming_batcher
    app.state.cache = cache
    app.state.rate_limiter = rate_limiter
    app.state.circuit_breaker = circuit_breaker
    app.state.usage_tracker = usage_tracker

    @app.middleware("http")
    async def api_key_rate_limit_middleware(
        request: Request, call_next
    ):
        if (
            request.url.path == "/health"
            or request.url.path.startswith("/metrics")
            or rate_limiter is None
        ):
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if api_key is None or not any(
            hmac.compare_digest(api_key, allowed_key)
            for allowed_key in settings.allowed_api_keys
        ):
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )

        result = await rate_limiter.check(api_key)
        if not result.allowed:
            return JSONResponse(
                {"detail": "Too Many Requests"},
                status_code=429,
                headers={"Retry-After": str(result.retry_after_seconds)},
            )

        request.state.api_key_priority = settings.priority_for_api_key(api_key)
        return await call_next(request)

    app.mount("/metrics", make_asgi_app())
    app.include_router(
        create_router(
            batcher,
            streaming_batcher,
            cache,
            circuit_breaker,
            usage_tracker,
        )
    )

    return app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
