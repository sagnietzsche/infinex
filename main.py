from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hmac
import logging
import os
import signal
from types import FrameType

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from api.admin import create_admin_router
from api.routes import create_router
from core.config import Settings, load_settings
from core.priority import normalize_priority
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
from services.shutdown import ShutdownDrain
from services.usage import UsageTracker
from services.virtual_keys import VirtualKeyMetadata, VirtualKeyStore


def _priority_for_virtual_key(metadata: VirtualKeyMetadata):
    if metadata.priority is not None:
        return normalize_priority(metadata.priority)
    if (metadata.tier or "").lower() == "premium":
        return "high"
    return None


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
    auth_enabled = bool(settings.allowed_api_keys or settings.admin_api_key)
    rate_limiter = (
        RedisTokenBucketRateLimiter(
            redis_url=settings.redis_url,
            capacity=settings.rate_limit_capacity,
            refill_rate_per_second=settings.rate_limit_refill_per_second,
        )
        if auth_enabled
        else None
    )
    virtual_key_store = VirtualKeyStore(redis_url=settings.redis_url)
    usage_tracker = UsageTracker(redis_url=settings.redis_url)
    shutdown_drain = ShutdownDrain(
        timeout_seconds=settings.shutdown_drain_timeout_seconds
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(signum: int, frame: FrameType | None) -> None:
            shutdown_drain.initiate()
            if callable(previous_sigterm_handler):
                previous_sigterm_handler(signum, frame)

        try:
            signal.signal(signal.SIGTERM, handle_sigterm)
        except ValueError:
            previous_sigterm_handler = None

        streaming_batcher.start()
        try:
            yield
        finally:
            if previous_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
            if shutdown_drain.is_shutting_down:
                await shutdown_drain.wait_until_complete()
            await batcher.close()
            await streaming_batcher.close()
            if cache is not None:
                await cache.close()
            if rate_limiter is not None:
                await rate_limiter.close()
            await virtual_key_store.close()
            await circuit_breaker.close()
            await usage_tracker.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.batcher = batcher
    app.state.streaming_batcher = streaming_batcher
    app.state.cache = cache
    app.state.rate_limiter = rate_limiter
    app.state.virtual_key_store = virtual_key_store
    app.state.circuit_breaker = circuit_breaker
    app.state.usage_tracker = usage_tracker
    app.state.shutdown_drain = shutdown_drain

    @app.middleware("http")
    async def api_key_rate_limit_middleware(
        request: Request, call_next
    ):
        if (
            request.url.path == "/health"
            or request.url.path.startswith("/admin")
            or request.url.path.startswith("/metrics")
            or not auth_enabled
        ):
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if api_key is None:
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )

        virtual_metadata: VirtualKeyMetadata | None = None
        static_key_allowed = any(
            hmac.compare_digest(api_key, allowed_key)
            for allowed_key in settings.allowed_api_keys
        )
        if not static_key_allowed:
            virtual_key = (
                await virtual_key_store.get(api_key)
                if settings.admin_api_key
                else None
            )
            if virtual_key is None:
                return JSONResponse(
                    {"detail": "Invalid or missing API key"},
                    status_code=401,
                )
            if virtual_key.revoked:
                return JSONResponse(
                    {"detail": "API key has been revoked"},
                    status_code=403,
                )
            virtual_metadata = virtual_key.metadata

        assert rate_limiter is not None
        result = await rate_limiter.check(
            api_key,
            capacity=(
                virtual_metadata.rate_limit_capacity
                if virtual_metadata is not None
                else None
            ),
            refill_rate_per_second=(
                virtual_metadata.rate_limit_refill_per_second
                if virtual_metadata is not None
                else None
            ),
        )
        if not result.allowed:
            return JSONResponse(
                {"detail": "Too Many Requests"},
                status_code=429,
                headers={"Retry-After": str(result.retry_after_seconds)},
            )

        if virtual_metadata is not None:
            request.state.api_key_priority = _priority_for_virtual_key(
                virtual_metadata
            )
        else:
            request.state.api_key_priority = settings.priority_for_api_key(api_key)
        return await call_next(request)

    @app.middleware("http")
    async def shutdown_drain_middleware(request: Request, call_next):
        if shutdown_drain.is_shutting_down:
            return JSONResponse(
                {"detail": "Gateway is shutting down"},
                status_code=503,
            )

        async with shutdown_drain.track_request():
            return await call_next(request)

    app.mount("/metrics", make_asgi_app())
    app.include_router(
        create_router(
            batcher,
            streaming_batcher,
            cache,
            circuit_breaker,
            usage_tracker,
            shutdown_drain,
        )
    )
    app.include_router(
        create_admin_router(
            admin_api_key=settings.admin_api_key,
            virtual_key_store=virtual_key_store,
            cache=cache,
            circuit_breaker=circuit_breaker,
            usage_tracker=usage_tracker,
        )
    )

    return app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
