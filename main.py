from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.routes import create_router
from core.config import Settings, load_settings
from infra.providers import build_provider, build_streaming_provider
from services.batcher import AsyncRequestBatcher, DynamicBatcher
from services.cache import ResponseCache


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    provider = build_provider(settings)
    streaming_provider = build_streaming_provider(settings)
    batcher = AsyncRequestBatcher(
        provider=provider,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
    )
    streaming_batcher = DynamicBatcher(
        provider=streaming_provider,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
    )
    cache = (
        ResponseCache(
            redis_url=settings.redis_url,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        if settings.cache_enabled
        else None
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        streaming_batcher.start()
        yield
        await batcher.close()
        await streaming_batcher.close()
        if cache is not None:
            await cache.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.batcher = batcher
    app.state.streaming_batcher = streaming_batcher
    app.state.cache = cache
    app.include_router(create_router(batcher, streaming_batcher, cache))

    return app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
