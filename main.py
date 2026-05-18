from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.routes import create_router
from core.config import Settings, load_settings
from infra.providers import build_provider
from services.batcher import AsyncRequestBatcher


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    provider = build_provider(settings)
    batcher = AsyncRequestBatcher(
        provider=provider,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await batcher.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.batcher = batcher
    app.include_router(create_router(batcher))

    return app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
