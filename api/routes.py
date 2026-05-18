from fastapi import APIRouter, HTTPException

from core.models import ChatCompletionRequest, ChatCompletionResponse
from services.batcher import AsyncRequestBatcher, BatcherStats


def create_router(batcher: AsyncRequestBatcher) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/stats", response_model=BatcherStats)
    async def stats() -> BatcherStats:
        return batcher.stats()

    @router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming responses are planned for feature 6.",
            )

        return await batcher.submit(request)

    return router
