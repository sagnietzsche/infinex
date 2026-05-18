from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from services.queue import QueueFullError, RequestQueueProtocol

router = APIRouter()


def get_request_queue(request: Request) -> RequestQueueProtocol:
    return request.app.state.request_queue


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: dict[str, Any], queue: RequestQueueProtocol = Depends(get_request_queue)
) -> Any:
    try:
        item = await queue.enqueue(payload)
    except QueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Request queue is full",
        ) from exc

    return await item.future
