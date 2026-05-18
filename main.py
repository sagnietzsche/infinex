import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from api.routes import router
from services.queue import (
    AsyncExecutor,
    InMemoryQueueConsumer,
    InMemoryRequestQueue,
    KafkaQueueConsumer,
    KafkaRequestQueue,
    KafkaResponseConsumer,
)

DEFAULT_QUEUE_MAXSIZE = 100
DEFAULT_QUEUE_BACKEND = "kafka"
DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_KAFKA_REQUEST_TOPIC = "llm-gateway.requests"
DEFAULT_KAFKA_RESPONSE_TOPIC = "llm-gateway.responses"
DEFAULT_KAFKA_CONSUMER_GROUP = "llm-gateway-workers"
Executor = AsyncExecutor | Callable[[dict[str, Any]], Awaitable[Any]]


class EchoLLMExecutor:
    """Placeholder LLM executor used until provider execution is wired in."""

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = payload.get("prompt", "")
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                content = last_message.get("content", content)

        return {
            "id": "queued-response",
            "object": "chat.completion",
            "model": payload.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }


def get_queue_maxsize() -> int:
    raw_value = os.getenv("REQUEST_QUEUE_MAXSIZE", str(DEFAULT_QUEUE_MAXSIZE))

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError("REQUEST_QUEUE_MAXSIZE must be an integer") from exc


def get_queue_backend() -> str:
    return os.getenv("REQUEST_QUEUE_BACKEND", DEFAULT_QUEUE_BACKEND).lower()


def create_app(
    *,
    queue_maxsize: int | None = None,
    executor: Executor | None = None,
    start_consumer: bool = True,
    queue_backend: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        backend = (queue_backend or get_queue_backend()).lower()
        maxsize = queue_maxsize if queue_maxsize is not None else get_queue_maxsize()
        llm_executor = executor or EchoLLMExecutor()

        if backend == "kafka":
            bootstrap_servers = os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", DEFAULT_KAFKA_BOOTSTRAP_SERVERS
            )
            request_topic = os.getenv(
                "KAFKA_REQUEST_TOPIC", DEFAULT_KAFKA_REQUEST_TOPIC
            )
            response_topic = os.getenv(
                "KAFKA_RESPONSE_TOPIC", DEFAULT_KAFKA_RESPONSE_TOPIC
            )
            consumer_group = os.getenv(
                "KAFKA_CONSUMER_GROUP", DEFAULT_KAFKA_CONSUMER_GROUP
            )
            gateway_id = os.getenv("GATEWAY_INSTANCE_ID", str(uuid.uuid4()))

            app.state.request_queue = KafkaRequestQueue(
                maxsize=maxsize,
                bootstrap_servers=bootstrap_servers,
                request_topic=request_topic,
                gateway_id=gateway_id,
            )
            app.state.queue_consumer = KafkaQueueConsumer(
                executor=llm_executor,
                bootstrap_servers=bootstrap_servers,
                request_topic=request_topic,
                response_topic=response_topic,
                group_id=consumer_group,
            )
            app.state.response_consumer = KafkaResponseConsumer(
                queue=app.state.request_queue,
                bootstrap_servers=bootstrap_servers,
                response_topic=response_topic,
                group_id=f"llm-gateway-responses-{gateway_id}",
                gateway_id=gateway_id,
            )
            await app.state.request_queue.start()
        elif backend == "memory":
            app.state.request_queue = InMemoryRequestQueue(maxsize=maxsize)
            app.state.queue_consumer = InMemoryQueueConsumer(
                app.state.request_queue,
                llm_executor,
            )
            app.state.response_consumer = None
        else:
            raise RuntimeError(f"Unsupported REQUEST_QUEUE_BACKEND: {backend}")

        if start_consumer:
            if app.state.response_consumer is not None:
                await app.state.response_consumer.start()
            consumer_start = app.state.queue_consumer.start()
            if isinstance(consumer_start, Awaitable):
                await consumer_start

        try:
            yield
        finally:
            if start_consumer:
                await app.state.queue_consumer.stop()
                if app.state.response_consumer is not None:
                    await app.state.response_consumer.stop()
            if backend == "kafka":
                await app.state.request_queue.stop()

    app = FastAPI(title="llm-gateway", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
