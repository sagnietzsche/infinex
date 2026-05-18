import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer


class QueueFullError(Exception):
    """Raised when a request cannot be accepted because the queue is full."""


@dataclass(slots=True)
class RequestItem:
    payload: dict[str, Any]
    future: asyncio.Future[Any]
    request_id: str


class RequestQueueProtocol(Protocol):
    @property
    def maxsize(self) -> int:
        ...

    @property
    def size(self) -> int:
        ...

    async def enqueue(self, payload: dict[str, Any]) -> RequestItem:
        ...


class AsyncExecutor(Protocol):
    def execute(self, payload: dict[str, Any]) -> Awaitable[Any]:
        ...


class InMemoryRequestQueue:
    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be greater than 0")

        self._queue: asyncio.Queue[RequestItem] = asyncio.Queue(maxsize=maxsize)

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def size(self) -> int:
        return self._queue.qsize()

    async def enqueue(self, payload: dict[str, Any]) -> RequestItem:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        item = RequestItem(
            payload=payload,
            future=future,
            request_id=str(uuid.uuid4()),
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            future.cancel()
            raise QueueFullError("request queue is full") from exc

        return item

    async def get(self) -> RequestItem:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()


class KafkaRequestQueue:
    def __init__(
        self,
        *,
        maxsize: int,
        bootstrap_servers: str,
        request_topic: str,
        gateway_id: str,
        producer_factory: Callable[..., Any] = AIOKafkaProducer,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be greater than 0")

        self._maxsize = maxsize
        self._bootstrap_servers = bootstrap_servers
        self._request_topic = request_topic
        self._gateway_id = gateway_id
        self._producer_factory = producer_factory
        self._request_id_factory = request_id_factory or (lambda: str(uuid.uuid4()))
        self._producer: Any | None = None
        self._pending: dict[str, RequestItem] = {}

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def size(self) -> int:
        return len(self._pending)

    async def start(self) -> None:
        self._producer = self._producer_factory(
            bootstrap_servers=self._bootstrap_servers,
        )
        await self._producer.start()

    async def stop(self) -> None:
        for item in self._pending.values():
            if not item.future.done():
                item.future.cancel()
        self._pending.clear()

        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def enqueue(self, payload: dict[str, Any]) -> RequestItem:
        if self._producer is None:
            raise RuntimeError("KafkaRequestQueue must be started before enqueue")

        if len(self._pending) >= self._maxsize:
            raise QueueFullError("request queue is full")

        request_id = self._request_id_factory()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        item = RequestItem(payload=payload, future=future, request_id=request_id)
        self._pending[request_id] = item

        message = json.dumps(
            {
                "request_id": request_id,
                "gateway_id": self._gateway_id,
                "payload": payload,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        try:
            await self._producer.send_and_wait(
                self._request_topic,
                message,
                key=request_id.encode("utf-8"),
            )
        except Exception:
            self._pending.pop(request_id, None)
            future.cancel()
            raise

        return item

    def resolve(self, request_id: str, response: Any) -> bool:
        item = self._pending.pop(request_id, None)
        if item is None:
            return False

        if not item.future.done():
            item.future.set_result(response)
        return True

    def reject(self, request_id: str, exc: Exception) -> bool:
        item = self._pending.pop(request_id, None)
        if item is None:
            return False

        if not item.future.done():
            item.future.set_exception(exc)
        return True


class InMemoryQueueConsumer:
    def __init__(
        self,
        queue: InMemoryRequestQueue,
        executor: AsyncExecutor | Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self._queue = queue
        self._executor = executor
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._task = asyncio.create_task(self._run(), name="request-queue-consumer")

    async def stop(self) -> None:
        if self._task is None:
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                response = await execute_payload(self._executor, item.payload)
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(response)
            finally:
                self._queue.task_done()


class KafkaQueueConsumer:
    def __init__(
        self,
        *,
        executor: AsyncExecutor | Callable[[dict[str, Any]], Awaitable[Any]],
        bootstrap_servers: str,
        request_topic: str,
        response_topic: str,
        group_id: str,
        consumer_factory: Callable[..., Any] = AIOKafkaConsumer,
        producer_factory: Callable[..., Any] = AIOKafkaProducer,
    ) -> None:
        self._executor = executor
        self._bootstrap_servers = bootstrap_servers
        self._request_topic = request_topic
        self._response_topic = response_topic
        self._group_id = group_id
        self._consumer_factory = consumer_factory
        self._producer_factory = producer_factory
        self._consumer: Any | None = None
        self._producer: Any | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._consumer = self._consumer_factory(
            self._request_topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            enable_auto_commit=True,
            auto_offset_reset="latest",
        )
        self._producer = self._producer_factory(
            bootstrap_servers=self._bootstrap_servers,
        )
        await self._consumer.start()
        await self._producer.start()
        self._task = asyncio.create_task(self._run(), name="kafka-request-consumer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def _run(self) -> None:
        if self._consumer is None or self._producer is None:
            raise RuntimeError("KafkaQueueConsumer must be started before running")

        async for message in self._iter_messages(self._consumer):
            request_id: str | None = None
            gateway_id: str | None = None
            try:
                envelope = json.loads(message.value.decode("utf-8"))
                request_id = envelope["request_id"]
                gateway_id = envelope["gateway_id"]
                payload = envelope["payload"]
                response = await execute_payload(self._executor, payload)
            except Exception as exc:
                if request_id is not None and gateway_id is not None:
                    await self._publish_response(
                        request_id=request_id,
                        gateway_id=gateway_id,
                        error=str(exc),
                    )
            else:
                await self._publish_response(
                    request_id=request_id,
                    gateway_id=gateway_id,
                    response=response,
                )

    async def _publish_response(
        self,
        *,
        request_id: str,
        gateway_id: str,
        response: Any | None = None,
        error: str | None = None,
    ) -> None:
        if self._producer is None:
            raise RuntimeError("KafkaQueueConsumer must be started before publishing")

        envelope = {
            "request_id": request_id,
            "gateway_id": gateway_id,
            "response": response,
            "error": error,
        }
        await self._producer.send_and_wait(
            self._response_topic,
            json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
            key=gateway_id.encode("utf-8"),
        )

    async def _iter_messages(self, consumer: Any) -> AsyncIterator[Any]:
        async for message in consumer:
            yield message


class KafkaResponseConsumer:
    def __init__(
        self,
        *,
        queue: KafkaRequestQueue,
        bootstrap_servers: str,
        response_topic: str,
        group_id: str,
        gateway_id: str,
        consumer_factory: Callable[..., Any] = AIOKafkaConsumer,
    ) -> None:
        self._queue = queue
        self._bootstrap_servers = bootstrap_servers
        self._response_topic = response_topic
        self._group_id = group_id
        self._gateway_id = gateway_id
        self._consumer_factory = consumer_factory
        self._consumer: Any | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._consumer = self._consumer_factory(
            self._response_topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            enable_auto_commit=True,
            auto_offset_reset="latest",
        )
        await self._consumer.start()
        self._task = asyncio.create_task(self._run(), name="kafka-response-consumer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

    async def _run(self) -> None:
        if self._consumer is None:
            raise RuntimeError("KafkaResponseConsumer must be started before running")

        async for message in self._iter_messages(self._consumer):
            envelope = json.loads(message.value.decode("utf-8"))
            if envelope["gateway_id"] != self._gateway_id:
                continue

            request_id = envelope["request_id"]
            error = envelope.get("error")
            if error:
                self._queue.reject(request_id, RuntimeError(error))
            else:
                self._queue.resolve(request_id, envelope.get("response"))

    async def _iter_messages(self, consumer: Any) -> AsyncIterator[Any]:
        async for message in consumer:
            yield message


async def execute_payload(
    executor: AsyncExecutor | Callable[[dict[str, Any]], Awaitable[Any]],
    payload: dict[str, Any],
) -> Any:
    if hasattr(executor, "execute"):
        return await executor.execute(payload)

    return await executor(payload)


RequestQueue = InMemoryRequestQueue
QueueConsumer = InMemoryQueueConsumer
