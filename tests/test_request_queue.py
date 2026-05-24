import asyncio
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import sleep
from typing import Any

from fastapi.testclient import TestClient

from core.config import Settings
from main import create_app
from services.queue import (
    InMemoryQueueConsumer,
    InMemoryRequestQueue,
    KafkaQueueConsumer,
    KafkaRequestQueue,
    KafkaResponseConsumer,
    QueueFullError,
)


class StaticExecutor:
    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"received": payload}


class FakeProducer:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.sent: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_and_wait(self, topic: str, value: bytes, key: bytes) -> None:
        self.sent.append({"topic": topic, "value": value, "key": key})


@dataclass(slots=True)
class FakeMessage:
    value: bytes


class FakeConsumer:
    def __init__(self, *topics: str, messages: list[FakeMessage], **kwargs: Any) -> None:
        self.topics = topics
        self.kwargs = kwargs
        self.messages = messages
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def __aiter__(self) -> "FakeConsumer":
        return self

    async def __anext__(self) -> FakeMessage:
        if not self.messages:
            raise StopAsyncIteration

        return self.messages.pop(0)


class InMemoryRequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_returns_item_with_pending_future(self) -> None:
        queue = InMemoryRequestQueue(maxsize=1)

        item = await queue.enqueue({"messages": []})

        self.assertEqual(item.payload, {"messages": []})
        self.assertFalse(item.future.done())
        self.assertEqual(queue.size, 1)

    async def test_enqueue_raises_when_full(self) -> None:
        queue = InMemoryRequestQueue(maxsize=1)
        await queue.enqueue({"first": True})

        with self.assertRaises(QueueFullError):
            await queue.enqueue({"second": True})

    async def test_get_returns_high_priority_before_older_low_priority(self) -> None:
        queue = InMemoryRequestQueue(maxsize=3)
        low = await queue.enqueue({"name": "low", "priority": "low"})
        normal = await queue.enqueue({"name": "normal"})
        high = await queue.enqueue({"name": "high", "priority": "high"})

        self.assertEqual((await queue.get()).request_id, high.request_id)
        self.assertEqual((await queue.get()).request_id, normal.request_id)
        self.assertEqual((await queue.get()).request_id, low.request_id)

    async def test_consumer_resolves_future_with_executor_response(self) -> None:
        queue = InMemoryRequestQueue(maxsize=1)
        consumer = InMemoryQueueConsumer(queue, StaticExecutor())
        consumer.start()

        try:
            item = await queue.enqueue({"model": "test"})
            response = await asyncio.wait_for(item.future, timeout=1)
        finally:
            await consumer.stop()

        self.assertEqual(response, {"received": {"model": "test"}})


class KafkaRequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_publishes_request_envelope_to_kafka(self) -> None:
        producer = FakeProducer()
        queue = KafkaRequestQueue(
            maxsize=1,
            bootstrap_servers="localhost:9092",
            request_topic="requests",
            gateway_id="gateway-1",
            producer_factory=lambda **kwargs: producer,
            request_id_factory=lambda: "req-1",
        )

        await queue.start()
        try:
            item = await queue.enqueue({"model": "test"})
        finally:
            await queue.stop()

        self.assertEqual(item.request_id, "req-1")
        self.assertEqual(producer.sent[0]["topic"], "requests")
        self.assertEqual(producer.sent[0]["key"], b"req-1")
        self.assertEqual(
            json.loads(producer.sent[0]["value"].decode("utf-8")),
            {
                "request_id": "req-1",
                "gateway_id": "gateway-1",
                "payload": {"model": "test"},
            },
        )

    async def test_enqueue_raises_when_pending_futures_reach_maxsize(self) -> None:
        queue = KafkaRequestQueue(
            maxsize=1,
            bootstrap_servers="localhost:9092",
            request_topic="requests",
            gateway_id="gateway-1",
            producer_factory=lambda **kwargs: FakeProducer(),
            request_id_factory=lambda: "req-1",
        )

        await queue.start()
        try:
            await queue.enqueue({"first": True})
            with self.assertRaises(QueueFullError):
                await queue.enqueue({"second": True})
        finally:
            await queue.stop()

    async def test_request_consumer_publishes_executor_response(self) -> None:
        response_producer = FakeProducer()
        envelope = json.dumps(
            {
                "request_id": "req-1",
                "gateway_id": "gateway-1",
                "payload": {"model": "test"},
            }
        ).encode("utf-8")
        fake_consumer = FakeConsumer("requests", messages=[FakeMessage(envelope)])

        consumer = KafkaQueueConsumer(
            executor=StaticExecutor(),
            bootstrap_servers="localhost:9092",
            request_topic="requests",
            response_topic="responses",
            group_id="workers",
            consumer_factory=lambda *args, **kwargs: fake_consumer,
            producer_factory=lambda **kwargs: response_producer,
        )

        await consumer.start()
        await asyncio.wait_for(consumer._task, timeout=1)
        await consumer.stop()

        self.assertEqual(response_producer.sent[0]["topic"], "responses")
        self.assertEqual(response_producer.sent[0]["key"], b"gateway-1")
        self.assertEqual(
            json.loads(response_producer.sent[0]["value"].decode("utf-8")),
            {
                "request_id": "req-1",
                "gateway_id": "gateway-1",
                "response": {"received": {"model": "test"}},
                "error": None,
            },
        )

    async def test_response_consumer_resolves_matching_pending_future(self) -> None:
        producer = FakeProducer()
        queue = KafkaRequestQueue(
            maxsize=1,
            bootstrap_servers="localhost:9092",
            request_topic="requests",
            gateway_id="gateway-1",
            producer_factory=lambda **kwargs: producer,
            request_id_factory=lambda: "req-1",
        )
        await queue.start()
        item = await queue.enqueue({"model": "test"})

        envelope = json.dumps(
            {
                "request_id": "req-1",
                "gateway_id": "gateway-1",
                "response": {"received": {"model": "test"}},
                "error": None,
            }
        ).encode("utf-8")
        fake_consumer = FakeConsumer("requests", messages=[FakeMessage(envelope)])

        consumer = KafkaResponseConsumer(
            queue=queue,
            bootstrap_servers="localhost:9092",
            response_topic="responses",
            group_id="gateway-1-responses",
            gateway_id="gateway-1",
            consumer_factory=lambda *args, **kwargs: fake_consumer,
        )

        await consumer.start()
        try:
            response = await asyncio.wait_for(item.future, timeout=1)
        finally:
            await consumer.stop()
            await queue.stop()

        self.assertEqual(response, {"received": {"model": "test"}})


class RouteTests(unittest.TestCase):
    def test_chat_completions_returns_batcher_result(self) -> None:
        app = create_app(
            Settings(
                cache_enabled=False,
                batch_max_wait_ms=5,
            )
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "Echo: test",
        )

    def test_chat_completions_returns_503_when_queue_is_full(self) -> None:
        app = create_app(
            Settings(
                cache_enabled=False,
                batch_max_size=10,
                batch_max_wait_ms=500,
                batch_queue_max_size=1,
            )
        )
        payload = {"messages": [{"role": "user", "content": "test"}]}

        with TestClient(app) as client:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(
                    lambda: client.post("/v1/chat/completions", json=payload)
                )

                for _ in range(20):
                    stats = client.get("/stats").json()
                    if stats["queued_requests"] == 1:
                        break
                    sleep(0.01)

                response = client.post("/v1/chat/completions", json=payload)
                first_response = first.result(timeout=2)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Request queue is full"})
        self.assertEqual(first_response.status_code, 200)
