from __future__ import annotations

import unittest
from unittest.mock import patch

from services.circuit_breaker import CircuitBreaker, CircuitState


class FakePipeline:
    def __init__(self, client: "FakeRedis") -> None:
        self.client = client
        self.commands: list[tuple[str, tuple[object, ...]]] = []

    def zadd(self, key: str, values: dict[str, int]) -> None:
        self.commands.append(("zadd", (key, values)))

    def expire(self, key: str, seconds: int) -> None:
        self.commands.append(("expire", (key, seconds)))

    async def execute(self) -> None:
        for command, args in self.commands:
            if command == "zadd":
                key, values = args
                assert isinstance(key, str)
                assert isinstance(values, dict)
                self.client.zsets.setdefault(key, {}).update(values)
            if command == "expire":
                key, seconds = args
                assert isinstance(key, str)
                assert isinstance(seconds, int)
                self.client.expire_calls.append((key, seconds))


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, int]] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.closed = False

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def eval(self, script: str, keys: int, key: str, now_ms: int) -> list[object]:
        state_hash = self.hashes.get(key, {})
        state = state_hash.get("state")
        if state is None or state == "CLOSED":
            return [1, "CLOSED", 0]
        if state == "OPEN":
            opened_until_ms = int(state_hash.get("opened_until_ms", "0"))
            if now_ms >= opened_until_ms:
                state_hash["state"] = "HALF_OPEN"
                state_hash["probe_in_flight"] = "1"
                self.hashes[key] = state_hash
                return [1, "HALF_OPEN", 1]
            return [0, "OPEN", 0]
        if state == "HALF_OPEN":
            if state_hash.get("probe_in_flight") in {None, "0"}:
                state_hash["probe_in_flight"] = "1"
                self.hashes[key] = state_hash
                return [1, "HALF_OPEN", 1]
            return [0, "HALF_OPEN", 0]
        return [1, "CLOSED", 0]

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def zcount(self, key: str, minimum: int, maximum: int) -> int:
        return sum(
            1 for score in self.zsets.get(key, {}).values()
            if minimum <= score <= maximum
        )

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.hashes.pop(key, None)
            self.zsets.pop(key, None)

    async def aclose(self) -> None:
        self.closed = True


class CircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    def _make_breaker(self) -> tuple[CircuitBreaker, FakeRedis]:
        client = FakeRedis()
        with patch("services.circuit_breaker.aioredis.from_url", return_value=client):
            breaker = CircuitBreaker(
                redis_url="redis://localhost:6379",
                providers=("openai", "anthropic"),
                error_threshold=0.5,
                window_seconds=60,
                cooldown_seconds=30,
            )
        return breaker, client

    async def test_failure_over_threshold_opens_provider_circuit(self) -> None:
        breaker, client = self._make_breaker()

        await breaker.record_failure("openai")
        decision = await breaker.before_request("openai")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, CircuitState.OPEN)
        self.assertIn(("cb:openai:total", 60), client.expire_calls)
        self.assertIn(("cb:openai:errors", 60), client.expire_calls)

    async def test_open_circuit_moves_to_half_open_for_single_probe(self) -> None:
        breaker, client = self._make_breaker()
        await breaker.record_failure("openai")
        client.hashes["cb:openai:state"]["opened_until_ms"] = "0"

        first = await breaker.before_request("openai")
        second = await breaker.before_request("openai")

        self.assertTrue(first.allowed)
        self.assertTrue(first.probe)
        self.assertEqual(first.state, CircuitState.HALF_OPEN)
        self.assertFalse(second.allowed)
        self.assertEqual(second.state, CircuitState.HALF_OPEN)

    async def test_successful_half_open_probe_closes_circuit(self) -> None:
        breaker, client = self._make_breaker()
        await breaker.record_failure("openai")
        client.hashes["cb:openai:state"]["opened_until_ms"] = "0"
        await breaker.before_request("openai")

        await breaker.record_success("openai")
        snapshot = await breaker.snapshot("openai")

        self.assertEqual(snapshot.state, CircuitState.CLOSED)
        self.assertEqual(snapshot.total, 0)

    async def test_failed_half_open_probe_reopens_circuit(self) -> None:
        breaker, client = self._make_breaker()
        await breaker.record_failure("openai")
        client.hashes["cb:openai:state"]["opened_until_ms"] = "0"
        await breaker.before_request("openai")

        await breaker.record_failure("openai")
        snapshot = await breaker.snapshot("openai")

        self.assertEqual(snapshot.state, CircuitState.OPEN)
        self.assertGreater(
            int(client.hashes["cb:openai:state"]["opened_until_ms"]),
            0,
        )

    async def test_provider_snapshots_are_independent(self) -> None:
        breaker, _ = self._make_breaker()

        await breaker.record_failure("openai")
        snapshots = await breaker.snapshots()

        self.assertEqual(snapshots["openai"].state, CircuitState.OPEN)
        self.assertEqual(snapshots["anthropic"].state, CircuitState.CLOSED)
