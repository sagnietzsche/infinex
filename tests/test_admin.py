from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin import create_admin_router
from core.config import Settings
from main import create_app
from services.circuit_breaker import CircuitSnapshot, CircuitState
from services.usage import UsageTotals
from services.virtual_keys import (
    VirtualKeyMetadata,
    VirtualKeyRecord,
    VirtualKeyStore,
)


class StubVirtualKeyStore:
    def __init__(self) -> None:
        self.created_metadata: VirtualKeyMetadata | None = None
        self.revoked_key: str | None = None

    async def create(self, metadata: VirtualKeyMetadata) -> str:
        self.created_metadata = metadata
        return "vg_created"

    async def revoke(self, key: str) -> None:
        self.revoked_key = key


class StubUsageTracker:
    async def get_usage(self, key: str) -> UsageTotals:
        self.key = key
        return UsageTotals(
            prompt_tokens=7,
            completion_tokens=8,
            total_tokens=15,
            estimated_cost_usd=0.123,
            request_count=4,
        )


class StubCache:
    async def flush(self, *, prefix: str | None = None) -> int:
        self.prefix = prefix
        return 3


class StubCircuitBreaker:
    async def snapshots(self) -> dict[str, CircuitSnapshot]:
        return {
            "openai": CircuitSnapshot(
                provider="openai",
                state=CircuitState.OPEN,
                error_rate=0.75,
                errors=3,
                total=4,
            ),
            "anthropic": CircuitSnapshot(
                provider="anthropic",
                state=CircuitState.CLOSED,
                error_rate=0.0,
                errors=0,
                total=2,
            ),
        }


def _admin_app(
    *,
    store: StubVirtualKeyStore | None = None,
    usage: StubUsageTracker | None = None,
    cache: StubCache | None = None,
    circuit_breaker: StubCircuitBreaker | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_admin_router(
            admin_api_key="admin-secret",
            virtual_key_store=store or StubVirtualKeyStore(),  # type: ignore[arg-type]
            usage_tracker=usage,  # type: ignore[arg-type]
            cache=cache,  # type: ignore[arg-type]
            circuit_breaker=circuit_breaker,  # type: ignore[arg-type]
        )
    )
    return app


def test_admin_routes_require_admin_key() -> None:
    with TestClient(_admin_app()) as client:
        missing = client.get("/admin/providers")
        wrong = client.get(
            "/admin/providers",
            headers={"X-Admin-Key": "wrong"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_create_key_returns_generated_key_once_and_persists_metadata() -> None:
    store = StubVirtualKeyStore()
    with TestClient(_admin_app(store=store)) as client:
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Key": "admin-secret"},
            json={
                "label": "prod",
                "tier": "premium",
                "rate_limit_capacity": 10,
                "rate_limit_refill_per_second": 2.5,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "key": "vg_created",
        "metadata": {
            "label": "prod",
            "tier": "premium",
            "rate_limit_capacity": 10,
            "rate_limit_refill_per_second": 2.5,
        },
    }
    assert store.created_metadata == VirtualKeyMetadata(
        label="prod",
        tier="premium",
        rate_limit_capacity=10,
        rate_limit_refill_per_second=2.5,
    )


def test_revoke_key_marks_key_revoked() -> None:
    store = StubVirtualKeyStore()
    with TestClient(_admin_app(store=store)) as client:
        response = client.delete(
            "/admin/keys/vg_created",
            headers={"X-Admin-Key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"revoked": True}
    assert store.revoked_key == "vg_created"


def test_admin_key_usage_endpoint_returns_lifetime_totals() -> None:
    usage = StubUsageTracker()
    with TestClient(_admin_app(usage=usage)) as client:
        response = client.get(
            "/admin/keys/vg_created/usage",
            headers={"X-Admin-Key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "prompt_tokens": 7,
        "completion_tokens": 8,
        "estimated_cost_usd": 0.123,
        "request_count": 4,
    }
    assert usage.key == "vg_created"


def test_cache_flush_forwards_optional_prefix() -> None:
    cache = StubCache()
    with TestClient(_admin_app(cache=cache)) as client:
        response = client.post(
            "/admin/cache/flush?prefix=llm-gateway:v1:test",
            headers={"X-Admin-Key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"deleted": 3}
    assert cache.prefix == "llm-gateway:v1:test"


def test_cache_flush_returns_400_for_invalid_prefix() -> None:
    class RejectingCache:
        async def flush(self, *, prefix: str | None = None) -> int:
            raise ValueError("cache flush prefix must target response cache keys")

    with TestClient(_admin_app(cache=RejectingCache())) as client:  # type: ignore[arg-type]
        response = client.post(
            "/admin/cache/flush?prefix=admin:key:",
            headers={"X-Admin-Key": "admin-secret"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "cache flush prefix must target response cache keys"
    }


def test_provider_state_returns_active_circuit_and_error_rate() -> None:
    with TestClient(_admin_app(circuit_breaker=StubCircuitBreaker())) as client:
        response = client.get(
            "/admin/providers",
            headers={"X-Admin-Key": "admin-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "openai": {
            "active": False,
            "circuit_state": "OPEN",
            "recent_error_rate": 0.75,
            "errors": 3,
            "total": 4,
        },
        "anthropic": {
            "active": True,
            "circuit_state": "CLOSED",
            "recent_error_rate": 0.0,
            "errors": 0,
            "total": 2,
        },
    }


def test_virtual_key_store_hashes_key_before_storing_metadata() -> None:
    mock_client = MagicMock()
    mock_client.hset = AsyncMock()
    mock_client.aclose = AsyncMock()
    with patch("services.virtual_keys.secrets.token_urlsafe", return_value="raw"), patch(
        "services.virtual_keys.aioredis.from_url",
        return_value=mock_client,
    ):
        store = VirtualKeyStore(redis_url="redis://localhost:6379")
        raw_key = asyncio.run(
            store.create(VirtualKeyMetadata(label="prod", tier="premium"))
        )

    hashed_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    assert raw_key == "vg_raw"
    assert mock_client.hset.call_args.args[0] == f"admin:key:{hashed_key}"
    assert mock_client.hset.call_args.kwargs["mapping"]["label"] == "prod"
    assert mock_client.hset.call_args.kwargs["mapping"]["tier"] == "premium"
    assert raw_key not in str(mock_client.hset.call_args)


def test_revoked_virtual_key_returns_403_on_regular_requests() -> None:
    class FakeStore:
        async def get(self, raw_key: str) -> VirtualKeyRecord:
            return VirtualKeyRecord(
                hashed_key=hashlib.sha256(raw_key.encode()).hexdigest(),
                metadata=VirtualKeyMetadata(),
                revoked=True,
            )

        async def close(self) -> None:
            pass

    class FakeLimiter:
        async def check(self, api_key: str, **kwargs):
            raise AssertionError("revoked keys should not reach rate limiter")

        async def close(self) -> None:
            pass

    class FakeCircuitBreaker:
        async def before_request(self, provider: str):
            from services.circuit_breaker import CircuitDecision

            return CircuitDecision(allowed=True, state=CircuitState.CLOSED)

        async def record_success(self, provider: str) -> None:
            pass

        async def record_failure(self, provider: str) -> None:
            pass

        async def snapshots(self) -> dict[str, CircuitSnapshot]:
            return {}

        async def close(self) -> None:
            pass

    class FakeUsageTracker:
        async def count_prompt_tokens(self, request):
            return 0

        async def record_request(self, **kwargs):
            return UsageTotals(0, 0, 0, 0.0)

        async def get_usage(self, key: str):
            return UsageTotals(0, 0, 0, 0.0)

        async def close(self) -> None:
            pass

    with patch("main.VirtualKeyStore", return_value=FakeStore()), patch(
        "main.RedisTokenBucketRateLimiter",
        return_value=FakeLimiter(),
    ), patch("main.CircuitBreaker", return_value=FakeCircuitBreaker()), patch(
        "main.UsageTracker",
        return_value=FakeUsageTracker(),
    ):
        app = create_app(Settings(cache_enabled=False, admin_api_key="admin-secret"))
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": "vg_revoked"},
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

    assert response.status_code == 403
    assert response.json() == {"detail": "API key has been revoked"}
