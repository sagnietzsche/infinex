from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from services.cache import ResponseCache
from services.circuit_breaker import CircuitBreaker
from services.usage import UsageTracker
from services.virtual_keys import VirtualKeyMetadata, VirtualKeyStore


class CreateVirtualKeyRequest(BaseModel):
    label: str | None = None
    tier: str | None = None
    priority: str | None = None
    rate_limit_capacity: int | None = Field(default=None, gt=0)
    rate_limit_refill_per_second: float | None = Field(default=None, gt=0)


class CreateVirtualKeyResponse(BaseModel):
    key: str
    metadata: dict[str, str | int | float]


def verify_admin_key_dependency(admin_api_key: str | None):
    async def verify_admin_key(
        x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
    ) -> None:
        if (
            not admin_api_key
            or x_admin_key is None
            or not hmac.compare_digest(x_admin_key, admin_api_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing admin key",
            )

    return verify_admin_key


def create_admin_router(
    *,
    admin_api_key: str | None,
    virtual_key_store: VirtualKeyStore,
    cache: ResponseCache | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    usage_tracker: UsageTracker | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix="/admin",
        dependencies=[Depends(verify_admin_key_dependency(admin_api_key))],
    )

    @router.post("/keys", response_model=CreateVirtualKeyResponse)
    async def create_key(body: CreateVirtualKeyRequest) -> CreateVirtualKeyResponse:
        metadata = VirtualKeyMetadata(
            label=body.label,
            tier=body.tier,
            priority=body.priority,
            rate_limit_capacity=body.rate_limit_capacity,
            rate_limit_refill_per_second=body.rate_limit_refill_per_second,
        )
        raw_key = await virtual_key_store.create(metadata)
        return CreateVirtualKeyResponse(
            key=raw_key,
            metadata=body.model_dump(exclude_none=True),
        )

    @router.delete("/keys/{key}")
    async def revoke_key(key: str) -> dict[str, bool]:
        await virtual_key_store.revoke(key)
        return {"revoked": True}

    @router.get("/keys/{key}/usage")
    async def key_usage(key: str) -> dict[str, int | float]:
        if usage_tracker is None:
            return _zero_admin_usage()
        usage = await usage_tracker.get_usage(key)
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "estimated_cost_usd": usage.estimated_cost_usd,
            "request_count": usage.request_count,
        }

    @router.post("/cache/flush")
    async def flush_cache(
        prefix: Annotated[str | None, Query()] = None,
    ) -> dict[str, int]:
        if cache is None:
            return {"deleted": 0}
        try:
            deleted = await cache.flush(prefix=prefix)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted": deleted}

    @router.get("/providers")
    async def providers() -> dict[str, dict[str, bool | int | float | str]]:
        if circuit_breaker is None:
            return {}
        snapshots = await circuit_breaker.snapshots()
        return {
            provider: {
                "active": snapshot.state.value != "OPEN",
                "circuit_state": snapshot.state.value,
                "recent_error_rate": snapshot.error_rate,
                "errors": snapshot.errors,
                "total": snapshot.total,
            }
            for provider, snapshot in snapshots.items()
        }

    return router


def _zero_admin_usage() -> dict[str, int | float]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost_usd": 0.0,
        "request_count": 0,
    }
