from __future__ import annotations

from dataclasses import dataclass
import hashlib
import secrets
import time

import redis.asyncio as aioredis


@dataclass(frozen=True)
class VirtualKeyMetadata:
    label: str | None = None
    tier: str | None = None
    priority: str | None = None
    rate_limit_capacity: int | None = None
    rate_limit_refill_per_second: float | None = None


@dataclass(frozen=True)
class VirtualKeyRecord:
    hashed_key: str
    metadata: VirtualKeyMetadata
    revoked: bool


class VirtualKeyStore:
    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str = "admin:key:",
    ) -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._key_prefix = key_prefix

    async def create(self, metadata: VirtualKeyMetadata) -> str:
        raw_key = f"vg_{secrets.token_urlsafe(32)}"
        hashed_key = hash_key(raw_key)
        mapping = {
            "hashed_key": hashed_key,
            "created_at": str(int(time.time())),
            "revoked": "false",
        }
        mapping.update(_metadata_mapping(metadata))
        await self._client.hset(self._redis_key_from_hash(hashed_key), mapping=mapping)
        return raw_key

    async def get(self, raw_key: str) -> VirtualKeyRecord | None:
        hashed_key = hash_key(raw_key)
        raw = await self._client.hgetall(self._redis_key_from_hash(hashed_key))
        if not raw:
            return None
        return VirtualKeyRecord(
            hashed_key=hashed_key,
            metadata=_metadata_from_hash(raw),
            revoked=_is_revoked(raw.get("revoked")),
        )

    async def revoke(self, raw_key: str) -> None:
        hashed_key = hash_key(raw_key)
        await self._client.hset(
            self._redis_key_from_hash(hashed_key),
            mapping={
                "hashed_key": hashed_key,
                "revoked": "true",
                "revoked_at": str(int(time.time())),
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _redis_key_from_hash(self, hashed_key: str) -> str:
        return f"{self._key_prefix}{hashed_key}"


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _metadata_mapping(metadata: VirtualKeyMetadata) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if metadata.label is not None:
        mapping["label"] = metadata.label
    if metadata.tier is not None:
        mapping["tier"] = metadata.tier
    if metadata.priority is not None:
        mapping["priority"] = metadata.priority
    if metadata.rate_limit_capacity is not None:
        mapping["rate_limit_capacity"] = str(metadata.rate_limit_capacity)
    if metadata.rate_limit_refill_per_second is not None:
        mapping["rate_limit_refill_per_second"] = str(
            metadata.rate_limit_refill_per_second
        )
    return mapping


def _metadata_from_hash(raw: dict[str, str]) -> VirtualKeyMetadata:
    return VirtualKeyMetadata(
        label=raw.get("label") or None,
        tier=raw.get("tier") or None,
        priority=raw.get("priority") or None,
        rate_limit_capacity=_int_value(raw.get("rate_limit_capacity")),
        rate_limit_refill_per_second=_float_value(
            raw.get("rate_limit_refill_per_second")
        ),
    )


def _is_revoked(value: str | None) -> bool:
    return (value or "").lower() == "true"


def _int_value(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _float_value(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
