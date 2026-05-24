from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis
import tiktoken

from core.config import MODEL_PRICING_USD_PER_1K_TOKENS
from core.models import ChatCompletionRequest, CompletionUsage
from services.observability import log_event


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageTotals:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class TokenCounter:
    def __init__(self) -> None:
        self._encodings: dict[str, Any] = {}

    async def count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        return await asyncio.to_thread(self.count_prompt_tokens_sync, request)

    async def count_text_tokens(self, *, model: str, text: str) -> int:
        return await asyncio.to_thread(
            self.count_text_tokens_sync,
            model=model,
            text=text,
        )

    def count_prompt_tokens_sync(self, request: ChatCompletionRequest) -> int:
        return sum(
            self.count_text_tokens_sync(
                model=request.model,
                text=f"{message.role}\n{message.content}",
            )
            for message in request.messages
        )

    def count_text_tokens_sync(self, *, model: str, text: str) -> int:
        normalized_model = _normalize_model_name(model)
        try:
            encoding = self._encodings[normalized_model]
        except KeyError:
            try:
                encoding = tiktoken.encoding_for_model(normalized_model)
            except KeyError:
                return _character_heuristic_tokens(text)
            self._encodings[normalized_model] = encoding
        return len(encoding.encode(text))


class UsageTracker:
    def __init__(
        self,
        *,
        redis_url: str,
        pricing_table: dict[str, dict[str, float]] | None = None,
        key_prefix: str = "llm-gateway:usage:v1:",
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._pricing_table = pricing_table or MODEL_PRICING_USD_PER_1K_TOKENS
        self._key_prefix = key_prefix
        self._token_counter = token_counter or TokenCounter()

    async def count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        return await self._token_counter.count_prompt_tokens(request)

    async def record_request(
        self,
        *,
        api_key: str | None,
        model: str,
        prompt_tokens: int | asyncio.Task[int],
        completion_text: str,
        completion_usage: CompletionUsage | None,
        log_fields: dict[str, Any],
    ) -> UsageTotals:
        prompt_token_count = await _resolve_prompt_tokens(prompt_tokens)
        completion_tokens = await self._completion_tokens(
            model=model,
            completion_text=completion_text,
            completion_usage=completion_usage,
        )
        usage = self.calculate_usage(
            model=model,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_tokens,
        )

        log_event(
            logger,
            "request.completed",
            **log_fields,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
        )

        if api_key is not None:
            await self._store_usage(api_key=api_key, usage=usage)

        return usage

    async def get_usage(self, api_key: str) -> UsageTotals:
        raw = await self._client.hgetall(self._redis_key(api_key))
        return UsageTotals(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
            estimated_cost_usd=float(raw.get("estimated_cost_usd", 0.0) or 0.0),
        )

    def calculate_usage(
        self, *, model: str, prompt_tokens: int, completion_tokens: int
    ) -> UsageTotals:
        pricing = _pricing_for_model(self._pricing_table, model)
        estimated_cost_usd = (
            (prompt_tokens / 1000) * pricing.get("input", 0.0)
            + (completion_tokens / 1000) * pricing.get("output", 0.0)
        )
        return UsageTotals(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 8),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _completion_tokens(
        self,
        *,
        model: str,
        completion_text: str,
        completion_usage: CompletionUsage | None,
    ) -> int:
        if completion_usage is not None and completion_usage.completion_tokens > 0:
            return completion_usage.completion_tokens
        return await self._token_counter.count_text_tokens(
            model=model,
            text=completion_text,
        )

    async def _store_usage(self, *, api_key: str, usage: UsageTotals) -> None:
        key = self._redis_key(api_key)
        try:
            await self._client.hincrby(key, "prompt_tokens", usage.prompt_tokens)
            await self._client.hincrby(
                key, "completion_tokens", usage.completion_tokens
            )
            await self._client.hincrby(key, "total_tokens", usage.total_tokens)
            await self._client.hincrbyfloat(
                key, "estimated_cost_usd", usage.estimated_cost_usd
            )
        except Exception as exc:
            log_event(
                logger,
                "usage.store_failed",
                level=logging.WARNING,
                error=repr(exc),
            )

    def _redis_key(self, api_key: str) -> str:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return self._key_prefix + digest


async def _resolve_prompt_tokens(prompt_tokens: int | asyncio.Task[int]) -> int:
    if isinstance(prompt_tokens, int):
        return prompt_tokens
    try:
        return await prompt_tokens
    except Exception as exc:
        log_event(
            logger,
            "usage.prompt_count_failed",
            level=logging.WARNING,
            error=repr(exc),
        )
        return 0


def _pricing_for_model(
    pricing_table: dict[str, dict[str, float]], model: str
) -> dict[str, float]:
    normalized_model = _normalize_model_name(model)
    return pricing_table.get(
        model,
        pricing_table.get(normalized_model, {"input": 0.0, "output": 0.0}),
    )


def _normalize_model_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _character_heuristic_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
