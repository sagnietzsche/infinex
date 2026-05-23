from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass
import logging
from typing import Any

from core.models import ChatCompletionRequest, ChatCompletionResponse
from infra.providers import LLMProvider, StreamingLLMProvider
from services.circuit_breaker import CircuitBreaker, CircuitOpenError
from services.observability import log_event, record_provider_failover
from services.retry import (
    RetryExhaustedError,
    TRANSIENT_STATUS_CODES,
    provider_status_code,
)


logger = logging.getLogger(__name__)
PROVIDER_USED_ATTR = "_provider_used"


@dataclass(frozen=True)
class ProviderRoute:
    name: str
    provider: LLMProvider
    streaming_provider: StreamingLLMProvider | None = None


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    error: str
    status_code: int | None
    retries_attempted: int | None = None


class AllProvidersFailedError(Exception):
    def __init__(self, failures: list[ProviderFailure]) -> None:
        super().__init__("All providers in the fallback chain failed")
        self.failures = failures


@dataclass(frozen=True)
class ProviderStreamMetadata:
    provider: str
    model: str


class ProviderRouter:
    def __init__(
        self,
        *,
        routes: list[ProviderRoute],
        model_mapping: Mapping[str, str],
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        if not routes:
            raise ValueError("at least one provider route is required")
        self._routes = routes
        self._model_mapping = model_mapping
        self._primary_provider = routes[0].name
        self._circuit_breaker = circuit_breaker

    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        failures: list[ProviderFailure] = []

        for index, route in enumerate(self._routes):
            if not await self._route_allowed(
                route=route, index=index, failures=failures
            ):
                continue

            routed_request = self._request_for_route(request, route.name)
            try:
                response = await route.provider.complete(routed_request)
            except Exception as exc:
                await self._record_provider_failure(route.name)
                if not _should_failover(exc):
                    raise
                failures.append(_failure_for(route.name, exc))
                self._record_failover(
                    route=route,
                    exc=exc,
                    next_route=self._next_route(index),
                )
                continue

            await self._record_provider_success(route.name)
            _set_provider_used(response, route.name)
            return response

        raise AllProvidersFailedError(failures)

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str | ProviderStreamMetadata]:
        failures: list[ProviderFailure] = []

        for index, route in enumerate(self._routes):
            if not await self._route_allowed(
                route=route, index=index, failures=failures
            ):
                continue

            provider = route.streaming_provider or route.provider
            routed_request = self._request_for_route(request, route.name)
            yielded = False
            metadata_sent = False
            try:
                async with aclosing(provider.stream(routed_request)) as stream:
                    async for token in stream:
                        yielded = True
                        if not metadata_sent:
                            metadata_sent = True
                            yield ProviderStreamMetadata(
                                provider=route.name,
                                model=routed_request.model,
                            )
                        yield token
                if not metadata_sent:
                    yield ProviderStreamMetadata(
                        provider=route.name,
                        model=routed_request.model,
                    )
                await self._record_provider_success(route.name)
                return
            except Exception as exc:
                await self._record_provider_failure(route.name)
                if yielded or not _should_failover(exc):
                    raise
                failures.append(_failure_for(route.name, exc))
                self._record_failover(
                    route=route,
                    exc=exc,
                    next_route=self._next_route(index),
                )
                continue

        raise AllProvidersFailedError(failures)

    def _request_for_route(
        self, request: ChatCompletionRequest, provider_name: str
    ) -> ChatCompletionRequest:
        model = self._mapped_model(provider_name=provider_name, model=request.model)
        if model == request.model:
            return request
        return request.model_copy(update={"model": model})

    def _mapped_model(self, *, provider_name: str, model: str) -> str:
        if provider_name == self._primary_provider:
            return model

        mapped = self._model_mapping.get(f"{self._primary_provider}/{model}")
        if mapped is None:
            return model

        mapped_provider, separator, mapped_model = mapped.partition("/")
        if separator and mapped_provider == provider_name and mapped_model:
            return mapped_model
        return model

    def _next_route(self, index: int) -> ProviderRoute | None:
        next_index = index + 1
        if next_index >= len(self._routes):
            return None
        return self._routes[next_index]

    async def _route_allowed(
        self,
        *,
        route: ProviderRoute,
        index: int,
        failures: list[ProviderFailure],
    ) -> bool:
        if self._circuit_breaker is None:
            return True

        decision = await self._circuit_breaker.before_request(route.name)
        if decision.allowed:
            return True

        exc = CircuitOpenError(route.name, decision.state)
        failures.append(_failure_for(route.name, exc))
        self._record_failover(
            route=route,
            exc=exc,
            next_route=self._next_route(index),
        )
        return False

    async def _record_provider_success(self, provider_name: str) -> None:
        if self._circuit_breaker is not None:
            await self._circuit_breaker.record_success(provider_name)

    async def _record_provider_failure(self, provider_name: str) -> None:
        if self._circuit_breaker is not None:
            await self._circuit_breaker.record_failure(provider_name)

    def _record_failover(
        self,
        *,
        route: ProviderRoute,
        exc: BaseException,
        next_route: ProviderRoute | None,
    ) -> None:
        if next_route is None:
            return

        reason = _failure_reason(exc)
        record_provider_failover(
            from_provider=route.name,
            to_provider=next_route.name,
            reason=reason,
        )
        log_event(
            logger,
            "provider.failover",
            level=logging.ERROR,
            from_provider=route.name,
            to_provider=next_route.name,
            reason=reason,
            status_code=provider_status_code(_original_exception(exc)),
            error=repr(_original_exception(exc)),
        )


def provider_used(response: ChatCompletionResponse) -> str | None:
    value = getattr(response, PROVIDER_USED_ATTR, None)
    return value if isinstance(value, str) else None


def _set_provider_used(
    response: ChatCompletionResponse, provider_name: str
) -> None:
    setattr(response, PROVIDER_USED_ATTR, provider_name)


def _should_failover(exc: BaseException) -> bool:
    if isinstance(exc, RetryExhaustedError):
        return True
    if _is_circuit_open(exc):
        return True

    status_code = provider_status_code(exc)
    if status_code is None:
        return True
    return status_code in TRANSIENT_STATUS_CODES or status_code >= 500


def _is_circuit_open(exc: BaseException) -> bool:
    if getattr(exc, "circuit_open", False):
        return True
    name = exc.__class__.__name__.lower()
    return "circuit" in name and "open" in name


def _failure_for(provider_name: str, exc: BaseException) -> ProviderFailure:
    original = _original_exception(exc)
    retries_attempted = (
        exc.retries_attempted if isinstance(exc, RetryExhaustedError) else None
    )
    return ProviderFailure(
        provider=provider_name,
        error=repr(original),
        status_code=provider_status_code(original),
        retries_attempted=retries_attempted,
    )


def _original_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, RetryExhaustedError):
        return exc.original
    return exc


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, RetryExhaustedError):
        return "retries_exhausted"
    if _is_circuit_open(exc):
        return "circuit_open"
    status_code = provider_status_code(exc)
    if status_code is not None:
        return f"status_{status_code}"
    return "provider_error"


def all_providers_error_body(exc: AllProvidersFailedError) -> dict[str, Any]:
    return {
        "error": {
            "code": "providers_unavailable",
            "message": "All providers in the fallback chain failed",
            "failures": [
                {
                    "provider": failure.provider,
                    "error": failure.error,
                    "status_code": failure.status_code,
                    "retries_attempted": failure.retries_attempted,
                }
                for failure in exc.failures
            ],
        }
    }
