from __future__ import annotations

import json
import logging
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from core.priority import PriorityLevel


REQUEST_COUNT = Counter(
    "request_count",
    "Number of client completion requests accepted by the gateway.",
    ["endpoint", "stream"],
)

LATENCY_HISTOGRAM = Histogram(
    "latency_histogram",
    "Latency measurements in seconds for gateway, batcher, and queue operations.",
    ["operation"],
)

CACHE_HIT_RATE = Gauge(
    "cache_hit_rate",
    "Ratio of cache hits to cache lookups since process start.",
)

CURRENT_QUEUE_DEPTH = Gauge(
    "current_queue_depth",
    "Current queued item depth by queue or batcher.",
    ["queue"],
)

CURRENT_QUEUE_DEPTH_BY_PRIORITY = Gauge(
    "current_queue_depth_by_priority",
    "Current queued item depth by queue or batcher and priority level.",
    ["queue", "priority"],
)

BATCH_SIZE = Histogram(
    "batch_size",
    "Observed request count per dispatched batch.",
    ["batcher"],
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

BATCH_FILL_RATIO = Gauge(
    "batch_fill_ratio",
    "Most recent dispatched batch size divided by configured max batch size.",
    ["batcher"],
)

_cache_lookups = 0
_cache_hits = 0


def record_request(*, endpoint: str, stream: bool) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, stream=str(stream).lower()).inc()


def record_cache_lookup(*, hit: bool) -> None:
    global _cache_hits, _cache_lookups

    _cache_lookups += 1
    if hit:
        _cache_hits += 1

    CACHE_HIT_RATE.set(_cache_hits / _cache_lookups)


def set_queue_depth(*, queue: str, depth: int) -> None:
    CURRENT_QUEUE_DEPTH.labels(queue=queue).set(depth)


def set_queue_depth_by_priority(
    *, queue: str, priority: PriorityLevel, depth: int
) -> None:
    CURRENT_QUEUE_DEPTH_BY_PRIORITY.labels(
        queue=queue,
        priority=priority,
    ).set(depth)


def observe_latency(*, operation: str, seconds: float) -> None:
    LATENCY_HISTOGRAM.labels(operation=operation).observe(seconds)


def observe_batch(*, batcher: str, size: int, max_size: int) -> None:
    BATCH_SIZE.labels(batcher=batcher).observe(size)
    BATCH_FILL_RATIO.labels(batcher=batcher).set(size / max_size)


PROVIDER_FAILOVER_COUNT = Counter(
    "provider_failover_count",
    "Number of provider failover events.",
    ["from_provider", "to_provider", "reason"],
)


def record_provider_failover(
    *, from_provider: str, to_provider: str, reason: str
) -> None:
    PROVIDER_FAILOVER_COUNT.labels(
        from_provider=from_provider,
        to_provider=to_provider,
        reason=reason,
    ).inc()


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))
