---
name: project-overview
description: Architecture and feature status of the llm-gateway project
metadata:
  type: project
---

FastAPI LLM gateway (Python 3.13, uv, pytest). Core layers:
- `infra/providers.py` — LLMProvider / StreamingLLMProvider protocols; EchoProvider for dev
- `services/batcher.py` — AsyncRequestBatcher (non-streaming) + DynamicBatcher (streaming)
- `services/cache.py` — ResponseCache backed by Redis asyncio; make_cache_key hashes model/messages/temperature/max_tokens with SHA-256
- `api/routes.py` — cache checked before batcher; hits bypass batcher entirely; streaming hits replay cached chunks
- `core/config.py` — Settings dataclass with REDIS_URL, CACHE_TTL_SECONDS, CACHE_ENABLED env vars
- `main.py` — wires cache into lifespan (creates + closes ResponseCache)

**Why:** Implemented per acceptance criteria: redis-py async, SHA-256 keys, TTL-configurable storage, cache hits bypass batcher.

**How to apply:** Pre-existing test failures in test_request_queue.py::RouteTests (2 tests) call create_app with a queue_maxsize kwarg that was never supported — not regressions.
