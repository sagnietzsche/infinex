---
name: project-overview
description: Architecture and feature status of the llm-gateway project
metadata:
  type: project
---

FastAPI LLM gateway (Python 3.13, uv, pytest). Core layers:
- `infra/providers/` — BaseProvider plus EchoProvider for dev and LiteLLM-backed OpenAI/Anthropic/Gemini/Ollama providers
- `services/batcher.py` — AsyncRequestBatcher (non-streaming) + DynamicBatcher (streaming)
- `services/cache.py` — ResponseCache backed by Redis asyncio; make_cache_key hashes model/messages/temperature/max_tokens with SHA-256
- `api/routes.py` — cache checked before batcher; hits bypass batcher entirely; streaming hits replay cached chunks
- `core/config.py` — Settings dataclass with PROVIDER, provider API keys, OLLAMA_BASE_URL, REDIS_URL, CACHE_TTL_SECONDS, CACHE_ENABLED env vars
- `main.py` — wires cache into lifespan (creates + closes ResponseCache)

**Current status:** Provider selection is controlled by `PROVIDER`, with `PROVIDER_MODE` retained as a backwards-compatible alias. Non-streaming and streaming requests continue through the existing batchers.
