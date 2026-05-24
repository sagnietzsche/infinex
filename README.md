# llm-gateway

[![CI and Deploy](https://github.com/sagnikc395/llm-gateway/actions/workflows/ci-deploy.yml/badge.svg)](https://github.com/sagnikc395/llm-gateway/actions/workflows/ci-deploy.yml)

`llm-gateway` is a FastAPI service that sits between your application and one or
more LLM providers. It exposes an OpenAI-style chat completions endpoint and
adds the operational pieces that are easy to forget when every service calls an
LLM provider directly: request scheduling, backpressure, caching, API key
control, rate limits, usage accounting, metrics, retries, failover, and an admin
API for virtual keys.

The default provider is `echo`, so the project can run locally without a paid
LLM API key. Real provider calls go through LiteLLM and currently support
OpenAI, Anthropic, Gemini or Google, and Ollama.

## What this is for

Use this gateway when you want one controlled place for LLM traffic instead of
giving every client direct access to provider credentials.

It is useful for:

- Hiding provider API keys from client apps.
- Applying per-key rate limits and priority levels.
- Smoothing traffic bursts with bounded queues.
- Avoiding repeated work with exact and semantic response caching.
- Tracking usage, token counts, and estimated cost per API key.
- Watching request volume, latency, cache hits, queue depth, and failover events.
- Retrying transient provider errors and falling back to another provider.
- Managing virtual API keys from an operator-only admin API.

One important detail: the batchers collect requests over a short scheduling
window and then dispatch provider calls concurrently. They do not currently send
a single provider-native batch inference request. That still gives the gateway a
central queue, priority ordering, backpressure, and useful batch metrics. A true
provider batch implementation could be added behind the provider interface.

## Quick start

Requirements:

- Python 3.13
- `uv`
- Redis for the default cache, readiness checks, circuit breaker state, usage
  storage, and virtual keys

Install dependencies:

```bash
uv sync --all-groups
```

For the lightest local demo, disable the cache so a missing Redis instance does
not break the first chat request:

```bash
CACHE_ENABLED=false uv run python main.py
```

The service starts at `http://0.0.0.0:8000` with reload enabled.

Try the default echo provider:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello gateway"}]}'
```

For the full default setup, run Redis and leave caching enabled:

```bash
redis-server
uv run python main.py
```

If Redis is not reachable, `/health` can still return `ok`, but `/ready` will
return `503` because readiness checks Redis and provider circuit state.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check with process uptime |
| `GET` | `/ready` | Readiness check for Redis, queue capacity, provider circuits, and shutdown drain |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/stats` | Non-streaming batcher counters |
| `POST` | `/v1/chat/completions` | Chat completion, streaming or non-streaming |
| `POST` | `/admin/keys` | Create a virtual API key |
| `DELETE` | `/admin/keys/{key}` | Revoke a virtual API key |
| `GET` | `/admin/keys/{key}/usage` | Lifetime usage totals for a key |
| `POST` | `/admin/cache/flush` | Clear cached responses, optionally with `?prefix=` |
| `GET` | `/admin/providers` | Provider circuit state and recent error rate |

## Chat requests

Non-streaming request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gateway-echo",
    "messages": [{"role": "user", "content": "explain the gateway"}]
  }'
```

Streaming request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "stream this back"}],
    "stream": true
  }'
```

Requests can include `priority` as `low`, `normal`, or `high`. When there is a
backlog, higher priority requests are dispatched first. Requests with the same
priority are ordered by enqueue time.

```json
{
  "messages": [{"role": "user", "content": "run soon"}],
  "priority": "high"
}
```

Clients can also pass `X-Trace-Id`. If they do not, the gateway creates one and
returns it in the `X-Trace-Id` response header.

## Providers

The provider is selected with `PROVIDER`. The default is `echo`.

```bash
PROVIDER=openai OPENAI_API_KEY=... uv run python main.py
PROVIDER=anthropic ANTHROPIC_API_KEY=... uv run python main.py
PROVIDER=gemini GOOGLE_API_KEY=... uv run python main.py
PROVIDER=ollama OLLAMA_BASE_URL=http://localhost:11434 uv run python main.py
```

Provider failover is configured with `PROVIDER_FALLBACK_CHAIN`:

```bash
PROVIDER_FALLBACK_CHAIN=openai,anthropic \
OPENAI_API_KEY=... \
ANTHROPIC_API_KEY=... \
uv run python main.py
```

Each provider in the chain gets its own Redis-backed circuit breaker. The router
skips open circuits, allows one half-open probe after the cooldown, and records
failover metrics. Model mapping for fallback providers currently lives in
`core/config.py`.

## Caching

When `CACHE_ENABLED=true`, completed responses are stored in Redis. The exact
cache key is built from:

- `model`
- `messages`
- `temperature`
- `max_tokens`

Responses include an `X-Cache` header with one of:

- `HIT-EXACT`
- `HIT-SEMANTIC`
- `MISS`

Semantic caching is enabled by default when caching is enabled. It embeds the
user-facing prompt, stores vectors in a Redis Stack HNSW index, and can reuse a
recent near-duplicate response when the model and generation settings match.

Semantic caching needs Redis Stack and an OpenAI-compatible embedding key for
the default `text-embedding-3-small` model.

```bash
SEMANTIC_CACHE_ENABLED=false uv run python main.py
CACHE_ENABLED=false uv run python main.py
```

## Auth, rate limits, and virtual keys

Regular API key checks are enabled when either `API_KEYS` or `ADMIN_API_KEY` is
set.

Static keys come from `API_KEYS`:

```bash
API_KEYS=dev-key RATE_LIMIT_CAPACITY=60 RATE_LIMIT_REFILL_PER_SECOND=1 uv run python main.py
```

Then call the gateway with `X-API-Key`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Static keys can include metadata for priority:

```bash
API_KEYS='dev-key,premium-key:tier=premium,slow-key:priority=low' uv run python main.py
```

`tier=premium` maps to high priority.

Set `ADMIN_API_KEY` to use the admin routes. Admin-created virtual keys are
stored in Redis by SHA-256 hash. The raw key is returned only when it is created.

```bash
ADMIN_API_KEY=operator-secret uv run python main.py

curl -s http://127.0.0.1:8000/admin/keys \
  -H 'content-type: application/json' \
  -H 'x-admin-key: operator-secret' \
  -d '{"label":"prod app","tier":"premium","rate_limit_capacity":120}'
```

If only `ADMIN_API_KEY` is set, regular chat requests must use a valid virtual
key created through the admin API.

## Observability

Prometheus metrics are exposed at `/metrics`. The gateway records:

- Accepted request count by endpoint and streaming mode.
- Latency histograms for HTTP, queue, batcher, and provider work.
- Cache hit rate.
- Current queue depth and queue depth by priority.
- Batch size and batch fill ratio.
- Provider failover count.

Application logs are structured JSON. Request logs include the trace ID where
one is available, and batcher logs include both trace IDs and batch IDs.

Usage accounting runs in the background after a request completes. Prompt tokens
are counted with `tiktoken` when possible, with a character-count fallback for
unknown models. Completion usage is taken from the provider response when the
provider returns it, otherwise the gateway counts completion text locally.

## Reliability behavior

The gateway retries provider errors with status `429`, `500`, `502`, `503`, and
`504`. Retries use full-jitter exponential backoff. Streaming calls are retried
only until the first token has been emitted.

When all retries fail, the response includes `X-Retries-Attempted`.

Provider failover happens for retry exhaustion, open circuits, and transient or
server-side provider errors. If every provider in the fallback chain fails, the
gateway returns `503` with the failure details.

During shutdown, the app stops accepting new non-probe traffic and waits for
in-flight requests and active streams up to `SHUTDOWN_DRAIN_TIMEOUT_SECONDS`.

During streaming, the server checks whether the client disconnected. If the
client drops, the stream is marked cancelled and the provider stream is closed
so the gateway stops spending tokens on a response that no one will read.

## Request queue module

`services/queue.py` contains reusable queue implementations:

- `InMemoryRequestQueue` for an in-process priority queue.
- `KafkaRequestQueue`, `KafkaQueueConsumer`, and `KafkaResponseConsumer` for a
  distributed request and response workflow.

The current FastAPI app uses the batchers in `services/batcher.py` directly.
The Kafka queue is available as a service module and is covered by tests, but it
is not currently selected by an environment variable in `create_app`.

## Configuration

All settings are read from environment variables.
Use [`.env.sample`](.env.sample) as the fill-in checklist for local shell
variables, Railway service variables, and GitHub deployment secrets.

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `llm-gateway` | FastAPI app name |
| `BATCH_MAX_SIZE` | `16` | Maximum requests collected per scheduling window |
| `BATCH_SIZE` | `16` | Alias for `BATCH_MAX_SIZE` |
| `BATCH_MAX_WAIT_MS` | `20` | Maximum wait before dispatching a partial batch |
| `MAX_WAIT_MS` | `20` | Alias for `BATCH_MAX_WAIT_MS` |
| `BATCH_QUEUE_MAX_SIZE` | `1024` | Maximum pending requests before `503` |
| `QUEUE_MAX_DEPTH` | `1024` | Alias for `BATCH_QUEUE_MAX_SIZE` |
| `PROVIDER` | `echo` | `echo`, `openai`, `anthropic`, `gemini`, `google`, or `ollama` |
| `PROVIDER_MODE` | `echo` | Backward-compatible alias for `PROVIDER` |
| `PROVIDER_FALLBACK_CHAIN` | empty | Ordered provider chain, such as `openai,anthropic` |
| `OPENAI_API_KEY` | empty | Required for `PROVIDER=openai`; also used by default semantic embeddings |
| `ANTHROPIC_API_KEY` | empty | Required for `PROVIDER=anthropic` |
| `GOOGLE_API_KEY` | empty | Required for `PROVIDER=gemini` or `PROVIDER=google` |
| `GEMINI_API_KEY` | empty | Alias for `GOOGLE_API_KEY` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `CACHE_ENABLED` | `true` | Enables Redis response caching |
| `CACHE_TTL_SECONDS` | `3600` | Exact cache TTL |
| `SEMANTIC_CACHE_ENABLED` | `true` | Enables Redis Stack vector cache lookups |
| `SEMANTIC_CACHE_TTL_SECONDS` | `3600` | Semantic cache TTL |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Minimum semantic similarity. Values over `1` are treated as percentages |
| `SEMANTIC_CACHE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for semantic cache |
| `SEMANTIC_CACHE_EMBEDDING_DIMENSION` | `1536` | Embedding dimension for Redis vector index |
| `API_KEYS` | empty | Comma-separated static API keys, optionally with metadata |
| `ADMIN_API_KEY` | empty | Required value for `X-Admin-Key` on `/admin` routes |
| `RATE_LIMIT_CAPACITY` | `60` | Token bucket burst size per key |
| `RATE_LIMIT_REFILL_PER_SECOND` | `1.0` | Token bucket refill rate per key |
| `MAX_RETRIES` | `3` | Retry attempts after the initial provider call fails |
| `RETRY_BASE_DELAY_MS` | `200` | Base retry backoff delay |
| `RETRY_MAX_DELAY_MS` | `5000` | Retry backoff cap |
| `CB_ERROR_THRESHOLD` | `0.5` | Circuit breaker error-rate threshold |
| `CB_WINDOW_SECONDS` | `60` | Circuit breaker sliding window |
| `CB_COOLDOWN_SECONDS` | `30` | Time before an open circuit allows a half-open probe |
| `SHUTDOWN_DRAIN_TIMEOUT_SECONDS` | `30.0` | Grace period for in-flight work during shutdown |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Load testing

The k6 script in `scripts/load/batching_sweet_spot.js` exercises concurrent
traffic, reports p95 and p99 latency, and can force the queue-full `503` path.

```bash
CACHE_ENABLED=false BATCH_SIZE=16 MAX_WAIT_MS=20 uv run python main.py
k6 run -e VUS=60 -e DURATION=2m scripts/load/batching_sweet_spot.js
```

Reports are written to:

- `reports/k6-batching-summary.json`
- `reports/k6-batching-summary.md`

See `docs/load-testing.md` for the full sweep and backpressure run.

## Tests and quality checks

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

GitHub Actions runs the same checks on pull requests and pushes to `main`.
Pushes to `main` also deploy to Railway through
[`.github/workflows/ci-deploy.yml`](.github/workflows/ci-deploy.yml) when the
Railway configuration is present.

## Deployment

Useful deployment links:

- [CI and deployment workflow](.github/workflows/ci-deploy.yml)
- [GitHub Actions run history](https://github.com/sagnikc395/llm-gateway/actions/workflows/ci-deploy.yml)
- [Railway deployment config](railway.json)
- [Production Dockerfile](Dockerfile)
- [Environment variable sample](.env.sample)

The [Dockerfile](Dockerfile) builds a Python 3.13 runtime image and starts:

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

[`railway.json`](railway.json) uses the Dockerfile builder and `/health` as the
deployment health check.

Required GitHub secret for Railway deploys:

```bash
RAILWAY_TOKEN=...
```

Optional GitHub variables:

```bash
RAILWAY_SERVICE=llm-gateway
RAILWAY_ENVIRONMENT=production
```

Runtime secrets should live in Railway variables, not in the repository:

```bash
railway variable set "REDIS_URL=redis://..." --service llm-gateway
railway variable set "OPENAI_API_KEY=..." --service llm-gateway
railway variable set "API_KEYS=..." --service llm-gateway
railway variable set "ADMIN_API_KEY=..." --service llm-gateway
```

## Project layout

```text
api/                 FastAPI routes for chat, admin, and health checks
core/                Settings, request models, priorities, and pricing data
infra/providers/     Echo provider and LiteLLM-backed provider adapters
services/            Batching, caching, retries, rate limits, usage, queues, and shutdown
docs/                Supporting project docs
scripts/load/        k6 load-testing scripts
tests/               pytest suite
main.py              App factory and local entry point
Dockerfile           Production container image
railway.json         Railway deployment settings
pyproject.toml       Python dependencies and tool config
```
