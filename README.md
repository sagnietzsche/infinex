## llm-gateway

A middleware service between your application and an LLM provider.

When you have many clients sending requests to an LLM, the naive one-request-per-connection setup has several problems:

1. **Throughput is bad.** GPUs are optimized for batched inference. Sending requests one-by-one wastes GPU efficiency.
2. **No control plane.** Every client holds the provider API key directly. There's no layer for per-user rate limiting, cost tracking, or access control.
3. **No observability.** You can't see error rates, latency distributions, or aggregate usage across all requests.
4. **No resilience.** If the provider goes down, your app dies. There's no place to inject retries, failover, or circuit breakers.
5. **Identical requests get re-computed.** If 10 users ask the same question, the LLM processes it 10 times.

The gateway addresses all of these at the middleware layer.

---

### Features

#### 1. Async Microbatcher

`POST /v1/chat/completions` (non-streaming) queues requests and flushes them as a parallel batch when either `BATCH_MAX_SIZE` is reached or `BATCH_MAX_WAIT_MS` expires. Each caller receives its own response.
The pending batch queue is bounded by `BATCH_QUEUE_MAX_SIZE`; when that queue is
full the gateway returns `503` with `{"detail":"Request queue is full"}`.
Requests can include `"priority": "low" | "normal" | "high"`; the default is
`"normal"`. When there is a backlog, queued requests dispatch by priority first
and enqueue age second, so older high-priority requests run before lower tiers.

```bash
# Non-streaming completion
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello gateway"}]}'

# Inspect batching counters
curl -s http://127.0.0.1:8000/stats
```

#### 2. Dynamic Streaming Batcher

`POST /v1/chat/completions` with `"stream": true` runs through `DynamicBatcher`, which collects concurrent requests into a batch, fires all provider calls in parallel, and fans tokens back to each caller over Server-Sent Events. Client sees incremental output; provider calls are still batched.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"stream this"}],"stream":true}'
```

#### 3. Request Queue

`services/queue.py` provides a bounded queue with backpressure. Two backends:

- **`InMemoryRequestQueue`** — heap-backed in-process queue, raises `QueueFullError` when at capacity.
- **`KafkaRequestQueue`** — publishes requests to a Kafka topic, resolves futures when a matching response arrives on the reply topic. Suitable for distributed, multi-gateway deployments.

#### 4. Client Disconnect Detection

During streaming, the response generator polls `request.is_disconnected()`. When the client drops, `item.cancelled` is set and the provider stream is closed immediately via `aclosing()`, so no tokens are wasted generating a response nobody will read.

#### 5. Redis Response Cache

`services/cache.py` caches completed responses in Redis, keyed by a SHA-256 hash of `(model, messages, temperature, max_tokens)`. Both the streaming and non-streaming paths check the cache before hitting the batcher. Cache hits replay stored chunks as SSE on the streaming path.

The exact cache is checked first. On an exact miss, the gateway can also embed
the user-facing prompt with `text-embedding-3-small` and query a Redis Stack
HNSW vector index for recent near-duplicates. Semantic hits return the cached
response without dispatching to the provider. Responses include `X-Cache` with
`HIT-EXACT`, `HIT-SEMANTIC`, or `MISS`.

```bash
# Run with caching enabled (semantic caching requires Redis Stack + OPENAI_API_KEY)
uv run python main.py

# Disable semantic caching while keeping exact caching enabled
SEMANTIC_CACHE_ENABLED=false uv run python main.py

# Disable caching
CACHE_ENABLED=false uv run python main.py
```

#### 6. API Key Auth + Redis Rate Limiting

When `API_KEYS` is configured, requests must include `X-API-Key` with one of the configured values. The same middleware applies a per-key Redis token bucket using a Lua script, so the read/update operation is atomic inside Redis. Limited requests return `429 Too Many Requests` with `Retry-After`.

```bash
API_KEYS=dev-key RATE_LIMIT_CAPACITY=60 RATE_LIMIT_REFILL_PER_SECOND=1 uv run python main.py

# Premium keys are accepted as normal auth keys and automatically get high priority.
API_KEYS='dev-key,premium-key:tier=premium' uv run python main.py

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"messages":[{"role":"user","content":"hello gateway"}]}'
```

#### 7. Prometheus Metrics + Trace Logs

Prometheus metrics are exposed at `/metrics`. The gateway tracks request volume,
latency histograms, cache hit rate, current queue depth, queue depth per
priority level, and batch fill metrics.

```bash
curl -s http://127.0.0.1:8000/metrics
```

Structured JSON logs include `trace_id`, and batcher events include both
`trace_id` and `batcher_id`. Clients can pass `X-Trace-Id`; otherwise the
gateway generates one and returns it in the `X-Trace-Id` response header.

#### 8. Provider Retries

Provider responses with `429`, `500`, `502`, `503`, or `504` are retried
automatically before the gateway returns a failure. Retries use full-jitter
exponential backoff and are applied to streaming calls only until the first
token is emitted. If retry attempts are exhausted, the provider error response
includes `X-Retries-Attempted`.

#### 9. Provider Failover + Circuit Breakers

`PROVIDER_FALLBACK_CHAIN` configures ordered provider failover. Each provider
also has an independent Redis-backed circuit breaker. The breaker tracks
success/error events in a sliding window and opens when the provider error rate
reaches `CB_ERROR_THRESHOLD`; open providers are skipped until
`CB_COOLDOWN_SECONDS` elapses, then one half-open probe is allowed.

#### 10. Admin Control Plane

Set `ADMIN_API_KEY` to enable protected `/admin` routes for operators. Every
admin request must include `X-Admin-Key`. Admin-created virtual API keys are
stored in Redis by SHA-256 hash under `admin:key:{hashed_key}`; the raw key is
returned only in the create response.

```bash
ADMIN_API_KEY=operator-secret uv run python main.py

curl -s http://127.0.0.1:8000/admin/keys \
  -H 'content-type: application/json' \
  -H 'x-admin-key: operator-secret' \
  -d '{"label":"prod app","tier":"premium","rate_limit_capacity":120}'

curl -s http://127.0.0.1:8000/admin/providers \
  -H 'x-admin-key: operator-secret'
```

---

### Running the service

```bash
uv run python main.py
```

The service starts on `http://0.0.0.0:8000` with hot-reload enabled.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check plus provider circuit states |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/stats` | Batcher counters |
| `POST` | `/v1/chat/completions` | Chat completion (streaming or non-streaming) |
| `POST` | `/admin/keys` | Create a virtual API key |
| `DELETE` | `/admin/keys/{key}` | Revoke a virtual API key |
| `GET` | `/admin/keys/{key}/usage` | Lifetime usage for a key |
| `POST` | `/admin/cache/flush` | Clear cached responses, optionally with `?prefix=` |
| `GET` | `/admin/providers` | Provider activity, circuit state, and recent error rate |

---

### Configuration

All settings are read from environment variables with the defaults shown below.

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_MAX_SIZE` | `16` | Max requests per batch |
| `BATCH_SIZE` | `16` | Alias for `BATCH_MAX_SIZE`; recommended tuning knob |
| `BATCH_MAX_WAIT_MS` | `20` | Max time to wait before flushing a partial batch |
| `MAX_WAIT_MS` | `20` | Alias for `BATCH_MAX_WAIT_MS`; recommended tuning knob |
| `BATCH_QUEUE_MAX_SIZE` | `1024` | Max pending requests accepted before returning `503` |
| `PROVIDER` | `echo` | LLM backend: `openai`, `anthropic`, `gemini`/`google`, `ollama`, or `echo` |
| `PROVIDER_MODE` | `echo` | Backwards-compatible alias for `PROVIDER` |
| `PROVIDER_FALLBACK_CHAIN` | empty | Ordered failover chain, e.g. `openai,anthropic`; first entry becomes the primary provider |
| `OPENAI_API_KEY` | empty | Required when `PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | empty | Required when `PROVIDER=anthropic` |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | empty | Required when `PROVIDER=gemini` or `PROVIDER=google` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL when `PROVIDER=ollama` |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL for cache, rate-limit, and circuit-breaker state |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime in seconds |
| `CACHE_ENABLED` | `true` | Set to `false` to disable response caching |
| `API_KEYS` | empty | Comma-separated allowed API keys. Empty disables auth/rate-limit middleware. Entries can include metadata, e.g. `premium-key:tier=premium` for automatic high priority or `slow-key:priority=low` |
| `ADMIN_API_KEY` | empty | Required `X-Admin-Key` value for protected `/admin` routes and virtual-key management |
| `RATE_LIMIT_CAPACITY` | `60` | Max burst size per API key |
| `RATE_LIMIT_REFILL_PER_SECOND` | `1.0` | Tokens restored per second per API key |
| `MAX_RETRIES` | `3` | Provider retry attempts after the first failed call |
| `RETRY_BASE_DELAY_MS` | `200` | Base delay for provider retry backoff |
| `RETRY_MAX_DELAY_MS` | `5000` | Maximum delay cap for provider retry backoff |
| `CB_ERROR_THRESHOLD` | `0.5` | Provider circuit opens at this error rate; values over `1` are treated as percentages |
| `CB_WINDOW_SECONDS` | `60` | Sliding window used for provider error-rate counters |
| `CB_COOLDOWN_SECONDS` | `30` | Time an open provider circuit stays blocked before one half-open probe |

Example with custom settings:

```bash
BATCH_SIZE=16 MAX_WAIT_MS=20 CACHE_TTL_SECONDS=300 uv run python main.py
```

Example with Anthropic:

```bash
PROVIDER=anthropic ANTHROPIC_API_KEY=... uv run python main.py
```

Example with provider failover:

```bash
PROVIDER_FALLBACK_CHAIN=openai,anthropic \
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
uv run python main.py
```

### Load testing

Use `scripts/load/batching_sweet_spot.js` to validate 50+ concurrent users,
generate p95/p99 reports, and force the queue-full backpressure path:

```bash
CACHE_ENABLED=false BATCH_SIZE=16 MAX_WAIT_MS=20 uv run python main.py
k6 run -e VUS=60 -e DURATION=2m scripts/load/batching_sweet_spot.js
```

Reports are written to `reports/k6-batching-summary.json` and
`reports/k6-batching-summary.md`. See `docs/load-testing.md` for the full
batching sweep and `503` backpressure run.

---

### Project Structure

```
.
├── api/
│   └── routes.py        # FastAPI router: chat completions, health, stats
├── core/
│   ├── config.py        # Settings dataclass, env-var loading
│   └── models.py        # Pydantic request/response models
├── infra/
│   └── providers/       # BaseProvider + LiteLLM-backed OpenAI/Anthropic/Gemini/Ollama providers
├── services/
│   ├── batcher.py       # AsyncRequestBatcher (non-streaming) + DynamicBatcher (streaming)
│   ├── cache.py         # ResponseCache backed by Redis
│   ├── circuit_breaker.py # Redis-backed provider circuit breaker
│   ├── rate_limit.py    # Redis Lua token-bucket rate limiter
│   └── queue.py         # InMemoryRequestQueue + KafkaRequestQueue
├── tests/               # pytest test suite
├── main.py              # App factory + entry point
└── pyproject.toml
```
