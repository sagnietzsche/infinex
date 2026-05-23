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

- **`InMemoryRequestQueue`** — `asyncio.Queue`-backed, raises `QueueFullError` when at capacity.
- **`KafkaRequestQueue`** — publishes requests to a Kafka topic, resolves futures when a matching response arrives on the reply topic. Suitable for distributed, multi-gateway deployments.

#### 4. Client Disconnect Detection

During streaming, the response generator polls `request.is_disconnected()`. When the client drops, `item.cancelled` is set and the provider stream is closed immediately via `aclosing()`, so no tokens are wasted generating a response nobody will read.

#### 5. Redis Response Cache

`services/cache.py` caches completed responses in Redis, keyed by a SHA-256 hash of `(model, messages, temperature, max_tokens)`. Both the streaming and non-streaming paths check the cache before hitting the batcher. Cache hits replay stored chunks as SSE on the streaming path.

```bash
# Run with caching enabled (requires Redis at localhost:6379)
uv run python main.py

# Disable caching
CACHE_ENABLED=false uv run python main.py
```

#### 6. API Key Auth + Redis Rate Limiting

When `API_KEYS` is configured, requests must include `X-API-Key` with one of the configured values. The same middleware applies a per-key Redis token bucket using a Lua script, so the read/update operation is atomic inside Redis. Limited requests return `429 Too Many Requests` with `Retry-After`.

```bash
API_KEYS=dev-key RATE_LIMIT_CAPACITY=60 RATE_LIMIT_REFILL_PER_SECOND=1 uv run python main.py

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"messages":[{"role":"user","content":"hello gateway"}]}'
```

#### 7. Prometheus Metrics + Trace Logs

Prometheus metrics are exposed at `/metrics`. The gateway tracks request volume,
latency histograms, cache hit rate, current queue depth, and batch fill metrics.

```bash
curl -s http://127.0.0.1:8000/metrics
```

Structured JSON logs include `trace_id`, and batcher events include both
`trace_id` and `batcher_id`. Clients can pass `X-Trace-Id`; otherwise the
gateway generates one and returns it in the `X-Trace-Id` response header.

---

### Running the service

```bash
uv run python main.py
```

The service starts on `http://0.0.0.0:8000` with hot-reload enabled.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/stats` | Batcher counters |
| `POST` | `/v1/chat/completions` | Chat completion (streaming or non-streaming) |

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
| `PROVIDER_MODE` | `echo` | LLM backend (`echo` is a deterministic local stub) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL for the response cache |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime in seconds |
| `CACHE_ENABLED` | `true` | Set to `false` to bypass Redis entirely |
| `API_KEYS` | empty | Comma-separated allowed API keys. Empty disables auth/rate-limit middleware |
| `RATE_LIMIT_CAPACITY` | `60` | Max burst size per API key |
| `RATE_LIMIT_REFILL_PER_SECOND` | `1.0` | Tokens restored per second per API key |

Example with custom settings:

```bash
BATCH_SIZE=16 MAX_WAIT_MS=20 CACHE_TTL_SECONDS=300 uv run python main.py
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
│   └── providers.py     # LLMProvider / StreamingLLMProvider protocols + EchoProvider
├── services/
│   ├── batcher.py       # AsyncRequestBatcher (non-streaming) + DynamicBatcher (streaming)
│   ├── cache.py         # ResponseCache backed by Redis
│   ├── rate_limit.py    # Redis Lua token-bucket rate limiter
│   └── queue.py         # InMemoryRequestQueue + KafkaRequestQueue
├── tests/               # pytest test suite
├── main.py              # App factory + entry point
└── pyproject.toml
```
