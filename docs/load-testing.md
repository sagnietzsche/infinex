# Load testing and batching tune-up

The k6 script in `scripts/load/batching_sweet_spot.js` validates the gateway
under high-concurrency traffic and writes p95/p99 reports to `reports/`.

## Recommended baseline

The current gateway defaults are tuned for the local echo provider and short
HTTP calls:

```bash
BATCH_SIZE=16
MAX_WAIT_MS=20
BATCH_QUEUE_MAX_SIZE=1024
```

`BATCH_SIZE` and `MAX_WAIT_MS` are aliases for the older
`BATCH_MAX_SIZE` and `BATCH_MAX_WAIT_MS` names. For this codebase, `16/20ms`
keeps the batching window short while allowing 50+ concurrent clients to fill
batches consistently. Increase `BATCH_SIZE` only if the provider benefits from
larger GPU batches and p95 latency remains inside the target.

## Steady high-concurrency run

Start the gateway with cache disabled so repeated prompts do not bypass the
batcher:

```bash
CACHE_ENABLED=false BATCH_SIZE=16 MAX_WAIT_MS=20 uv run python main.py
```

Run k6 with at least 50 VUs:

```bash
k6 run \
  -e BASE_URL=http://127.0.0.1:8000 \
  -e VUS=60 \
  -e DURATION=2m \
  -e BATCH_SIZE=16 \
  -e MAX_WAIT_MS=20 \
  scripts/load/batching_sweet_spot.js
```

The script sends traffic across multiple model names and generates:

- `reports/k6-batching-summary.json`
- `reports/k6-batching-summary.md`

Both reports include `http_req_duration` p95 and p99 latency. The default
thresholds are p95 under 250ms, p99 under 500ms, and less than 1% failed
requests for the steady scenario.

## Batch parameter sweep

Run the same test against a small matrix and compare p95, p99, request rate,
and batch fill metrics from `/metrics`. Start the gateway once per pair, then
run k6 against that process:

```bash
CACHE_ENABLED=false BATCH_SIZE=8 MAX_WAIT_MS=10 uv run python main.py
k6 run -e BATCH_SIZE=8 -e MAX_WAIT_MS=10 scripts/load/batching_sweet_spot.js
```

Repeat for `BATCH_SIZE=8,16,32` and `MAX_WAIT_MS=10,20,40`.

Pick the smallest `MAX_WAIT_MS` that still gives healthy batch fill ratios, then
increase `BATCH_SIZE` until p95 or p99 starts to regress. For the local echo
provider, `BATCH_SIZE=16` and `MAX_WAIT_MS=20` are the baseline sweet spot.

## Queue-full backpressure run

To force the 503 path, make the pending batch queue smaller than the batch size
and use a long wait window:

```bash
CACHE_ENABLED=false \
BATCH_QUEUE_MAX_SIZE=8 \
BATCH_SIZE=64 \
MAX_WAIT_MS=1000 \
uv run python main.py
```

Then run the backpressure scenario:

```bash
k6 run \
  -e BASE_URL=http://127.0.0.1:8000 \
  -e VUS=60 \
  -e DURATION=30s \
  -e RUN_BACKPRESSURE=true \
  -e BACKPRESSURE_VUS=200 \
  -e BACKPRESSURE_DURATION=30s \
  scripts/load/batching_sweet_spot.js
```

The `queue_full_503` threshold requires at least one explicit 503 response with
`{"detail":"Request queue is full"}`.
