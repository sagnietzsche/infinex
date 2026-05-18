## llm-gateway

building an llm gateway

When you have lots of clients sending requests to an LLM (like GPT-4), each request takes time. The LLM provider charges per request, and there are rate limits on how many you can send per minute. If you have an app with 100 users all chatting at once, you have 100 separate requests in flight, each one taking a few seconds.

There are some problems with this naive setup:

1. **Throughput is bad.** GPUs are optimized to process many requests at once (batching). When requests go one-by-one to the provider, you waste GPU efficiency. The provider may batch internally, but if you're hitting their API yourself, you're not getting that benefit at *your* layer.
2. **No control plane.** Every client knows the provider's API key. There's no way to track who used how much, rate-limit specific users, or cache responses.
3. **No observability.** You can't see what's happening across all your requests — error rates, latency distributions, etc.
4. **Cold redundancy.** If the provider goes down, your app dies. There's no place to inject retries, failover, or circuit breakers.
5. **Identical requests get re-computed.** If 10 users ask the same question, the LLM processes it 10 times.



The gateway adds the features the naive setup lacks:

1. **Batching** — collect concurrent requests, send them as parallel/batched calls to the provider, distribute responses back
2. **Caching** — if the same prompt has been asked recently, return the cached response without calling the provider
3. **Rate limiting** — per-API-key limits to prevent abuse and stay within provider quotas
4. **Auth** — API key validation
5. **Observability** — log every request, expose metrics
6. **Streaming** — even with batching/caching, responses still stream token-by-token to clients (so they see incremental output, not a 5-second wait then a wall of text)

It's a middleware service between your app and a provider.

### Kafka Queue

The request handoff uses Kafka by default. HTTP handlers publish requests to
`KAFKA_REQUEST_TOPIC` and wait on an in-process Future keyed by `request_id`.
A worker consumes requests, executes the LLM call, publishes the result to
`KAFKA_RESPONSE_TOPIC`, and the gateway response consumer resolves the waiting
Future.

Configuration:

```
REQUEST_QUEUE_BACKEND=kafka
REQUEST_QUEUE_MAXSIZE=100
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_REQUEST_TOPIC=llm-gateway.requests
KAFKA_RESPONSE_TOPIC=llm-gateway.responses
KAFKA_CONSUMER_GROUP=llm-gateway-workers
GATEWAY_INSTANCE_ID=<optional stable instance id>
```

Use `REQUEST_QUEUE_BACKEND=memory` for local tests without Kafka.

### Project Structure

```
.
├── api/             # Transport: FastAPI routes, middleware, dependencies
├── services/        # Business Logic: Batcher, Queue management, Rate limiting
├── infra/           # Infrastructure: Redis, OpenAI client, Database
├── core/            # Cross-cutting: Config, Logging, Shared Models
├── main.py          # Entry point
└── uv.lock          # Package management
```
