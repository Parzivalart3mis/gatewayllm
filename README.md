# GatewayLLM

A self-hosted, OpenAI-compatible LLM gateway in Go. It fronts multiple providers
with policy-based routing, automatic failover, and a two-tier semantic cache —
shipped as **one stateless binary**, not a constellation of microservices.

Point any existing OpenAI client at it and change nothing but the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="glm_live_...")

client.chat.completions.create(
    model="fast",                                    # a gateway alias, not a vendor model
    messages=[{"role": "user", "content": "Hello"}],
)
```

`fast` resolves to Groq, falls back to Gemini, then OpenAI. The caller never
learns which one answered.

---

## Why it's built this way

**One binary, three stores.** All state lives in Redis, Qdrant, and Postgres, so
scaling is `--scale gateway=3` behind a load balancer — no leader election, no
sticky sessions, no service mesh. The only thing outside the binary is the
embedding model, because the mature ones ship as Python and reimplementing
inference in Go to preserve purity would be a bad trade.

**Two interfaces carry the design.** `Provider` and `Embedder` are the load-bearing
abstractions. Adding a backend is one new file:

```go
type Provider interface {
    Name() string
    Chat(ctx, *ChatRequest) (*ChatResponse, error)
    ChatStream(ctx, *ChatRequest, func(*ChatChunk) error) error
    Embed(ctx, *EmbeddingRequest) (*EmbeddingResponse, error)
    Models() []string
}
```

OpenAI and Groq share one implementation (identical wire protocol). Gemini has
its own — different roles, different envelope, different parameter names — which
is what proves the interface actually holds.

**No overlap between stores.** Each fact lives in exactly one place:

| Store | Owns | Why there |
|---|---|---|
| Redis | exact-match cache, rate-limit buckets, breaker state | shared across replicas; sub-ms reads |
| Qdrant | semantic cache (vector + completion + params) | vector similarity search |
| Postgres | tenants, hashed keys, pricing, usage ledger | durable, queryable after the fact |

---

## The two-tier cache

This is the hard part, and the interesting one.

**Tier 1 — exact (Redis).** The normalized request (alias + messages +
temperature + max_tokens) is hashed into a key. An identical repeat returns in
one round trip with no embedding call and no provider call.

**Tier 2 — semantic (Qdrant).** Only on an exact miss: embed the prompt, search
for a similar one, and serve the cached completion **only if** similarity clears
the threshold *and* the parameters are compatible.

### The correctness guards are the point

A cache that answers a question nobody asked is worse than no cache. The latency
win is invisible; a wrong answer is not.

- **High temperature bypasses entirely.** Above `max_temperature` (default 0.3)
  the caller is asking for variety. Returning the same completion every time
  would satisfy the API contract while defeating the actual request — the one
  case where a *correct* cache hit is still wrong.
- **`X-No-Cache: true` (or `Cache-Control: no-cache`) is absolute.** A caller
  asking for a fresh answer has a reason.
- **Similarity is necessary, not sufficient.** A vector match means the prompts
  are *similar*. Before serving, the entry must also match on model, sit within
  `max_temp_delta` of the request's temperature, fit under its `max_tokens`, and
  be inside its TTL. A 2000-token cached answer is never served to a
  `max_tokens: 50` request.
- **Tenants are isolated in the key itself**, and by a Qdrant pre-filter — one
  tenant can never read another's completions.
- **False hits are tracked next to hit rate.** `gateway_cache_false_hits_total`
  sits beside the hit-rate panel because a hit rate on its own is a vanity
  metric.

### Tuning the threshold

The default is **0.95**, deliberately conservative. Below ~0.90, paraphrases that
*mean different things* start matching. Tune it with evidence, not vibes: the
`gateway_cache_similarity` histogram records accepted and rejected scores
separately, and the dashboard plots both. If the two distributions converge, the
threshold is in the noise. Config validation refuses anything below 0.80.

---

## Quick start

```bash
git clone <repo> && cd gatewayllm
cp .env.example .env          # add at least one provider key
make up                       # gateway + sidecar + Redis + Qdrant + Postgres + Jaeger + Grafana
make seed                     # create a demo tenant, print an API key
```

```bash
curl localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer glm_live_..." \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","temperature":0,"messages":[{"role":"user","content":"What is a monad?"}]}'
```

Send it twice and watch the `X-Cache` response header go `miss` → `exact_hit`.
Rephrase the question and it becomes `semantic_hit`, with `X-Cache-Similarity`
showing the score.

| Service | URL |
|---|---|
| Gateway API | http://localhost:8080 |
| Grafana | http://localhost:3000 |
| Jaeger traces | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Qdrant dashboard | http://localhost:6333/dashboard |

First boot takes a few minutes: the sidecar image bakes in the embedding model.

### Without Docker

```bash
make build
./bin/gateway -config config.yaml
```

Every store is optional. With `cache`, `rate_limit`, and `meter` disabled, the
gateway runs as a pure proxy with no backing services at all.

### Deploying to the cloud

The container runs on any container host. For **Google Cloud Run** — the target
the spec names — see [deploy/cloudrun/README.md](deploy/cloudrun/README.md): the
same image, with managed stores (Upstash Redis, Qdrant Cloud, Neon Postgres) and
hosted embeddings swapped in via environment variables. `config.yaml` is fully
env-driven — `$PORT`, `METRICS_ADDR=inline` for single-port routing, store URLs,
and the embedder mode all come from the environment — so one image serves both
compose and Cloud Run with no code change.

A serverless-function platform (Vercel, Lambda) is a poor fit: the async meter
and background cache writes rely on the process staying alive after the response
returns, which those platforms don't guarantee.

---

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | streaming and non-streaming |
| POST | `/v1/embeddings` | routed and failed over like chat |
| GET | `/v1/models` | lists configured aliases |
| GET | `/healthz` `/readyz` | liveness / readiness |
| GET | `:9090/metrics` | Prometheus |

Response headers: `X-Cache` (`exact_hit` / `semantic_hit` / `miss` / `bypass`),
`X-Cache-Similarity`, `X-Provider`, `X-Request-Id`, and the `X-RateLimit-*` family.

---

## Routing

Aliases decouple the name clients use from the model that serves it. With the
`priority` strategy, config order *is* failover order:

```yaml
router:
  aliases:
    smart:
      strategy: priority
      targets:
        - { provider: openai, model: gpt-4o }        # try first
        - { provider: gemini, model: gemini-2.0-flash }  # if OpenAI is down
        - { provider: groq,   model: llama-3.3-70b-versatile }
```

`weighted` splits traffic to compare providers on live traffic; the unpicked
target still serves as a fallback.

### Failover is classification-driven

Retrying blindly wastes latency and quota. Every upstream error is classified,
and the classification decides what happens next:

| Failure | Retry? | Fail over? | Why |
|---|---|---|---|
| rate limit (429) | yes, honoring `Retry-After` | yes | transient |
| unavailable (5xx) | yes, jittered backoff | yes | transient |
| auth (401) | no | yes | this key is broken; another provider's may not be |
| invalid request | no | **no** | every provider rejects it identically |
| content filter | no | **no** | deterministic; shopping it around is pointless |
| context too long | no | yes | a bigger window may fit |

**Client errors never trip the circuit breaker.** If they did, one caller sending
malformed requests could take a provider offline for every other tenant.

**Streaming failover stops at the first byte.** Once a chunk reaches the client
the response has begun; switching providers mid-stream would splice two different
completions into one answer. After that point, failures surface as a terminal SSE
error event instead.

---

## Observability

The trace waterfall per request is `auth → cache-lookup → embed → route →
provider-call-with-retries → cache-write`. On a cache hit, the provider-call span
is **absent** — that gap is the feature, visible in Jaeger.

The Grafana dashboard (auto-provisioned, `GatewayLLM → Overview`) shows live hit
rate, cost saved, false hits, per-provider p95 latency, breaker state, and
spend-vs-avoided-spend.

```bash
go run ./cmd/glmctl usage -since 24h
```

Output shape (illustrative values — measured numbers live in
[Status](#status)):

```
TENANT  REQUESTS  TOKENS  SPENT    SAVED    HIT RATE
demo    1284      412903  $0.8241  $0.3117  38.1%
```

---

## Operating

```bash
go run ./cmd/glmctl create-tenant -id acme -name "Acme Corp"
go run ./cmd/glmctl create-key -tenant acme -rpm 600 -label "prod"
go run ./cmd/glmctl list-keys -tenant acme
go run ./cmd/glmctl revoke-key -id key_xxx
```

Keys are stored as SHA-256 digests and shown exactly once — a database leak
yields nothing replayable. (A fast hash is correct here, unlike for passwords:
keys are 32 bytes of entropy, so there is nothing to brute-force, and auth is on
the hot path of every request.)

---

## Design decisions worth defending

**Rate limiting and the breaker run as Redis Lua scripts.** Refill-check-decrement
must be atomic. Done as read-modify-write from Go, two concurrent requests both
read the same token count and both spend it — letting a tenant exceed their limit
under exactly the concurrency the limit exists to control.

**The cache fails open; the limiter fails open; auth fails closed.** A broken
cache degrades to a miss because the provider call is always a correct fallback.
A broken rate limiter allows traffic rather than turning a Redis blip into a full
outage. A broken *auth backend* returns 503, never 401 — an outage must not look
to clients like their credentials were revoked.

**Usage writes drop rather than block.** The ledger is for billing and
dashboards; stalling live traffic to protect its completeness is the wrong trade.
Drops are counted so the gap is visible rather than silent.

**Cache keys length-prefix every field.** Plain concatenation lets `("ab","c")`
and `("a","bc")` collide. Prompt case is preserved (it carries meaning); model
alias case is not (it doesn't).

**Cache hits replay the original token counts** instead of reporting zeros, so
the saving stays visible to the client.

---

## Development

```bash
make test        # race detector
make lint        # vet + gofmt
make cover
```

Tests need no network and no containers: a `Mock` provider implements the same
`Provider` interface, `miniredis` gives real Redis semantics (Lua included), and
the Qdrant and embedder tiers are tested against fake HTTP servers. The suite
asserts the things that would be dangerous to get wrong — that every field
changing a completion changes the cache key, that tenants can't read each other's
entries, that a hit skips the provider entirely, and that streaming refuses to
fail over past the commit point.

```
cmd/gateway   cmd/glmctl
internal/
  api         HTTP surface, auth, rate limiting
  cache       two-tier orchestration + guards
  embed       Embedder: sidecar | api
  router      aliasing + policy
  provider    Provider: openai | groq | gemini | mock
  resilience  timeout/retry/breaker/failover
  meter       async usage + cost ledger
  obs         OTel + Prometheus
  store       Redis / Qdrant / Postgres clients
sidecar/      Python embedder
```

---

## Status

Built in the spec's value-first order: passthrough proxy → multi-provider
failover → exact cache + rate limiting → semantic cache → metering and
observability. All five stages are complete.

### Measured

Live run (2026-07-19): the gateway against **real Groq traffic**
(`llama-3.1-8b-instant`) with the Redis exact tier, replaying a 30-prompt corpus
for 5 rounds — 180 requests total. Reproduce with the bundled tool:

```bash
go run ./cmd/glmbench -url http://localhost:8080 -key <key> -prompts bench/prompts.txt
```

| X-Cache | count | p50 | p95 |
|---|---|---|---|
| `miss` (real provider call) | 30 | 128 ms | 293 ms |
| `exact_hit` | 150 | **0.42 ms** | 1 ms |

- **~300× lower p50** on a cache hit than a provider call (0.42 ms vs 128 ms)
- **83.3% hit rate** on this workload, cutting provider spend **6×**
  ($0.00015 spent vs $0.00075 avoided, from `gateway_cost_usd_total` /
  `gateway_saved_usd_total`)
- **15,000+ req/s** cache-hit throughput at concurrency 8 (loopback)

Caveats, stated plainly: the hit rate is a property of this replay workload
(every prompt repeated 5×), not a production estimate — real-traffic hit rates
depend entirely on prompt repetition. Latencies include this machine's network
path to Groq. Numbers came from a local binary + local Redis, not the full
compose stack.

### Not yet measured

The **semantic tier** (Qdrant + embedder) and the Grafana dashboard on live
traffic — both need the full compose stack or a cloud deploy. The semantic
tier's threshold tuning and false-hit rate are exactly the numbers a paraphrase
corpus (e.g. Quora Question Pairs) through `glmbench` would produce; the
instrumentation (`gateway_cache_similarity`, `gateway_cache_false_hits_total`)
is already wired.
