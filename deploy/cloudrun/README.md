# Deploying GatewayLLM to Google Cloud Run

Cloud Run runs the gateway as a **container** — a long-lived process, which is
what this project is (unlike a serverless-function platform, where the async
meter and background cache writes would be killed the moment a response returns).
The existing [Dockerfile](../../Dockerfile) is deployed as-is; only the backing
stores move to managed services.

## What you need

| Piece | Managed service | Free tier | What to grab |
|---|---|---|---|
| Redis | [Upstash](https://upstash.com) | yes | the `rediss://` connection URL |
| Vectors | [Qdrant Cloud](https://cloud.qdrant.io) | yes (1 GB) | cluster URL + API key |
| Postgres | [Neon](https://neon.tech) | yes | connection string (`?sslmode=require`) |
| Embeddings | OpenAI | pay-per-use | reuse `OPENAI_API_KEY` |
| Provider keys | OpenAI / Groq / Gemini | — | at least one |

The Python embedder sidecar does **not** run on Cloud Run — a Cloud Run service
is one container. Instead the deploy uses hosted embeddings (`EMBED_KIND=api`),
which the config already supports. That is the one behavioural change from the
compose stack, and it costs a fraction of a cent per cache lookup on a miss.

## How the image adapts to Cloud Run

No separate cloud image is needed — `config.yaml` is fully environment-driven and
is baked into the container. The variables below reshape it:

| Env var | Effect |
|---|---|
| `PORT` | injected by Cloud Run; the listener binds it (`addr: ":${PORT:-8080}"`) |
| `METRICS_ADDR=inline` | mounts `/metrics` on the main port — Cloud Run routes to only one |
| `EMBED_KIND=api` | use hosted embeddings instead of the sidecar |
| `EMBEDDER_URL`, `EMBED_MODEL`, `VECTOR_SIZE=1536` | point at OpenAI embeddings; the vector size **must** match the Qdrant collection |
| `REDIS_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `POSTGRES_URL` | the managed stores |
| `OTLP_ENDPOINT` | leave empty unless you run a collector (see Observability) |

## Steps

### 1. Provision the stores

Sign up for the three services above and copy their connection strings. On Neon,
use the **pooled** connection string.

### 2. Configure secrets

Copy the template and fill it in:

```bash
cp deploy/cloudrun/.env.cloud.example deploy/cloudrun/.env.cloud
$EDITOR deploy/cloudrun/.env.cloud
```

### 3. Deploy

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
./deploy/cloudrun/deploy.sh
```

The script builds the image with Cloud Build (from the Dockerfile), pushes it,
and deploys the service with every env var set. First deploy takes a few minutes.

### 4. Seed a tenant and key

Auth reads from Postgres, so create a key against the **same** Neon database the
service uses. `glmctl` runs locally and connects over the public Neon endpoint:

```bash
export POSTGRES_URL='postgres://...neon.tech/gatewayllm?sslmode=require'
go run ./cmd/glmctl create-tenant -id demo -name "Demo"
go run ./cmd/glmctl create-key   -tenant demo -label "cloud run"
```

The migration runs automatically on the service's first boot; `glmctl` also runs
it, so either order works.

### 5. Call it

```bash
curl https://YOUR-SERVICE-URL/v1/chat/completions \
  -H "Authorization: Bearer glm_live_..." \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","temperature":0,"messages":[{"role":"user","content":"hello"}]}'
```

Send it twice; `X-Cache` goes `miss` → `exact_hit`.

## Two settings that matter

**`--no-cpu-throttling` (the script sets it).** By default Cloud Run throttles a
container's CPU to near zero between requests. That would stall the async meter
flush and the background cache write until the next request arrived — the writes
would land eventually, but late. Disabling throttling keeps background goroutines
running so usage and cache writes complete promptly. It shifts billing to
instance-based; on the free tier this is still comfortably free for demo traffic.

**`--min-instances=1` (the script sets it).** `build()` connects all three stores
and runs migrations at startup. With scale-to-zero, that repeats on every cold
start (harmless — migrations are idempotent — but wasteful and adds cold-start
latency). One warm instance avoids it. Set it to `0` if you would rather scale to
zero and accept cold starts.

## Observability

Prometheus, Grafana, and Jaeger are **not** deployed here — they are separate
long-running services, out of scope for a single Cloud Run service. Options:

- **Metrics:** `/metrics` is live on the service URL (inline). Point Grafana Cloud
  or Google Managed Prometheus at it. Note it is unauthenticated — restrict scrape
  access with Cloud Run IAM if the service is not already private.
- **Traces:** set `OTLP_ENDPOINT` to a collector (Grafana Tempo, Honeycomb, or the
  Google Cloud Trace OTLP endpoint) to get the per-request waterfall.

Leaving both unset is fine — the gateway runs identically, just without the
dashboards.

## Security note

The deploy script uses `--allow-unauthenticated` so the API is reachable (it has
its own API-key auth). That also exposes `/metrics` publicly. If you want the
scrape endpoint private, make the service require IAM auth and put an authenticating
proxy in front, or move metrics back to a dedicated port and scrape it from inside
the VPC.
