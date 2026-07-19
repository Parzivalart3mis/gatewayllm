#!/usr/bin/env bash
#
# Deploy GatewayLLM to Google Cloud Run from source (Cloud Build compiles the
# Dockerfile). Reads deploy/cloudrun/.env.cloud for configuration.
#
#   ./deploy/cloudrun/deploy.sh
#
# Prereqs: gcloud CLI authenticated (`gcloud auth login`) and a project selected
# (`gcloud config set project ...`).

set -euo pipefail

# Resolve the repo root so the script works from any CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/deploy/cloudrun/.env.cloud"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found. Copy .env.cloud.example to .env.cloud and fill it in." >&2
  exit 1
fi

# Load the env file. Every KEY=VALUE becomes a shell variable.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${SERVICE_NAME:=gatewayllm}"
: "${GCP_REGION:=us-central1}"

# Fail early on the non-negotiables rather than after a slow build.
require() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: $name is required in .env.cloud" >&2
    exit 1
  fi
}
require REDIS_URL
require POSTGRES_URL

# Qdrant and an embeddings backend are only needed when the semantic tier is on.
# With SEMANTIC_ENABLED=false the gateway ships the exact-match tier alone and
# needs neither, so demanding them here would block a valid deploy.
if [[ "${SEMANTIC_ENABLED:-true}" != "false" ]]; then
  require QDRANT_URL
  if [[ "${EMBED_KIND:-api}" == "api" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "error: the semantic tier needs an embeddings key." >&2
    echo "       Set OPENAI_API_KEY, or set SEMANTIC_ENABLED=false to deploy" >&2
    echo "       with the exact-match cache only." >&2
    exit 1
  fi
fi

# Build the comma-separated env-var list Cloud Run expects. Only non-empty
# values are forwarded, so an unset optional key does not overwrite the config
# default with an empty string.
declare -a PAIRS=()
add() {
  local key="$1" val="${2:-}"
  [[ -n "$val" ]] && PAIRS+=("${key}=${val}")
}

add OPENAI_API_KEY   "${OPENAI_API_KEY:-}"
add GROQ_API_KEY     "${GROQ_API_KEY:-}"
add GEMINI_API_KEY   "${GEMINI_API_KEY:-}"
add REDIS_URL        "$REDIS_URL"
add QDRANT_URL       "$QDRANT_URL"
add QDRANT_API_KEY   "${QDRANT_API_KEY:-}"
add POSTGRES_URL     "$POSTGRES_URL"
add EMBED_KIND       "${EMBED_KIND:-api}"
add EMBEDDER_URL     "${EMBEDDER_URL:-https://api.openai.com/v1}"
add EMBED_MODEL      "${EMBED_MODEL:-text-embedding-3-small}"
add VECTOR_SIZE      "${VECTOR_SIZE:-1536}"
add SEMANTIC_ENABLED "${SEMANTIC_ENABLED:-true}"
add METRICS_ADDR     "${METRICS_ADDR:-inline}"
add OTLP_ENDPOINT    "${OTLP_ENDPOINT:-}"

# Join with commas. A literal comma inside a value would break this; connection
# URLs do not contain commas, so it is safe here.
ENV_VARS="$(IFS=,; echo "${PAIRS[*]}")"

echo "Deploying '$SERVICE_NAME' to $GCP_REGION ..."
echo "  ${#PAIRS[@]} env vars set (values hidden)"

gcloud run deploy "$SERVICE_NAME" \
  --source "$ROOT" \
  --region "$GCP_REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 1 \
  --max-instances 5 \
  --no-cpu-throttling \
  --timeout 600 \
  --set-env-vars "$ENV_VARS"

echo
echo "Deployed. Service URL:"
gcloud run services describe "$SERVICE_NAME" --region "$GCP_REGION" \
  --format='value(status.url)'
echo
echo "Next: seed a key against the same Postgres, then call /v1/chat/completions."
echo "See deploy/cloudrun/README.md step 4."
