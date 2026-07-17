"""Embedding sidecar for GatewayLLM.

This is the one component that lives outside the Go binary. The mature embedding
models ship as Python, and reimplementing inference in Go to preserve
single-binary purity would be a bad trade. Keeping it local instead of calling a
hosted API means the semantic cache costs nothing per lookup and no prompt text
leaves the host.

Contract (consumed by internal/embed/sidecar.go):
    GET  /info    -> {"model": str, "dims": int}
    POST /embed   -> {"vectors": [[float]], "model": str, "dims": int}
    GET  /healthz -> {"status": "ok"}
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("embedder")

# all-MiniLM-L6-v2: 384 dimensions, ~80MB, and fast enough on CPU to sit on a
# cache lookup path. Larger models score marginally better on retrieval but
# would put tens of milliseconds in front of every semantic-tier miss.
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Cap the batch so one request cannot exhaust memory on a small VPS.
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))

app = FastAPI(title="gatewayllm-embedder", version="1.0.0")

_model: SentenceTransformer | None = None
# encode() is not documented as thread-safe; FastAPI serves sync handlers from a
# thread pool, so concurrent requests would otherwise race inside the model.
_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Return the loaded model, loading it on first use."""
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    return _model


@app.on_event("startup")
def load_model() -> None:
    """Load the model at startup.

    Done eagerly so the container is not marked ready until inference actually
    works: a lazy load would make the first real request pay a multi-second
    download and then time out the cache lookup that triggered it.
    """
    global _model
    log.info("loading embedding model %s", MODEL_NAME)
    _model = SentenceTransformer(MODEL_NAME)
    dims = _model.get_sentence_embedding_dimension()
    log.info("model ready: %s (%d dimensions)", MODEL_NAME, dims)


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    # Accepted and ignored: the sidecar serves one model, but the field keeps the
    # request shape identical to the API embedder's so the two are swappable.
    model: str | None = None


class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    model: str
    dims: int


@app.get("/healthz")
def healthz() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness: the model is loaded and can serve."""
    if _model is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    """Report the model and its dimensionality.

    The gateway calls this to verify the embedder agrees with the Qdrant
    collection's vector size, so a model swap fails at startup rather than
    producing vectors the collection rejects.
    """
    model = get_model()
    return {"model": MODEL_NAME, "dims": model.get_sentence_embedding_dimension()}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed one or more texts, returning vectors in input order."""
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(req.texts)} exceeds the limit of {MAX_BATCH}",
        )

    model = get_model()
    with _lock:
        # normalize_embeddings=True is required, not cosmetic: the Qdrant
        # collection uses cosine distance, which assumes unit-length vectors.
        vectors = model.encode(
            req.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return EmbedResponse(
        vectors=vectors.tolist(),
        model=MODEL_NAME,
        dims=int(vectors.shape[1]),
    )
