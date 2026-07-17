FROM python:3.12-slim

WORKDIR /app

# Install deps first: this layer is cached across code changes, and the torch
# dependency tree is far too large to re-resolve on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Bake the model into the image. Downloading it at container start would make
# every cold start wait on a network fetch, and would break entirely on a host
# with no egress to HuggingFace.
ENV EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
    import os; SentenceTransformer(os.environ['EMBED_MODEL'])"

# Run as a non-root user.
RUN useradd -m -u 1000 embedder && chown -R embedder:embedder /app /home/embedder
USER embedder

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/readyz')"

# Single worker: the model holds hundreds of MB of weights, and each worker
# would load its own copy. Scale with replicas, not workers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
