# ── Single-stage build (CPU-only torch keeps image manageable) ─────────────────
FROM python:3.12-slim

WORKDIR /app

# Install build dependencies needed by some packages, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than default CUDA build)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/

# Create directories for persistent data
RUN mkdir -p data models chroma_db cache

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/cache/stats')" || exit 1

# Default: run data pipeline then start API server
# For production, split these into separate steps (prepare_data as init container)
CMD ["sh", "-c", "python -m scripts.prepare_data && uvicorn src.api:app --host 0.0.0.0 --port 8000"]
