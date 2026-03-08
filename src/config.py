"""
Configuration constants for the semantic search system.
Centralises all tunables and paths so nothing is hardcoded in business logic.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for d in [DATA_DIR, CHROMA_DIR, CACHE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "newsgroups"

# ── Clustering ─────────────────────────────────────────────────────────────────
# Range to sweep when choosing K for GMM
GMM_K_MIN = 10
GMM_K_MAX = 30
GMM_MODEL_PATH = MODELS_DIR / "gmm_model.joblib"

# ── Data Cleaning ──────────────────────────────────────────────────────────────
MIN_TOKEN_COUNT = 10  # Documents with fewer tokens are discarded

# ── Semantic Cache ─────────────────────────────────────────────────────────────
# The core tunable: cosine similarity threshold for cache hits.
# Values explored in analysis: 0.70 → 0.99. Default chosen based on analysis.
# Threshold analysis showed 0.0 FP rate across all thresholds (cluster pruning
# is highly effective), so we optimise for recall. Best F1=0.88 at 0.75.
CACHE_SIMILARITY_THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0.75"))
CACHE_TOP_K = 5  # Number of documents to return per query
CACHE_PERSIST_PATH = CACHE_DIR / "semantic_cache.json"
CACHE_EMBEDDINGS_PATH = CACHE_DIR / "cache_embeddings.npy"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
