"""
Part 4 — FastAPI Service

Exposes the semantic cache as a live API with three endpoints:
- POST /query     → semantic search with cache
- GET /cache/stats → cache statistics
- DELETE /cache    → flush cache

State management:
Uses FastAPI's `lifespan` context manager (the modern approach — `on_event`
is deprecated). All shared state (embedding model, GMM, ChromaDB collection,
semantic cache) is initialised once at startup and attached to `app.state`.
This avoids globals and makes testing easier.

On shutdown, the cache is serialised to disk so it survives restarts.
"""

import ast
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from src.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TOP_K,
    API_HOST,
    API_PORT,
)
from src.data_preparation import load_embedding_model
from src.vector_store import get_chroma_client, get_or_create_collection, query_collection
from src.clustering import load_gmm
from src.semantic_cache import SemanticCache
from src.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentResult,
    CacheStatsResponse,
    CacheFlushResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load all models and restore cache from disk.
    Shutdown: persist cache to disk.
    """
    logger.info("Starting up — loading models and data...")

    # Load embedding model
    app.state.embedding_model = load_embedding_model()

    # Load GMM + PCA
    gmm, pca = load_gmm()
    app.state.gmm = gmm
    app.state.pca = pca

    # Connect to ChromaDB
    client = get_chroma_client()
    app.state.collection = get_or_create_collection(client)
    logger.info(
        "ChromaDB collection '%s' loaded with %d documents",
        app.state.collection.name,
        app.state.collection.count(),
    )

    # Initialise semantic cache and try to restore from disk
    app.state.cache = SemanticCache(threshold=CACHE_SIMILARITY_THRESHOLD)
    app.state.cache.load_from_disk()

    logger.info("Startup complete. Cache threshold=%.2f", CACHE_SIMILARITY_THRESHOLD)

    yield  # ← app is now running

    # Shutdown: persist cache
    logger.info("Shutting down — saving cache to disk...")
    app.state.cache.save_to_disk()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Semantic Search System — 20 Newsgroups",
    description=(
        "Fuzzy-clustered semantic search with a custom semantic cache layer. "
        "No Redis, no caching libraries — everything built from first principles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Semantic Search System — 20 Newsgroups",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """Embed a single query and return normalised vector."""
    model = app.state.embedding_model
    embedding = model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    return np.array(embedding[0], dtype=np.float32)


def get_cluster_info(query_embedding: np.ndarray):
    """
    Run the query embedding through PCA + GMM to get cluster distribution.
    Returns (dominant_cluster, cluster_probabilities).
    """
    pca = app.state.pca
    gmm = app.state.gmm

    reduced = pca.transform(query_embedding.reshape(1, -1))
    probs = gmm.predict_proba(reduced)[0]
    dominant = int(np.argmax(probs))
    return dominant, probs


def compute_result(query_embedding: np.ndarray) -> list:
    """
    On cache miss: query ChromaDB for top-K similar documents.
    Returns list of DocumentResult-compatible dicts.
    """
    collection = app.state.collection
    raw = query_collection(collection, query_embedding, n_results=CACHE_TOP_K)

    results = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]
        similarity = 1.0 - distance  # ChromaDB cosine distance → similarity

        # Parse cluster distribution from metadata string
        cluster_dist_str = meta.get("cluster_dist_top3", "[]")
        try:
            cluster_dist_top3 = ast.literal_eval(cluster_dist_str)
        except (ValueError, SyntaxError):
            cluster_dist_top3 = []

        results.append({
            "text_preview": meta.get("text_preview", ""),
            "similarity_score": round(similarity, 4),
            "newsgroup_label": meta.get("newsgroup_label", "unknown"),
            "cluster_distribution_top3": cluster_dist_top3,
        })

    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Semantic search with cache.

    Flow:
    1. Embed the query
    2. Get cluster distribution from GMM
    3. Check semantic cache (cluster-aware lookup)
    4. On hit: return cached result
    5. On miss: query ChromaDB, store in cache, return
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1: embed
    query_embedding = embed_query(query_text)

    # Step 2: cluster info
    dominant_cluster, cluster_probs = get_cluster_info(query_embedding)

    # Step 3: cache lookup
    cache = app.state.cache
    cache_result = cache.lookup(query_embedding, cluster_probs)

    if cache_result is not None:
        # Cache HIT
        entry, similarity = cache_result
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(similarity, 4),
            result=[DocumentResult(**doc) for doc in entry.result],
            dominant_cluster=dominant_cluster,
        )
    else:
        # Cache MISS — compute from vector DB
        result = compute_result(query_embedding)

        # Store in cache for future hits
        cache.store(
            query=query_text,
            query_embedding=query_embedding,
            result=result,
            dominant_cluster=dominant_cluster,
            cluster_probabilities=cluster_probs,
        )

        return QueryResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=[DocumentResult(**doc) for doc in result],
            dominant_cluster=dominant_cluster,
        )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats_endpoint():
    """Return current cache statistics."""
    return CacheStatsResponse(**app.state.cache.get_stats())


@app.delete("/cache", response_model=CacheFlushResponse)
async def cache_flush_endpoint():
    """Flush the entire cache and reset all stats."""
    previous = app.state.cache.size
    app.state.cache.flush()
    return CacheFlushResponse(
        message="Cache flushed successfully",
        previous_entries=previous,
    )
