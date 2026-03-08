"""
Part 1 — Vector Store (ChromaDB)

Handles persisting embeddings and metadata into ChromaDB, and provides
retrieval functions used by the cache and API layers.

Why ChromaDB:
- Persistent by default (SQLite + parquet under the hood) — survives restarts
- Native Python client — no separate server process to manage
- Supports metadata filtering — we use this for cluster-based retrieval
- Cosine similarity built-in — aligns with our normalised embeddings
- Lightweight enough for Docker — no JVM, no heavy dependencies

Trade-off: ChromaDB is single-process. For production multi-worker deployments,
Qdrant or Milvus would be needed. For this task's scale (~20K docs), ChromaDB
is the right fit.
"""

import logging
from typing import Dict, List, Optional

import chromadb
import numpy as np

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client backed by disk."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(
    client: chromadb.PersistentClient, name: str = CHROMA_COLLECTION_NAME
) -> chromadb.Collection:
    """Get existing collection or create a new one with cosine distance."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(
    collection: chromadb.Collection,
    texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    original_indices: List[int],
    target_names: List[str],
    cluster_distributions: Optional[np.ndarray] = None,
    batch_size: int = 500,
) -> None:
    """
    Upsert documents into ChromaDB with metadata.

    Each document gets:
    - id: deterministic string based on original index
    - embedding: 384-dim vector
    - document: first 500 chars of text (ChromaDB stores this for retrieval)
    - metadata:
        - original_index: position in the raw dataset
        - newsgroup_label: human-readable category name
        - label_id: integer label
        - text_preview: first 200 chars for quick display
        - dominant_cluster: (if cluster_distributions provided) highest-prob cluster
        - cluster_dist_top3: (if provided) JSON string of top-3 (cluster, prob) pairs

    We batch in chunks of 500 to stay within ChromaDB's per-call limits.
    """
    n = len(texts)
    logger.info("Ingesting %d documents into ChromaDB...", n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = [f"doc_{original_indices[i]}" for i in range(start, end)]
        docs = [texts[i][:500] for i in range(start, end)]
        embs = embeddings[start:end].tolist()

        metadatas = []
        for i in range(start, end):
            meta: Dict = {
                "original_index": int(original_indices[i]),
                "newsgroup_label": target_names[int(labels[i])],
                "label_id": int(labels[i]),
                "text_preview": texts[i][:200],
            }
            if cluster_distributions is not None:
                dist = cluster_distributions[i]
                dominant = int(np.argmax(dist))
                top3_idx = np.argsort(dist)[-3:][::-1]
                top3 = [(int(idx), round(float(dist[idx]), 4)) for idx in top3_idx]
                meta["dominant_cluster"] = dominant
                meta["cluster_dist_top3"] = str(top3)
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embs,
            metadatas=metadatas,
        )

    logger.info("Ingestion complete: %d documents in collection '%s'", n, collection.name)


def query_collection(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    n_results: int = 5,
    where: Optional[Dict] = None,
) -> Dict:
    """
    Query ChromaDB for the top-k most similar documents.

    Args:
        collection: ChromaDB collection
        query_embedding: 1-D array of shape (384,)
        n_results: number of results to return
        where: optional metadata filter dict

    Returns:
        dict with keys: ids, documents, metadatas, distances
        (distances are cosine distances; similarity = 1 - distance)
    """
    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    return results
