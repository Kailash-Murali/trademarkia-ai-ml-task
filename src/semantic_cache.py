"""
Part 3 — Semantic Cache (Built from First Principles)

No Redis, Memcached, or caching libraries. This entire cache is handwritten.

Architecture:
─────────────
The cache stores previously computed query results and avoids redundant
vector DB lookups when a semantically similar query has been seen before.

Key insight: two users asking "What's the latest on Middle East politics?" and
"Tell me about political developments in the Middle East" should get the same
cached result — a traditional exact-match cache fails here entirely.

Data structures:
- entries: dict mapping entry_index (int) → CacheEntry
- embeddings: numpy array of cached query embeddings (N × 384)
- cluster_index: dict mapping cluster_id → list of entry indices
  This is the critical optimisation: instead of comparing a new query against
  ALL cached embeddings (O(N)), we first identify the query's cluster(s) via
  GMM and only compare against entries in those clusters (O(N/K) average case).
- threshold: the tunable similarity threshold for cache hits

The Tunable — Similarity Threshold:
────────────────────────────────────
This is the single most important parameter in the system. It controls the
boundary between "close enough to reuse" and "different enough to recompute".

- Too low (e.g., 0.70): high hit rate, but the cache returns results for
  queries that are semantically different → stale/wrong results.
- Too high (e.g., 0.99): near-zero hit rate, cache is useless — only exact
  paraphrases match.
- Sweet spot (empirically ~0.85): balances reuse with correctness. The
  threshold analysis script explores this in detail.

The threshold is exposed as an environment variable (CACHE_THRESHOLD) so it
can be tuned without code changes.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_PERSIST_PATH,
    CACHE_EMBEDDINGS_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached query result."""

    query: str
    embedding_index: int  # index into the cache's embedding matrix
    result: Any  # the computed result (list of retrieved documents)
    dominant_cluster: int
    cluster_probabilities: List[float]  # full GMM distribution for this query
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "embedding_index": self.embedding_index,
            "result": self.result,
            "dominant_cluster": self.dominant_cluster,
            "cluster_probabilities": self.cluster_probabilities,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CacheEntry":
        return cls(**d)


@dataclass
class CacheStats:
    """Cache performance counters."""

    hit_count: int = 0
    miss_count: int = 0

    @property
    def total_queries(self) -> int:
        return self.hit_count + self.miss_count

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.hit_count / self.total_queries

    def to_dict(self) -> Dict:
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
        }


class SemanticCache:
    """
    Cluster-aware semantic cache for query results.

    Lookup algorithm:
    1. Embed the incoming query.
    2. Predict query's cluster distribution via GMM.
    3. Identify candidate clusters: any cluster where P(cluster | query) > 0.1.
    4. Retrieve cached entries from those clusters only (cluster_index).
    5. Compute cosine similarity between query embedding and candidate embeddings.
    6. If max similarity >= threshold → cache hit.
    7. Else → cache miss, compute result and store.

    Why cluster-based pruning matters:
    As the cache grows to thousands of entries, brute-force comparison against
    every cached embedding becomes a bottleneck. Cluster pruning reduces the
    search space by ~K× on average (where K is the number of clusters).
    For K=20, that's a 20× speedup at scale.
    """

    # Minimum GMM probability for a cluster to be considered "relevant"
    # for cache lookup. Clusters with lower probability are pruned.
    CLUSTER_RELEVANCE_THRESHOLD = 0.1

    def __init__(self, threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.entries: Dict[int, CacheEntry] = {}  # entry_index → CacheEntry
        self._embeddings_list: List[np.ndarray] = []  # grows dynamically
        self._embeddings_matrix: Optional[np.ndarray] = None  # cached matrix
        self._matrix_dirty: bool = True  # needs rebuild
        self.cluster_index: Dict[int, List[int]] = {}  # cluster_id → entry indices
        self.stats = CacheStats()
        self._next_index = 0

    @property
    def embeddings_matrix(self) -> Optional[np.ndarray]:
        """Lazy-rebuild the embedding matrix when entries change."""
        if self._matrix_dirty and self._embeddings_list:
            self._embeddings_matrix = np.vstack(self._embeddings_list)
            self._matrix_dirty = False
        return self._embeddings_matrix

    @property
    def size(self) -> int:
        return len(self.entries)

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_probabilities: np.ndarray,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Check if a semantically similar query exists in the cache.

        Args:
            query_embedding: normalised 1-D embedding of the new query
            cluster_probabilities: GMM probability distribution for the query

        Returns:
            (matched_entry, similarity_score) on hit, None on miss
        """
        if self.size == 0:
            self.stats.miss_count += 1
            return None

        # Step 1: identify relevant clusters (P > threshold)
        relevant_clusters = np.where(
            cluster_probabilities > self.CLUSTER_RELEVANCE_THRESHOLD
        )[0].tolist()

        # Step 2: gather candidate entry indices from those clusters
        candidate_indices = set()
        for cluster_id in relevant_clusters:
            if cluster_id in self.cluster_index:
                candidate_indices.update(self.cluster_index[cluster_id])

        if not candidate_indices:
            # No cached entries in relevant clusters
            self.stats.miss_count += 1
            return None

        candidate_indices = sorted(candidate_indices)

        # Step 3: compute cosine similarity against candidates only
        matrix = self.embeddings_matrix
        if matrix is None:
            self.stats.miss_count += 1
            return None

        candidate_embeddings = matrix[candidate_indices]
        # Since embeddings are L2-normalised, dot product = cosine similarity
        similarities = candidate_embeddings @ query_embedding

        best_local_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_local_idx])
        best_entry_idx = candidate_indices[best_local_idx]

        # Step 4: threshold check
        if best_similarity >= self.threshold:
            self.stats.hit_count += 1
            return self.entries[best_entry_idx], best_similarity
        else:
            self.stats.miss_count += 1
            return None

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: int,
        cluster_probabilities: np.ndarray,
    ) -> CacheEntry:
        """
        Store a new query result in the cache.

        The entry is indexed by its dominant cluster in cluster_index for
        efficient lookup. We also add it to any cluster where P > relevance
        threshold, so queries with ambiguous cluster membership can be found
        from multiple cluster pathways.
        """
        entry_index = self._next_index
        self._next_index += 1

        entry = CacheEntry(
            query=query,
            embedding_index=entry_index,
            result=result,
            dominant_cluster=dominant_cluster,
            cluster_probabilities=cluster_probabilities.tolist(),
        )

        self.entries[entry_index] = entry
        self._embeddings_list.append(query_embedding.reshape(1, -1))
        self._matrix_dirty = True

        # Index entry under all relevant clusters
        relevant_clusters = np.where(
            cluster_probabilities > self.CLUSTER_RELEVANCE_THRESHOLD
        )[0].tolist()
        # Always include the dominant cluster
        if dominant_cluster not in relevant_clusters:
            relevant_clusters.append(dominant_cluster)

        for cluster_id in relevant_clusters:
            if cluster_id not in self.cluster_index:
                self.cluster_index[cluster_id] = []
            self.cluster_index[cluster_id].append(entry_index)

        logger.debug(
            "Cache STORE: '%s...' → entry %d (cluster %d)",
            query[:50],
            entry_index,
            dominant_cluster,
        )

        return entry

    def flush(self) -> None:
        """Clear all cache data and reset stats."""
        self.entries.clear()
        self._embeddings_list.clear()
        self._embeddings_matrix = None
        self._matrix_dirty = True
        self.cluster_index.clear()
        self.stats = CacheStats()
        self._next_index = 0
        logger.info("Cache flushed")

    def get_stats(self) -> Dict:
        """Return cache statistics as a dict."""
        return {
            "total_entries": self.size,
            "hit_count": self.stats.hit_count,
            "miss_count": self.stats.miss_count,
            "hit_rate": round(self.stats.hit_rate, 4),
        }

    # ── Serialisation ──────────────────────────────────────────────────────

    def save_to_disk(self) -> None:
        """
        Persist cache to disk as JSON (entries + metadata) + numpy (embeddings).
        This allows cache state to survive server restarts.
        """
        if self.size == 0:
            logger.info("Cache is empty, skipping save")
            return

        # Save entries as JSON
        data = {
            "threshold": self.threshold,
            "next_index": self._next_index,
            "stats": self.stats.to_dict(),
            "cluster_index": {str(k): v for k, v in self.cluster_index.items()},
            "entries": {
                str(k): v.to_dict() for k, v in self.entries.items()
            },
        }
        with open(CACHE_PERSIST_PATH, "w") as f:
            json.dump(data, f)

        # Save embedding matrix as numpy
        if self.embeddings_matrix is not None:
            np.save(CACHE_EMBEDDINGS_PATH, self.embeddings_matrix)

        logger.info("Cache saved to disk (%d entries)", self.size)

    def load_from_disk(self) -> bool:
        """
        Restore cache from disk. Returns True if successful.
        """
        if not CACHE_PERSIST_PATH.exists() or not CACHE_EMBEDDINGS_PATH.exists():
            logger.info("No cached state found on disk")
            return False

        try:
            with open(CACHE_PERSIST_PATH, "r") as f:
                data = json.load(f)

            embeddings = np.load(CACHE_EMBEDDINGS_PATH)

            self.threshold = data.get("threshold", self.threshold)
            self._next_index = data["next_index"]
            self.stats = CacheStats(**data["stats"])
            self.cluster_index = {int(k): v for k, v in data["cluster_index"].items()}
            self.entries = {
                int(k): CacheEntry.from_dict(v) for k, v in data["entries"].items()
            }
            self._embeddings_list = [embeddings[i:i+1] for i in range(len(embeddings))]
            self._matrix_dirty = True

            logger.info("Cache restored from disk (%d entries)", self.size)
            return True

        except Exception as e:
            logger.warning("Failed to load cache from disk: %s", e)
            self.flush()
            return False
