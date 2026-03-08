"""
Data Pipeline — Prepare corpus, embed, cluster, and ingest into ChromaDB.

Run this once before starting the API server:
    python -m scripts.prepare_data

This script:
1. Loads and cleans the 20 Newsgroups dataset
2. Embeds all documents with sentence-transformers
3. Sweeps K for GMM and selects the optimal number of clusters
4. Fits the final GMM and computes fuzzy cluster assignments
5. Ingests everything into ChromaDB with cluster metadata
6. Saves the GMM + PCA models to disk
7. Saves analysis artifacts (K sweep results, embeddings) for the notebook
"""

import logging
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, ".")

from src.data_preparation import prepare_corpus
from src.vector_store import get_chroma_client, get_or_create_collection, ingest_documents
from src.clustering import (
    reduce_dimensions,
    sweep_k,
    fit_gmm,
    get_cluster_distributions,
    save_gmm,
    compute_entropy,
    get_boundary_documents,
    cluster_label_matrix,
)
from src.config import DATA_DIR, GMM_K_MIN, GMM_K_MAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ── Part 1: Prepare corpus ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PART 1: Corpus Preparation & Embedding")
    logger.info("=" * 60)

    texts, labels, original_indices, target_names, embeddings = prepare_corpus()
    logger.info("Corpus ready: %d documents, %d-dim embeddings", len(texts), embeddings.shape[1])

    # Save embeddings and metadata for analysis notebook
    np.save(DATA_DIR / "embeddings.npy", embeddings)
    np.save(DATA_DIR / "labels.npy", labels)
    with open(DATA_DIR / "target_names.json", "w") as f:
        json.dump(target_names, f)
    with open(DATA_DIR / "original_indices.json", "w") as f:
        json.dump(original_indices, f)
    # Save texts (one per line, newlines replaced)
    with open(DATA_DIR / "cleaned_texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")

    logger.info("Raw artifacts saved to %s", DATA_DIR)

    # ── Part 2: Fuzzy Clustering ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PART 2: Fuzzy Clustering (GMM)")
    logger.info("=" * 60)

    # Step 2.1: Dimensionality reduction
    reduced_embeddings, pca = reduce_dimensions(embeddings)

    # Step 2.2: Sweep K to find optimal number of clusters
    logger.info("Sweeping K from %d to %d...", GMM_K_MIN, GMM_K_MAX)
    sweep_results = sweep_k(reduced_embeddings)

    # Save sweep results for notebook visualization
    with open(DATA_DIR / "k_sweep_results.json", "w") as f:
        json.dump(sweep_results, f)

    # Select K: use BIC minimum as primary criterion
    k_values = sweep_results["k_values"]
    bic_scores = sweep_results["bic_scores"]
    sil_scores = sweep_results["silhouette_scores"]

    best_bic_idx = int(np.argmin(bic_scores))
    best_sil_idx = int(np.argmax(sil_scores))

    logger.info(
        "Best K by BIC: %d (BIC=%.0f) | Best K by Silhouette: %d (Sil=%.4f)",
        k_values[best_bic_idx],
        bic_scores[best_bic_idx],
        k_values[best_sil_idx],
        sil_scores[best_sil_idx],
    )

    # Decision: use BIC-optimal K, as it directly penalises model complexity
    # while rewarding fit. Silhouette is computed on hard assignments and
    # undervalues the fuzzy nature of our clustering. If BIC and silhouette
    # disagree strongly, we prefer BIC as the more principled criterion for GMM.
    chosen_k = k_values[best_bic_idx]
    logger.info("Chosen K = %d (BIC-optimal)", chosen_k)

    # Step 2.3: Fit final GMM
    gmm = fit_gmm(reduced_embeddings, n_clusters=chosen_k)
    cluster_distributions = get_cluster_distributions(gmm, reduced_embeddings)

    # Save GMM + PCA for API server
    save_gmm(gmm, pca)

    # Save cluster distributions for analysis
    np.save(DATA_DIR / "cluster_distributions.npy", cluster_distributions)

    # Log some cluster analysis
    entropy = compute_entropy(cluster_distributions)
    logger.info(
        "Entropy stats: mean=%.3f, median=%.3f, max=%.3f",
        entropy.mean(),
        np.median(entropy),
        entropy.max(),
    )

    boundary_indices = get_boundary_documents(cluster_distributions, top_n=5)
    logger.info("Top 5 boundary documents (highest entropy):")
    for idx in boundary_indices:
        top2 = np.argsort(cluster_distributions[idx])[-2:][::-1]
        probs = [cluster_distributions[idx][c] for c in top2]
        logger.info(
            "  Doc %d: clusters %s, probs %s, text: '%s...'",
            original_indices[idx],
            top2.tolist(),
            [round(p, 3) for p in probs],
            texts[idx][:100],
        )

    # Cluster-label co-occurrence
    co_matrix = cluster_label_matrix(cluster_distributions, labels, chosen_k)
    np.save(DATA_DIR / "cluster_label_matrix.npy", co_matrix)

    # ── Part 1 continued: Ingest into ChromaDB ─────────────────────────────
    logger.info("=" * 60)
    logger.info("Ingesting into ChromaDB with cluster metadata...")
    logger.info("=" * 60)

    client = get_chroma_client()
    # Delete existing collection if any (fresh ingest)
    try:
        client.delete_collection(name="newsgroups")
        logger.info("Deleted existing collection")
    except Exception:
        pass

    collection = get_or_create_collection(client)
    ingest_documents(
        collection=collection,
        texts=texts,
        embeddings=embeddings,
        labels=labels,
        original_indices=original_indices,
        target_names=target_names,
        cluster_distributions=cluster_distributions,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Documents: %d", len(texts))
    logger.info("  Clusters: %d", chosen_k)
    logger.info("  ChromaDB: %d documents in collection", collection.count())
    logger.info("  Models saved to: models/")
    logger.info("  Data artifacts: data/")
    logger.info("=" * 60)
    logger.info("You can now start the API with:")
    logger.info("  uvicorn src.api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
