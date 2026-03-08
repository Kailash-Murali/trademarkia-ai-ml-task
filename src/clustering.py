"""
Part 2 — Fuzzy Clustering via Gaussian Mixture Model

Why GMM over K-Means or hard clustering:
- GMM produces a probability distribution over K clusters for each document.
  This is the fuzzy assignment required by the task — a document about gun
  legislation gets non-zero membership in both a "politics" and a "firearms"
  cluster, proportional to its semantic proximity to each centroid.
- K-Means forces a single hard assignment, which destroys this nuance.
- GMM assumes each cluster is a Gaussian in embedding space. Since
  sentence-transformer embeddings tend to form roughly spherical clusters
  after L2 normalisation, this assumption is reasonable.

Why not LDA:
- LDA operates on word count vectors, not dense embeddings. Running LDA
  would discard the semantic information already captured by the embeddings.
  GMM on embeddings preserves the full semantic geometry.

Trade-off:
- GMM with full covariance on 384-dim embeddings is expensive. We use
  'tied' covariance (all clusters share one covariance matrix) to reduce
  parameters while preserving soft assignment quality. If quality is
  insufficient, switching to 'diag' or PCA-reduced embeddings is the
  upgrade path.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.config import GMM_K_MIN, GMM_K_MAX, GMM_MODEL_PATH

logger = logging.getLogger(__name__)

# ── Dimensionality Reduction ──────────────────────────────────────────────────
# GMM with full/tied covariance on 384 dimensions is expensive and can overfit.
# PCA to 50 dimensions retains ~85-90% of variance in typical sentence-transformer
# embeddings while making GMM fitting tractable and more stable numerically.
PCA_COMPONENTS = 50


def reduce_dimensions(
    embeddings: np.ndarray, n_components: int = PCA_COMPONENTS
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce embedding dimensionality via PCA before clustering.
    Returns reduced embeddings and the fitted PCA model (for transforming queries).
    """
    logger.info(
        "Reducing %d-dim embeddings to %d via PCA...",
        embeddings.shape[1],
        n_components,
    )
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = sum(pca.explained_variance_ratio_)
    logger.info("PCA: %.1f%% variance retained with %d components", explained * 100, n_components)
    return reduced, pca


# ── K Selection ────────────────────────────────────────────────────────────────

def sweep_k(
    embeddings: np.ndarray,
    k_min: int = GMM_K_MIN,
    k_max: int = GMM_K_MAX,
    step: int = 2,
) -> Dict[str, List]:
    """
    Sweep K values for GMM and compute BIC + silhouette scores.

    BIC (Bayesian Information Criterion):
    - Penalises model complexity. Lower is better.
    - Balances fit quality against number of parameters.

    Silhouette Score:
    - Computed on the hard assignment (argmax of probabilities).
    - Measures cluster cohesion vs separation. Range [-1, 1], higher is better.

    Returns dict with 'k_values', 'bic_scores', 'silhouette_scores'.
    """
    k_values = list(range(k_min, k_max + 1, step))
    bic_scores = []
    silhouette_scores_list = []

    logger.info("Sweeping K from %d to %d (step=%d)...", k_min, k_max, step)

    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="tied",
            random_state=42,
            n_init=3,
            max_iter=200,
        )
        gmm.fit(embeddings)

        bic = gmm.bic(embeddings)
        hard_labels = gmm.predict(embeddings)

        # Silhouette on a subsample for speed (full corpus is slow)
        n_sample = min(5000, len(embeddings))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embeddings), n_sample, replace=False)
        sil = silhouette_score(embeddings[idx], hard_labels[idx])

        bic_scores.append(bic)
        silhouette_scores_list.append(sil)

        logger.info("K=%d → BIC=%.0f, Silhouette=%.4f", k, bic, sil)

    return {
        "k_values": k_values,
        "bic_scores": bic_scores,
        "silhouette_scores": silhouette_scores_list,
    }


# ── Fitting the Final Model ───────────────────────────────────────────────────

def fit_gmm(embeddings: np.ndarray, n_clusters: int) -> GaussianMixture:
    """
    Fit the final GMM with the chosen K.

    covariance_type='tied': all clusters share one covariance matrix.
    This dramatically reduces parameters (from K*D*D to D*D) and is a
    reasonable assumption when clusters are roughly similar in shape —
    which holds for normalised sentence embeddings.

    n_init=5: run 5 random initialisations, keep the best (by log-likelihood).
    This reduces sensitivity to initialisation.
    """
    logger.info("Fitting GMM with K=%d...", n_clusters)
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="tied",
        random_state=42,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(embeddings)
    logger.info("GMM converged: %s", gmm.converged_)
    return gmm


def get_cluster_distributions(
    gmm: GaussianMixture, embeddings: np.ndarray
) -> np.ndarray:
    """
    Get the soft cluster assignment for each document.

    Returns:
        np.ndarray of shape (N, K) — each row sums to 1.0
        Entry [i, j] = P(document i belongs to cluster j)
    """
    return gmm.predict_proba(embeddings)


def save_gmm(gmm: GaussianMixture, pca: PCA, path=None):
    """Persist the fitted GMM and PCA to disk."""
    if path is None:
        path = GMM_MODEL_PATH
    data = {"gmm": gmm, "pca": pca}
    joblib.dump(data, path)
    logger.info("GMM + PCA saved to %s", path)


def load_gmm(path=None) -> Tuple[GaussianMixture, PCA]:
    """Load a previously fitted GMM and PCA from disk."""
    if path is None:
        path = GMM_MODEL_PATH
    data = joblib.load(path)
    logger.info("GMM + PCA loaded from %s", path)
    return data["gmm"], data["pca"]


# ── Analysis Helpers ───────────────────────────────────────────────────────────

def compute_entropy(distributions: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of each document's cluster distribution.
    High entropy = document is genuinely split between clusters (boundary case).
    Low entropy = confident single-cluster assignment.

    Returns array of shape (N,).
    """
    # Clip to avoid log(0)
    p = np.clip(distributions, 1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def get_boundary_documents(
    distributions: np.ndarray, top_n: int = 10
) -> np.ndarray:
    """
    Return indices of the top_n documents with highest entropy —
    these are the most ambiguous, sitting at cluster boundaries.
    """
    entropy = compute_entropy(distributions)
    return np.argsort(entropy)[-top_n:][::-1]


def cluster_label_matrix(
    distributions: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_labels: int = 20,
) -> np.ndarray:
    """
    Build a (n_clusters x n_labels) co-occurrence matrix.
    Entry [c, l] = sum of P(cluster=c) for all documents with label l.

    This is soft — a document half-assigned to cluster 3 and half to cluster 7
    contributes 0.5 to both rows for its label column.
    """
    matrix = np.zeros((n_clusters, n_labels))
    for i in range(len(labels)):
        label = labels[i]
        matrix[:, label] += distributions[i]
    return matrix
