"""
Part 1 — Corpus Preparation & Embedding

Handles loading the 20 Newsgroups dataset, cleaning it, and generating
dense vector embeddings suitable for semantic search and clustering.

Design decisions are documented inline at each filtering step.
"""

import re
import logging
from typing import List, Tuple

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    MIN_TOKEN_COUNT,
)

logger = logging.getLogger(__name__)


# ── Data Loading & Cleaning ────────────────────────────────────────────────────

def load_raw_dataset() -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load the full 20 Newsgroups dataset with headers, footers, and quotes
    stripped by sklearn.

    Why remove headers/footers/quotes:
    - Headers contain the newsgroup name and sender metadata, which would let
      a model cheat on classification/clustering by learning surface patterns
      rather than semantic content.
    - Footers are typically email signatures — repeated boilerplate that adds
      noise to embeddings.
    - Quotes duplicate content from parent posts, inflating pairwise similarity
      between messages in the same thread.

    Returns:
        texts: raw document strings
        labels: integer label per document
        target_names: list of newsgroup category names
    """
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=False,
    )
    return dataset.data, np.array(dataset.target), dataset.target_names


def clean_text(text: str) -> str:
    """
    Normalise a single document's text.

    Steps:
    - Replace non-ASCII characters with space. The dataset contains email
      artifacts (=20, stranded control chars) that carry no semantic value
      and confuse tokenisers.
    - Collapse multiple whitespace/newlines into single spaces.
    - Strip leading/trailing whitespace.
    """
    # Remove non-ASCII — these are mostly MIME artifacts and control characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse whitespace (including \n, \t, multiple spaces)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_corpus(
    texts: List[str], labels: np.ndarray
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Apply all cleaning filters to the corpus. Returns cleaned texts, their
    corresponding labels, and the original indices (for traceability).

    Filtering decisions (each logged with counts):

    1. Empty/whitespace-only documents → discarded.
       After header/footer/quote removal, some posts are literally empty.

    2. Documents with fewer than MIN_TOKEN_COUNT words → discarded.
       Very short posts (e.g., "Me too", "Thanks") carry negligible semantic
       signal and degrade embedding quality. 10 words is the threshold —
       roughly one sentence, the minimum for a meaningful embedding.

    3. Exact-duplicate texts → only the first occurrence kept.
       Cross-posted messages and automated reposts create duplicates that
       would bias clustering toward those topics.
    """
    cleaned_texts = []
    cleaned_labels = []
    original_indices = []

    seen_texts = set()
    stats = {"total": len(texts), "empty": 0, "short": 0, "duplicate": 0, "kept": 0}

    for i, (text, label) in enumerate(zip(texts, labels)):
        text = clean_text(text)

        # Filter 1: empty after cleaning
        if not text:
            stats["empty"] += 1
            continue

        # Filter 2: too short to carry semantic signal
        word_count = len(text.split())
        if word_count < MIN_TOKEN_COUNT:
            stats["short"] += 1
            continue

        # Filter 3: exact duplicate (on normalised text)
        if text in seen_texts:
            stats["duplicate"] += 1
            continue
        seen_texts.add(text)

        cleaned_texts.append(text)
        cleaned_labels.append(label)
        original_indices.append(i)

    stats["kept"] = len(cleaned_texts)
    cleaned_labels = np.array(cleaned_labels)

    logger.info(
        "Corpus cleaning complete: %d total → %d kept "
        "(dropped %d empty, %d short, %d duplicate)",
        stats["total"],
        stats["kept"],
        stats["empty"],
        stats["short"],
        stats["duplicate"],
    )

    return cleaned_texts, cleaned_labels, original_indices


# ── Embedding ──────────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model.

    Model choice: all-MiniLM-L6-v2
    - 384-dimensional output — compact, fast cosine similarity
    - 6-layer MiniLM distilled from a larger model — good quality/speed tradeoff
    - Trained on 1B+ sentence pairs — strong semantic similarity out of the box
    - ~80MB — fits comfortably in CPU memory and Docker images

    For this corpus (short newsgroup posts averaging ~100 words), a model
    optimised for sentence-level semantics is a better fit than paragraph or
    document-level models.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def embed_documents(
    model: SentenceTransformer, texts: List[str], normalize: bool = True
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    Args:
        model: loaded SentenceTransformer
        texts: list of cleaned document strings
        normalize: if True, L2-normalise embeddings so cosine similarity
                   reduces to a simple dot product — faster at query time.

    Returns:
        embeddings: np.ndarray of shape (len(texts), 384), float32
    """
    logger.info("Embedding %d documents (batch_size=%d)...", len(texts), EMBEDDING_BATCH_SIZE)
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.array(embeddings, dtype=np.float32)


# ── Convenience ────────────────────────────────────────────────────────────────

def prepare_corpus() -> Tuple[List[str], np.ndarray, List[int], List[str], np.ndarray]:
    """
    Full pipeline: load → clean → embed.

    Returns:
        texts, labels, original_indices, target_names, embeddings
    """
    texts, labels, target_names = load_raw_dataset()
    texts, labels, original_indices = clean_corpus(texts, labels)
    model = load_embedding_model()
    embeddings = embed_documents(model, texts)
    return texts, labels, original_indices, list(target_names), embeddings
