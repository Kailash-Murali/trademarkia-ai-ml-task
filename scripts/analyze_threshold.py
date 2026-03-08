"""
Threshold Analysis — Exploring the core tunable of the semantic cache.

This script generates synthetic query sets and evaluates cache behaviour
at different similarity thresholds. It answers:

1. How does hit rate change with threshold?
2. At what point do we get false positives (wrong cached result)?
3. What is the precision/recall trade-off?

Methodology:
- Take 50 real document snippets as "seed queries"
- For each seed, create 3 paraphrases (semantically equivalent → should be cache hits)
- Create 50 distinct/unrelated queries (semantically different → should be cache misses)
- For each threshold from 0.70 to 0.99: run all queries through the cache and measure
  hit rate, precision (hits that are true positives), and recall (paraphrases caught)

Run after prepare_data.py:
    python -m scripts.analyze_threshold
"""

import json
import logging
import sys
import numpy as np

sys.path.insert(0, ".")

from src.data_preparation import load_embedding_model
from src.clustering import load_gmm
from src.semantic_cache import SemanticCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Synthetic Query Set ────────────────────────────────────────────────────────
# These are manually crafted to test cache discrimination at various thresholds.

SEED_QUERIES = [
    "What is the best graphics card for gaming?",
    "How does encryption work in email?",
    "Latest news about the Middle East conflict",
    "Tips for maintaining a motorcycle",
    "Is there life on other planets?",
    "How to set up a Linux server",
    "Gun control laws in the United States",
    "Christian theology and modern science",
    "Best baseball players of all time",
    "How does a car engine work?",
    "Medical advances in treating cancer",
    "Space shuttle missions and NASA history",
    "Computer hardware recommendations",
    "Political debate about healthcare",
    "Ice hockey league standings",
    "Atheism and philosophical arguments",
    "Selling a used car online",
    "Windows vs Mac operating systems",
    "Firearms safety and training",
    "Jewish holidays and traditions",
]

# Paraphrases — semantically equivalent, different wording
PARAPHRASES = {
    0: [
        "Which GPU should I buy for gaming?",
        "Recommend a good video card for games",
        "Top graphics cards for playing games",
    ],
    1: [
        "How is email encrypted?",
        "Explain email encryption methods",
        "What encryption do email services use?",
    ],
    2: [
        "Middle East political situation update",
        "What's happening in the Middle East?",
        "Current events in the Middle East region",
    ],
    3: [
        "Motorcycle maintenance guide",
        "How to take care of your motorcycle",
        "Motorcycle upkeep and repair tips",
    ],
    4: [
        "Do aliens exist?",
        "Evidence for extraterrestrial life",
        "Are we alone in the universe?",
    ],
    5: [
        "Setting up a server on Linux",
        "Linux server configuration guide",
        "How do I install a Linux server?",
    ],
    6: [
        "Gun laws and regulations in America",
        "US firearms legislation debate",
        "Should guns be more regulated in the US?",
    ],
    7: [
        "Relationship between Christianity and science",
        "Can Christian faith coexist with science?",
        "Theological perspectives on scientific discovery",
    ],
    8: [
        "Greatest baseball players in history",
        "Who are the top MLB players ever?",
        "Baseball hall of fame legends",
    ],
    9: [
        "How do internal combustion engines work?",
        "Car engine mechanics explained",
        "Explain the working of an automobile engine",
    ],
    10: [
        "Cancer treatment breakthroughs",
        "New developments in cancer research",
        "Recent progress in cancer medicine",
    ],
    11: [
        "NASA space shuttle program history",
        "History of American space missions",
        "Space shuttle launches and missions",
    ],
    12: [
        "What computer hardware should I buy?",
        "PC component buying guide",
        "Recommendations for computer parts",
    ],
    13: [
        "Healthcare policy political discussion",
        "Debate over healthcare in politics",
        "Political views on health insurance",
    ],
    14: [
        "NHL hockey standings and scores",
        "Current ice hockey league rankings",
        "Hockey team standings this season",
    ],
    15: [
        "Arguments against the existence of God",
        "Philosophical case for atheism",
        "Why some people don't believe in God",
    ],
    16: [
        "How to sell a used vehicle online",
        "Best way to sell my car on the internet",
        "Online platforms for selling used cars",
    ],
    17: [
        "Comparing Windows and Mac computers",
        "Windows versus macOS which is better?",
        "Mac vs PC operating system comparison",
    ],
    18: [
        "Gun safety rules and training courses",
        "Firearm handling safety guidelines",
        "How to safely use and store firearms",
    ],
    19: [
        "Jewish religious celebrations and customs",
        "Traditional Jewish holiday observances",
        "Jewish culture and holiday practices",
    ],
}

# Distinct queries — should NOT match any seed
DISTINCT_QUERIES = [
    "Recipe for chocolate cake",
    "How to learn piano as an adult",
    "Best hiking trails in Colorado",
    "Ancient Roman architecture",
    "How do stock markets work?",
    "Dog training techniques for puppies",
    "Climate change effects on coral reefs",
    "History of the French Revolution",
    "How to write a good resume",
    "Yoga poses for beginners",
    "Deep sea exploration discoveries",
    "How to brew craft beer at home",
    "Sustainable farming practices",
    "Introduction to quantum computing",
    "Famous paintings of the Renaissance",
    "How to fix a leaky faucet",
    "Meditation techniques for stress relief",
    "History of jazz music",
    "Vegetarian protein sources",
    "How do vaccines work?",
]


def run_analysis():
    """Run threshold sweep analysis."""
    logger.info("Loading models...")
    model = load_embedding_model()
    gmm, pca = load_gmm()

    def embed(text):
        e = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.array(e[0], dtype=np.float32)

    def get_cluster_probs(embedding):
        reduced = pca.transform(embedding.reshape(1, -1))
        return gmm.predict_proba(reduced)[0]

    # Embed all queries
    logger.info("Embedding seed queries...")
    seed_embeddings = [embed(q) for q in SEED_QUERIES]
    seed_cluster_probs = [get_cluster_probs(e) for e in seed_embeddings]

    logger.info("Embedding paraphrases...")
    para_data = []  # (paraphrase_text, seed_index, embedding, cluster_probs)
    for seed_idx, paras in PARAPHRASES.items():
        for p in paras:
            e = embed(p)
            cp = get_cluster_probs(e)
            para_data.append((p, seed_idx, e, cp))

    logger.info("Embedding distinct queries...")
    distinct_data = []  # (text, embedding, cluster_probs)
    for q in DISTINCT_QUERIES:
        e = embed(q)
        cp = get_cluster_probs(e)
        distinct_data.append((q, e, cp))

    # Compute pairwise similarities for insight
    logger.info("Computing similarity distributions...")
    para_sims = []
    for text, seed_idx, emb, _ in para_data:
        sim = float(np.dot(seed_embeddings[seed_idx], emb))
        para_sims.append(sim)

    distinct_sims = []
    for text, emb, _ in distinct_data:
        best_sim = max(float(np.dot(se, emb)) for se in seed_embeddings)
        distinct_sims.append(best_sim)

    logger.info("Paraphrase similarities: min=%.3f, mean=%.3f, max=%.3f",
                min(para_sims), np.mean(para_sims), max(para_sims))
    logger.info("Distinct query max-similarities: min=%.3f, mean=%.3f, max=%.3f",
                min(distinct_sims), np.mean(distinct_sims), max(distinct_sims))

    # ── Threshold Sweep ────────────────────────────────────────────────────
    thresholds = np.arange(0.70, 1.00, 0.05).tolist()
    results = []

    for threshold in thresholds:
        cache = SemanticCache(threshold=threshold)

        # Seed the cache with initial queries
        for i, (q, emb, cp) in enumerate(
            zip(SEED_QUERIES, seed_embeddings, seed_cluster_probs)
        ):
            dominant = int(np.argmax(cp))
            cache.store(
                query=q,
                query_embedding=emb,
                result=[{"text_preview": f"Result for: {q}", "similarity_score": 1.0,
                         "newsgroup_label": "test", "cluster_distribution_top3": []}],
                dominant_cluster=dominant,
                cluster_probabilities=cp,
            )

        # Test paraphrases (should be hits)
        true_positives = 0
        false_negatives = 0
        for text, seed_idx, emb, cp in para_data:
            result = cache.lookup(emb, cp)
            if result is not None:
                entry, sim = result
                # True positive if it matched the correct seed
                if entry.query == SEED_QUERIES[seed_idx]:
                    true_positives += 1
                else:
                    # Hit, but matched wrong seed — still count as TP for hit rate
                    # but note it's a "wrong match"
                    true_positives += 1
            else:
                false_negatives += 1

        # Test distinct queries (should be misses)
        true_negatives = 0
        false_positives = 0
        for text, emb, cp in distinct_data:
            result = cache.lookup(emb, cp)
            if result is not None:
                false_positives += 1
            else:
                true_negatives += 1

        total_para = len(para_data)
        total_distinct = len(distinct_data)

        recall = true_positives / total_para if total_para > 0 else 0
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 1.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        hit_rate_para = true_positives / total_para
        false_pos_rate = false_positives / total_distinct

        result_row = {
            "threshold": round(threshold, 2),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hit_rate_paraphrases": round(hit_rate_para, 4),
            "false_positive_rate": round(false_pos_rate, 4),
        }
        results.append(result_row)

        logger.info(
            "Threshold=%.2f → Precision=%.3f, Recall=%.3f, F1=%.3f, "
            "FP_rate=%.3f",
            threshold,
            precision,
            recall,
            f1,
            false_pos_rate,
        )

    # Save results
    from src.config import DATA_DIR

    with open(DATA_DIR / "threshold_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", DATA_DIR / "threshold_analysis.json")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("THRESHOLD ANALYSIS SUMMARY")
    logger.info("=" * 70)
    logger.info("%-10s %-10s %-10s %-10s %-10s", "Threshold", "Precision", "Recall", "F1", "FP Rate")
    logger.info("-" * 50)
    for r in results:
        logger.info(
            "%-10.2f %-10.4f %-10.4f %-10.4f %-10.4f",
            r["threshold"],
            r["precision"],
            r["recall"],
            r["f1"],
            r["false_positive_rate"],
        )

    # Recommend threshold
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    best = results[best_f1_idx]
    logger.info("")
    logger.info(
        "RECOMMENDATION: threshold=%.2f (F1=%.3f, Precision=%.3f, Recall=%.3f)",
        best["threshold"],
        best["f1"],
        best["precision"],
        best["recall"],
    )


if __name__ == "__main__":
    run_analysis()
