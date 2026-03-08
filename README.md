# Semantic Search System — 20 Newsgroups

A lightweight semantic search system with fuzzy clustering and semantic caching over the 20 Newsgroups dataset (~18K documents, 20 categories).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Service                        │
│  POST /query  │  GET /cache/stats  │  DELETE /cache      │
├───────────────┴────────────────────┴────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Semantic     │◄──│  Sentence    │  all-MiniLM-L6-v2 │
│  │  Cache        │    │  Transformer │  384-dim          │
│  │  (in-memory)  │    └──────────────┘                   │
│  │  cluster-aware│                                       │
│  │  lookup       │    ┌──────────────┐                   │
│  └──────┬───────┘    │  GMM + PCA   │  30 clusters      │
│         │            │  (soft assign)│  fuzzy membership  │
│         ▼            └──────────────┘                   │
│  ┌──────────────┐                                       │
│  │  ChromaDB     │  17,731 documents                    │
│  │  (persistent) │  cosine similarity                   │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

## Components

| Component | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, fast on CPU, strong for short text |
| Vector DB | ChromaDB (persistent) | Lightweight, disk-backed, native Python |
| Clustering | GMM (K=30, tied covariance) | Soft assignment via `predict_proba()` |
| Cache | Custom (dict + numpy) | Cluster-aware lookup, zero external deps |
| Dimensionality reduction | PCA (50 components) | Makes GMM tractable on 384-dim embeddings |

## Quick Start

### 1. Set up environment

```bash
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run data pipeline

```bash
python -m scripts.prepare_data
```

This takes ~20 minutes on CPU. It:
- Downloads and cleans the 20 Newsgroups dataset (18,846 → 17,731 documents)
- Embeds all documents with sentence-transformers
- Sweeps K=10..30 for GMM and fits the final model (K=30)
- Ingests everything into ChromaDB with cluster metadata

### 3. Start the API server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 4. Test the endpoints

```bash
# Query (cache miss)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the best graphics card for gaming?"}'

# Query (cache hit — paraphrase)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which GPU should I buy for games?"}'

# Cache stats
curl http://localhost:8000/cache/stats

# Flush cache
curl -X DELETE http://localhost:8000/cache
```

## Docker

```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

Or with docker-compose:

```bash
docker-compose up --build
```

## Project Structure

```
├── src/
│   ├── config.py              # Centralised configuration
│   ├── data_preparation.py    # Part 1: corpus cleaning & embedding
│   ├── vector_store.py        # Part 1: ChromaDB operations
│   ├── clustering.py          # Part 2: GMM fuzzy clustering
│   ├── semantic_cache.py      # Part 3: custom semantic cache
│   ├── schemas.py             # Pydantic request/response models
│   └── api.py                 # Part 4: FastAPI service
├── scripts/
│   ├── prepare_data.py        # Full data pipeline
│   └── analyze_threshold.py   # Cache threshold analysis
├── notebooks/
│   └── analysis.ipynb         # Cluster & threshold visualizations
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Design Decisions

### Data Cleaning (Part 1)

- **Headers/footers/quotes removed**: Headers leak newsgroup names (model cheats), footers are signatures (noise), quotes duplicate parent content (inflates similarity).
- **Min 10 words**: Documents shorter than one sentence carry negligible semantic signal.
- **Exact deduplication**: Cross-posted messages bias clustering.
- **Result**: 18,846 → 17,731 documents (515 empty, 539 short, 61 duplicate).

### Fuzzy Clustering (Part 2)

- **GMM over K-Means**: GMM produces probability distributions, not hard assignments. A document about gun legislation gets membership in both "politics" and "firearms" clusters.
- **PCA to 50 dims**: GMM with tied covariance on 384 dims is expensive. PCA retains ~49% variance while making fitting tractable.
- **K=30 chosen via BIC**: BIC penalises complexity while rewarding fit. The BIC curve plateaus at K=28-30, suggesting the true semantic structure is finer-grained than the 20 label categories.
- **Tied covariance**: All clusters share one covariance matrix, reducing parameters by ~30× versus full covariance.

### Semantic Cache (Part 3)

- **Cluster-aware lookup**: Instead of comparing a query against ALL cached entries (O(N)), we identify the query's clusters via GMM and only compare against entries in those clusters (O(N/K)).
- **Similarity threshold = 0.75**: Threshold analysis showed precision=1.0 across all thresholds (cluster pruning eliminates false positives). Best F1=0.88 at θ=0.70, but θ=0.75 provides a safety margin. Configurable via `CACHE_THRESHOLD` env var.
- **No external dependencies**: The entire cache is a Python class with dict + numpy. Serialises to JSON + .npy for persistence.

### Threshold Analysis Results

| Threshold | Precision | Recall | F1 | FP Rate |
|---|---|---|---|---|
| 0.70 | 1.000 | 0.783 | 0.878 | 0.0 |
| 0.75 | 1.000 | 0.667 | 0.800 | 0.0 |
| 0.80 | 1.000 | 0.500 | 0.667 | 0.0 |
| 0.85 | 1.000 | 0.300 | 0.462 | 0.0 |
| 0.90 | 1.000 | 0.067 | 0.125 | 0.0 |
| 0.95 | 1.000 | 0.033 | 0.065 | 0.0 |

Key insight: precision is 1.0 at every threshold because the cluster-based pruning
is highly effective at eliminating unrelated queries before similarity comparison.
The only trade-off is recall — how many valid paraphrases get caught.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `CACHE_THRESHOLD` | `0.75` | Cosine similarity threshold for cache hits |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |

## API Endpoints

### POST /query
```json
// Request
{ "query": "natural language query" }

// Response (cache hit)
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "similar cached query",
  "similarity_score": 0.91,
  "result": [
    {
      "text_preview": "...",
      "similarity_score": 0.62,
      "newsgroup_label": "comp.graphics",
      "cluster_distribution_top3": [[3, 0.997], [18, 0.003], [14, 0.0001]]
    }
  ],
  "dominant_cluster": 3
}
```

### GET /cache/stats
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE /cache
```json
{
  "message": "Cache flushed successfully",
  "previous_entries": 42
}
```
