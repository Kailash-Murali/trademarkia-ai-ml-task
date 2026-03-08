"""
Pydantic models for request/response validation.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /query body."""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="Natural language query"
    )


# ── Response Models ────────────────────────────────────────────────────────────

class DocumentResult(BaseModel):
    """A single retrieved document."""

    text_preview: str
    similarity_score: float
    newsgroup_label: str
    cluster_distribution_top3: List[Tuple[int, float]]


class QueryResponse(BaseModel):
    """POST /query response."""

    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: List[DocumentResult]
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    """GET /cache/stats response."""

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class CacheFlushResponse(BaseModel):
    """DELETE /cache response."""

    message: str
    previous_entries: int
