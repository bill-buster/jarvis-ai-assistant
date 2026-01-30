"""Semantic search endpoints for JARVIS v2."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class SearchResult(BaseModel):
    """A single search result."""
    message_id: int
    text: str
    sender: str
    is_from_me: bool
    timestamp: str | None
    chat_id: str
    chat_name: str | None
    similarity: float


class SearchResponse(BaseModel):
    """Search response."""
    results: list[SearchResult]
    query: str
    total: int


@router.get("/search", response_model=SearchResponse)
async def search_messages(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    chat_id: str | None = Query(None, description="Filter to specific chat"),
) -> SearchResponse:
    """Search messages using semantic similarity.

    Uses embeddings to find messages similar in meaning to the query,
    not just keyword matching.
    """
    try:
        from core.embeddings import get_embedding_store

        store = get_embedding_store()

        # Use semantic search
        results = store.find_similar(
            query=q,
            chat_id=chat_id,
            limit=limit,
            min_similarity=0.4,  # Lower threshold for broader results
        )

        return SearchResponse(
            results=[
                SearchResult(
                    message_id=r.message_id,
                    text=r.text,
                    sender=r.sender,
                    is_from_me=r.is_from_me,
                    timestamp=r.timestamp.isoformat() if r.timestamp else None,
                    chat_id=r.chat_id,
                    chat_name=r.chat_name,
                    similarity=r.similarity,
                )
                for r in results
            ],
            query=q,
            total=len(results),
        )

    except Exception:
        raise HTTPException(status_code=500, detail="Search failed")
