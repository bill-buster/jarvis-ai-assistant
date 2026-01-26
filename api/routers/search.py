"""Semantic search API endpoints.

Provides endpoints for AI-powered semantic search across iMessage conversations
using sentence embeddings.

Uses the all-MiniLM-L6-v2 model for encoding messages and queries,
then finds matches by cosine similarity rather than keyword matching.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.dependencies import get_imessage_reader
from api.schemas import ErrorResponse, MessageResponse
from integrations.imessage import ChatDBReader
from jarvis.semantic_search import SearchFilters, SemanticSearcher

router = APIRouter(prefix="/search", tags=["search"])


class SemanticSearchRequest(BaseModel):
    """Request body for semantic search.

    Example:
        ```json
        {
            "query": "dinner plans this weekend",
            "limit": 20,
            "threshold": 0.3,
            "filters": {
                "sender": "+15551234567",
                "after": "2024-01-01T00:00:00Z"
            }
        }
        ```
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language search query",
        examples=["dinner plans", "meeting tomorrow", "project deadline"],
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0). Higher = more relevant results only.",
    )
    index_limit: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum messages to search through (higher = slower but more thorough)",
    )


class SemanticSearchFilters(BaseModel):
    """Optional filters for semantic search.

    Example:
        ```json
        {
            "sender": "+15551234567",
            "chat_id": "chat123",
            "after": "2024-01-01T00:00:00Z",
            "before": "2024-12-31T23:59:59Z",
            "has_attachments": true
        }
        ```
    """

    sender: str | None = Field(
        default=None,
        description="Filter by sender phone/email. Use 'me' for your own messages.",
        examples=["+15551234567", "me"],
    )
    chat_id: str | None = Field(
        default=None,
        description="Filter to a specific conversation",
        examples=["chat123456789"],
    )
    after: datetime | None = Field(
        default=None,
        description="Only messages after this date (ISO 8601)",
        examples=["2024-01-01T00:00:00Z"],
    )
    before: datetime | None = Field(
        default=None,
        description="Only messages before this date (ISO 8601)",
        examples=["2024-12-31T23:59:59Z"],
    )
    has_attachments: bool | None = Field(
        default=None,
        description="Filter by presence of attachments",
    )


class SemanticSearchRequestWithFilters(SemanticSearchRequest):
    """Request body for semantic search with filters.

    Example:
        ```json
        {
            "query": "dinner plans this weekend",
            "limit": 20,
            "threshold": 0.3,
            "filters": {
                "sender": "+15551234567",
                "after": "2024-01-01T00:00:00Z"
            }
        }
        ```
    """

    filters: SemanticSearchFilters | None = Field(
        default=None,
        description="Optional filters to narrow search scope",
    )


class SemanticSearchResultItem(BaseModel):
    """A single semantic search result with similarity score.

    Example:
        ```json
        {
            "message": {
                "id": 12345,
                "chat_id": "chat123",
                "sender": "+15551234567",
                "text": "Let's get dinner this Saturday!",
                "date": "2024-01-15T18:30:00Z",
                "is_from_me": false
            },
            "similarity": 0.87
        }
        ```
    """

    message: MessageResponse = Field(
        ...,
        description="The matching message",
    )
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0-1.0). Higher = more relevant.",
        examples=[0.87, 0.72, 0.55],
    )


class SemanticSearchResponse(BaseModel):
    """Response containing semantic search results.

    Example:
        ```json
        {
            "query": "dinner plans",
            "results": [...],
            "total_results": 15,
            "threshold_used": 0.3,
            "messages_searched": 1000
        }
        ```
    """

    query: str = Field(
        ...,
        description="The original search query",
    )
    results: list[SemanticSearchResultItem] = Field(
        ...,
        description="List of matching messages with similarity scores",
    )
    total_results: int = Field(
        ...,
        description="Number of results returned",
        ge=0,
    )
    threshold_used: float = Field(
        ...,
        description="The similarity threshold that was applied",
    )
    messages_searched: int = Field(
        ...,
        description="Number of messages that were searched",
        ge=0,
    )


class CacheStatsResponse(BaseModel):
    """Embedding cache statistics.

    Example:
        ```json
        {
            "embedding_count": 5000,
            "size_bytes": 7680000,
            "size_mb": 7.32
        }
        ```
    """

    embedding_count: int = Field(
        ...,
        description="Number of cached embeddings",
        ge=0,
    )
    size_bytes: int = Field(
        ...,
        description="Total size of cached embeddings in bytes",
        ge=0,
    )
    size_mb: float = Field(
        ...,
        description="Total size of cached embeddings in megabytes",
        ge=0,
    )


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    response_model_exclude_unset=True,
    summary="Semantic search across messages",
    response_description="Search results ranked by semantic similarity",
    responses={
        200: {
            "description": "Search completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "query": "dinner plans",
                        "results": [
                            {
                                "message": {
                                    "id": 12345,
                                    "chat_id": "chat123",
                                    "sender": "+15551234567",
                                    "sender_name": "John",
                                    "text": "Let's get dinner this Saturday!",
                                    "date": "2024-01-15T18:30:00Z",
                                    "is_from_me": False,
                                    "attachments": [],
                                    "reactions": [],
                                },
                                "similarity": 0.87,
                            }
                        ],
                        "total_results": 1,
                        "threshold_used": 0.3,
                        "messages_searched": 1000,
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
    },
)
def semantic_search(
    request: SemanticSearchRequestWithFilters,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> SemanticSearchResponse:
    """Search messages by semantic similarity.

    Uses AI-powered semantic search to find messages by meaning rather than
    exact keyword matching. This allows finding messages that are conceptually
    similar even if they don't contain the exact search terms.

    **How it works:**
    1. Your query is converted to a semantic embedding using all-MiniLM-L6-v2
    2. Messages are compared using cosine similarity
    3. Results are ranked by relevance score

    **Example Queries:**
    - "dinner plans" - finds messages about eating out, restaurants, food plans
    - "meeting tomorrow" - finds scheduling discussions, appointment confirmations
    - "running late" - finds messages about delays, being behind schedule

    **Performance Notes:**
    - First search may be slower as messages are indexed
    - Subsequent searches are faster due to embedding caching
    - Use `index_limit` to control search thoroughness vs speed

    **Filtering:**
    Use the `filters` object to narrow results by:
    - `sender`: Phone number, email, or "me" for your own messages
    - `chat_id`: Specific conversation only
    - `after`/`before`: Date range
    - `has_attachments`: Messages with/without attachments

    Args:
        request: Search query and options

    Returns:
        SemanticSearchResponse with ranked results and metadata

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    # Convert filters
    search_filters = None
    if request.filters:
        search_filters = SearchFilters(
            sender=request.filters.sender,
            chat_id=request.filters.chat_id,
            after=request.filters.after,
            before=request.filters.before,
            has_attachments=request.filters.has_attachments,
        )

    # Create searcher with custom threshold
    searcher = SemanticSearcher(
        reader=reader,
        similarity_threshold=request.threshold,
    )

    # Perform search
    results = searcher.search(
        query=request.query,
        filters=search_filters,
        limit=request.limit,
        index_limit=request.index_limit,
    )

    # Convert to response format
    result_items = [
        SemanticSearchResultItem(
            message=MessageResponse.model_validate(r.message),
            similarity=round(r.similarity, 4),
        )
        for r in results
    ]

    return SemanticSearchResponse(
        query=request.query,
        results=result_items,
        total_results=len(result_items),
        threshold_used=request.threshold,
        messages_searched=request.index_limit,
    )


@router.get(
    "/semantic/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get embedding cache statistics",
    response_description="Cache statistics including size and count",
)
def get_cache_stats() -> CacheStatsResponse:
    """Get statistics about the semantic search embedding cache.

    Returns information about the cached message embeddings including
    the number of cached embeddings and storage size.

    This is useful for monitoring cache growth and deciding when to clear it.

    Returns:
        CacheStatsResponse with cache statistics
    """
    from jarvis.semantic_search import EmbeddingCache

    cache = EmbeddingCache()
    stats = cache.stats()
    cache.close()

    return CacheStatsResponse(
        embedding_count=stats["embedding_count"],
        size_bytes=stats["size_bytes"],
        size_mb=stats["size_mb"],
    )


@router.delete(
    "/semantic/cache",
    summary="Clear embedding cache",
    response_description="Confirmation of cache clear",
)
def clear_cache() -> dict[str, str]:
    """Clear the semantic search embedding cache.

    This removes all cached message embeddings. The next semantic search
    will need to recompute embeddings, which will be slower but ensures
    fresh results.

    Use this if:
    - Embeddings seem stale or incorrect
    - You want to free up disk space
    - You've updated the embedding model

    Returns:
        Confirmation message
    """
    from jarvis.semantic_search import EmbeddingCache

    cache = EmbeddingCache()
    stats_before = cache.stats()
    cache.clear()
    cache.close()

    return {
        "status": "success",
        "message": f"Cleared {stats_before['embedding_count']} cached embeddings "
        f"({stats_before['size_mb']} MB)",
    }
