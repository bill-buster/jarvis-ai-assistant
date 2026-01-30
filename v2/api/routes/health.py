"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..schemas import EmbeddingCacheStats, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API and system health."""
    from core.imessage import MessageReader
    from core.models import get_model_loader

    # Check iMessage access
    imessage_ok = False
    try:
        reader = MessageReader()
        imessage_ok = reader.check_access()
    except Exception as e:
        logger.debug(f"iMessage health check failed: {e}")

    # Check model status
    model_loaded = False
    try:
        loader = get_model_loader()
        model_loaded = loader.is_loaded
    except Exception as e:
        logger.debug(f"Model loader health check failed: {e}")

    return HealthResponse(
        status="ok",
        version="2.0.0",
        model_loaded=model_loaded,
        imessage_accessible=imessage_ok,
    )


@router.get("/health/cache", response_model=EmbeddingCacheStats)
async def cache_stats() -> EmbeddingCacheStats:
    """Get embedding cache statistics."""
    from core.embeddings import get_embedding_cache

    try:
        cache = get_embedding_cache()
        stats = cache.stats()
        return EmbeddingCacheStats(
            total_entries=stats.total_entries,
            hits=stats.hits,
            misses=stats.misses,
            hit_rate=stats.hit_rate,
        )
    except Exception as e:
        logger.debug(f"Cache stats check failed: {e}")
        return EmbeddingCacheStats(
            total_entries=0,
            hits=0,
            misses=0,
            hit_rate=0.0,
            error="Cache unavailable",
        )
