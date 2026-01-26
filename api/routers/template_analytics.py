"""Template Analytics API endpoints.

Provides endpoints for monitoring template matching performance,
hit rates, and optimization opportunities.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from jarvis.metrics import get_template_analytics
from models.templates import _load_templates

router = APIRouter(prefix="/metrics/templates", tags=["template-analytics"])


class TemplateAnalyticsResponse(BaseModel):
    """Response model for template analytics summary."""

    total_queries: int
    template_hits: int
    model_fallbacks: int
    hit_rate_percent: float
    cache_hit_rate: float
    unique_templates_matched: int
    queries_per_second: float
    uptime_seconds: float


class TopTemplateItem(BaseModel):
    """A single template in the top templates list."""

    template_name: str
    match_count: int


class MissedQueryItem(BaseModel):
    """A single missed query entry."""

    query_hash: str
    similarity: float
    best_template: str | None
    timestamp: str


class CategoryAverageItem(BaseModel):
    """Average similarity for a template category."""

    category: str
    average_similarity: float


class TemplateInfo(BaseModel):
    """Information about a single template."""

    name: str
    pattern_count: int
    sample_patterns: list[str]


@router.get("", response_model=TemplateAnalyticsResponse)
def get_template_analytics_summary() -> TemplateAnalyticsResponse:
    """Get template analytics summary.

    Returns comprehensive statistics about template matching performance:
    - Total queries processed
    - Template hit rate (% of queries matched above 0.7 threshold)
    - Cache hit rate for query embeddings
    - Queries per second throughput
    """
    analytics = get_template_analytics()
    stats = analytics.get_stats()
    return TemplateAnalyticsResponse(**stats)


@router.get("/top")
def get_top_templates(limit: int = 20) -> list[TopTemplateItem]:
    """Get most frequently matched templates.

    Args:
        limit: Maximum number of templates to return (default: 20)

    Returns:
        List of top templates sorted by match count
    """
    analytics = get_template_analytics()
    top = analytics.get_top_templates(limit=limit)
    return [TopTemplateItem(**item) for item in top]


@router.get("/missed")
def get_missed_queries(limit: int = 50) -> list[MissedQueryItem]:
    """Get queries that fell through to model generation.

    These are optimization opportunities - queries that almost matched
    templates but fell below the 0.7 threshold.

    Args:
        limit: Maximum number of queries to return (default: 50)

    Returns:
        List of missed queries with similarity scores
    """
    analytics = get_template_analytics()
    missed = analytics.get_missed_queries(limit=limit)
    return [MissedQueryItem(**item) for item in missed]


@router.get("/categories")
def get_category_averages() -> list[CategoryAverageItem]:
    """Get average similarity scores per template category.

    Categories are extracted from template names (e.g., 'quick', 'summarize').

    Returns:
        List of categories with their average similarity scores
    """
    analytics = get_template_analytics()
    averages = analytics.get_category_averages()
    return [
        CategoryAverageItem(category=cat, average_similarity=round(avg, 4))
        for cat, avg in sorted(averages.items(), key=lambda x: x[1], reverse=True)
    ]


@router.get("/templates")
def list_available_templates() -> list[TemplateInfo]:
    """List all available templates.

    Returns information about all templates in the system including
    pattern counts and sample patterns.

    Returns:
        List of template information
    """
    templates = _load_templates()
    return [
        TemplateInfo(
            name=t.name,
            pattern_count=len(t.patterns),
            sample_patterns=t.patterns[:3],  # First 3 patterns as samples
        )
        for t in templates
    ]


@router.get("/coverage")
def get_template_coverage() -> dict[str, Any]:
    """Get overall template coverage statistics.

    Provides a summary of template system coverage including:
    - Template vs model-generated response breakdown
    - Coverage percentage
    - Total available templates

    Returns:
        Dictionary with coverage statistics
    """
    analytics = get_template_analytics()
    stats = analytics.get_stats()
    templates = _load_templates()

    total_patterns = sum(len(t.patterns) for t in templates)

    return {
        "total_templates": len(templates),
        "total_patterns": total_patterns,
        "responses_from_templates": stats["template_hits"],
        "responses_from_model": stats["model_fallbacks"],
        "coverage_percent": stats["hit_rate_percent"],
        "template_efficiency": (
            round(stats["template_hits"] / len(templates), 2)
            if len(templates) > 0 and stats["template_hits"] > 0
            else 0.0
        ),
    }


@router.get("/export")
def export_raw_analytics() -> JSONResponse:
    """Export raw analytics data as JSON for analysis.

    Returns complete raw data including:
    - All counters and timestamps
    - Complete template match history
    - All missed queries
    - Category similarity data

    Returns:
        JSON response with raw analytics data
    """
    analytics = get_template_analytics()
    raw_data = analytics.export_raw()
    return JSONResponse(
        content=raw_data,
        headers={"Content-Disposition": "attachment; filename=template_analytics.json"},
    )


@router.post("/reset")
def reset_template_analytics() -> dict[str, str]:
    """Reset all template analytics data.

    Clears all counters, matched template history, missed queries,
    and category statistics. Use this to start fresh analytics collection.

    Returns:
        Confirmation message
    """
    analytics = get_template_analytics()
    analytics.reset()
    return {"status": "ok", "message": "Template analytics reset successfully"}


@router.get("/dashboard")
def get_dashboard_data() -> dict[str, Any]:
    """Get all data needed for the template analytics dashboard.

    Combines multiple endpoints into a single response for efficient
    dashboard rendering.

    Returns:
        Dictionary with all dashboard data
    """
    analytics = get_template_analytics()
    stats = analytics.get_stats()
    templates = _load_templates()

    return {
        "summary": stats,
        "top_templates": analytics.get_top_templates(limit=20),
        "missed_queries": analytics.get_missed_queries(limit=20),
        "category_averages": [
            {"category": cat, "average_similarity": round(avg, 4)}
            for cat, avg in analytics.get_category_averages().items()
        ],
        "coverage": {
            "total_templates": len(templates),
            "total_patterns": sum(len(t.patterns) for t in templates),
            "responses_from_templates": stats["template_hits"],
            "responses_from_model": stats["model_fallbacks"],
            "coverage_percent": stats["hit_rate_percent"],
        },
        "pie_chart_data": {
            "template_responses": stats["template_hits"],
            "model_responses": stats["model_fallbacks"],
        },
    }
