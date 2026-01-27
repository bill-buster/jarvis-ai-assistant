"""Quality Metrics API endpoints.

Provides endpoints for the quality metrics dashboard:
- GET /quality/summary - overall quality metrics
- GET /quality/trends - metrics over time
- GET /quality/contact/{id} - per-contact quality
- GET /quality/recommendations - suggested improvements
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field

from jarvis.intent import IntentType
from jarvis.quality_metrics import (
    AcceptanceStatus,
    ConversationType,
    ResponseSource,
    get_quality_metrics,
)

router = APIRouter(prefix="/quality", tags=["quality"])


# Response Models


class QualitySummaryResponse(BaseModel):
    """Overall quality metrics summary."""

    total_responses: int
    template_responses: int
    model_responses: int
    template_hit_rate_percent: float
    model_fallback_rate_percent: float
    avg_hhem_score: float | None
    hhem_score_count: int
    acceptance_rate_percent: float
    accepted_unchanged_count: int
    accepted_modified_count: int
    rejected_count: int
    avg_edit_distance: float | None
    avg_template_latency_ms: float
    avg_model_latency_ms: float
    uptime_seconds: float
    responses_per_second: float


class TrendDataPoint(BaseModel):
    """A single data point in the trends timeline."""

    date: str
    template_hit_rate_percent: float
    model_hit_rate_percent: float
    avg_hhem_score: float | None
    acceptance_rate_percent: float
    avg_edit_distance: float | None
    avg_template_latency_ms: float
    avg_model_latency_ms: float
    total_responses: int


class ContactQualityResponse(BaseModel):
    """Quality metrics for a specific contact."""

    contact_id: str
    total_responses: int
    template_responses: int
    model_responses: int
    acceptance_rate: float
    avg_hhem_score: float | None
    avg_edit_distance: float | None
    avg_latency_ms: float


class TimeOfDayQualityResponse(BaseModel):
    """Quality metrics by hour of day."""

    hour: int
    total_responses: int
    template_hit_rate: float
    acceptance_rate: float
    avg_latency_ms: float


class IntentQualityResponse(BaseModel):
    """Quality metrics by intent type."""

    intent: str
    total_responses: int
    template_hit_rate: float
    acceptance_rate: float
    avg_hhem_score: float | None
    avg_latency_ms: float


class ConversationTypeQualityResponse(BaseModel):
    """Quality metrics by conversation type."""

    one_on_one: dict[str, Any]
    group: dict[str, Any]


class RecommendationResponse(BaseModel):
    """Actionable recommendation for improving quality."""

    category: str
    priority: str
    title: str
    description: str
    metric_value: float | None = None
    target_value: float | None = None


class RecordResponseRequest(BaseModel):
    """Request to record a response event."""

    source: str = Field(..., description="Source: 'template' or 'model'")
    intent: str = Field(..., description="Intent type: 'reply', 'summarize', etc.")
    contact_id: str = Field(..., description="Contact or conversation ID")
    conversation_type: str = Field(..., description="Type: '1:1' or 'group'")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    hhem_score: float | None = Field(None, description="HHEM score for model responses")


class RecordAcceptanceRequest(BaseModel):
    """Request to record user acceptance of a suggestion."""

    contact_id: str = Field(..., description="Contact or conversation ID")
    status: str = Field(
        ..., description="Status: 'accepted_unchanged', 'accepted_modified', 'rejected'"
    )
    edit_distance: int | None = Field(None, description="Edit distance if modified")


class QualityDashboardResponse(BaseModel):
    """All data needed for the quality dashboard in a single response."""

    summary: QualitySummaryResponse
    trends: list[TrendDataPoint]
    top_contacts: list[ContactQualityResponse]
    time_of_day: list[TimeOfDayQualityResponse]
    by_intent: list[IntentQualityResponse]
    by_conversation_type: dict[str, Any]
    recommendations: list[RecommendationResponse]


# Helper functions


def _parse_response_source(source: str) -> ResponseSource:
    """Parse response source from string."""
    source_lower = source.lower()
    if source_lower == "template":
        return ResponseSource.TEMPLATE
    elif source_lower == "model":
        return ResponseSource.MODEL
    else:
        raise ValueError(f"Invalid source: {source}")


def _parse_intent(intent: str) -> IntentType:
    """Parse intent type from string."""
    intent_lower = intent.lower()
    for intent_type in IntentType:
        if intent_type.value == intent_lower:
            return intent_type
    raise ValueError(f"Invalid intent: {intent}")


def _parse_conversation_type(conv_type: str) -> ConversationType:
    """Parse conversation type from string."""
    if conv_type in ("1:1", "one_on_one"):
        return ConversationType.ONE_ON_ONE
    elif conv_type in ("group",):
        return ConversationType.GROUP
    else:
        raise ValueError(f"Invalid conversation type: {conv_type}")


def _parse_acceptance_status(status: str) -> AcceptanceStatus:
    """Parse acceptance status from string."""
    status_lower = status.lower()
    if status_lower == "accepted_unchanged":
        return AcceptanceStatus.ACCEPTED_UNCHANGED
    elif status_lower == "accepted_modified":
        return AcceptanceStatus.ACCEPTED_MODIFIED
    elif status_lower == "rejected":
        return AcceptanceStatus.REJECTED
    else:
        raise ValueError(f"Invalid acceptance status: {status}")


# Endpoints


@router.get("/summary", response_model=QualitySummaryResponse)
def get_quality_summary() -> QualitySummaryResponse:
    """Get overall quality metrics summary.

    Returns comprehensive statistics about response generation performance:
    - Template hit rate (% of queries matched by templates)
    - HHEM scores for model-generated responses
    - User acceptance rate
    - Response latencies
    """
    metrics = get_quality_metrics()
    summary = metrics.get_summary()
    return QualitySummaryResponse(**summary)


@router.get("/trends", response_model=list[TrendDataPoint])
def get_quality_trends(
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history"),
) -> list[TrendDataPoint]:
    """Get quality metrics over time.

    Returns daily snapshots of quality metrics for trend analysis.

    Args:
        days: Number of days of history to return (default: 30, max: 365)

    Returns:
        List of daily quality snapshots
    """
    metrics = get_quality_metrics()
    trends = metrics.get_trends(days=days)
    return [TrendDataPoint(**t) for t in trends]


@router.get("/contact/{contact_id}", response_model=ContactQualityResponse)
def get_contact_quality(
    contact_id: str = Path(..., description="Contact or conversation ID"),
) -> ContactQualityResponse:
    """Get quality metrics for a specific contact.

    Returns quality metrics aggregated for interactions with a specific contact.

    Args:
        contact_id: ID of the contact or conversation

    Returns:
        Quality metrics for the specified contact
    """
    metrics = get_quality_metrics()
    contacts = metrics.get_contact_quality(contact_id=contact_id)

    if not contacts:
        # Return empty metrics for unknown contact
        return ContactQualityResponse(
            contact_id=contact_id,
            total_responses=0,
            template_responses=0,
            model_responses=0,
            acceptance_rate=0.0,
            avg_hhem_score=None,
            avg_edit_distance=None,
            avg_latency_ms=0.0,
        )

    contact = contacts[0]
    return ContactQualityResponse(
        contact_id=contact.contact_id,
        total_responses=contact.total_responses,
        template_responses=contact.template_responses,
        model_responses=contact.model_responses,
        acceptance_rate=contact.acceptance_rate,
        avg_hhem_score=contact.avg_hhem_score,
        avg_edit_distance=contact.avg_edit_distance,
        avg_latency_ms=contact.avg_latency_ms,
    )


@router.get("/contacts", response_model=list[ContactQualityResponse])
def get_all_contacts_quality(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum contacts to return"),
) -> list[ContactQualityResponse]:
    """Get quality metrics for all contacts.

    Returns quality metrics aggregated by contact, sorted by activity.

    Args:
        limit: Maximum number of contacts to return (default: 20)

    Returns:
        List of quality metrics per contact
    """
    metrics = get_quality_metrics()
    contacts = metrics.get_contact_quality()[:limit]

    return [
        ContactQualityResponse(
            contact_id=c.contact_id,
            total_responses=c.total_responses,
            template_responses=c.template_responses,
            model_responses=c.model_responses,
            acceptance_rate=c.acceptance_rate,
            avg_hhem_score=c.avg_hhem_score,
            avg_edit_distance=c.avg_edit_distance,
            avg_latency_ms=c.avg_latency_ms,
        )
        for c in contacts
    ]


@router.get("/time-of-day", response_model=list[TimeOfDayQualityResponse])
def get_time_of_day_quality() -> list[TimeOfDayQualityResponse]:
    """Get quality metrics aggregated by hour of day.

    Returns quality metrics for each hour (0-23) to identify patterns
    in response quality throughout the day.

    Returns:
        List of 24 entries, one for each hour
    """
    metrics = get_quality_metrics()
    time_data = metrics.get_time_of_day_quality()

    return [
        TimeOfDayQualityResponse(
            hour=t.hour,
            total_responses=t.total_responses,
            template_hit_rate=t.template_hit_rate,
            acceptance_rate=t.acceptance_rate,
            avg_latency_ms=t.avg_latency_ms,
        )
        for t in time_data
    ]


@router.get("/by-intent", response_model=list[IntentQualityResponse])
def get_intent_quality() -> list[IntentQualityResponse]:
    """Get quality metrics aggregated by intent type.

    Returns quality metrics for each intent type (reply, summarize, search, etc.)
    to identify which intents have better/worse quality.

    Returns:
        List of quality metrics per intent type
    """
    metrics = get_quality_metrics()
    intent_data = metrics.get_intent_quality()

    return [
        IntentQualityResponse(
            intent=i.intent.value,
            total_responses=i.total_responses,
            template_hit_rate=i.template_hit_rate,
            acceptance_rate=i.acceptance_rate,
            avg_hhem_score=i.avg_hhem_score,
            avg_latency_ms=i.avg_latency_ms,
        )
        for i in intent_data
    ]


@router.get("/by-conversation-type")
def get_conversation_type_quality() -> dict[str, Any]:
    """Get quality metrics by conversation type.

    Returns quality metrics comparing 1:1 and group conversations.

    Returns:
        Dictionary with metrics for each conversation type
    """
    metrics = get_quality_metrics()
    return metrics.get_conversation_type_quality()


@router.get("/recommendations", response_model=list[RecommendationResponse])
def get_recommendations() -> list[RecommendationResponse]:
    """Get actionable recommendations for improving quality.

    Analyzes current metrics and generates prioritized suggestions
    for improving response generation quality.

    Returns:
        List of recommendations sorted by priority
    """
    metrics = get_quality_metrics()
    recommendations = metrics.get_recommendations()

    return [
        RecommendationResponse(
            category=r.category,
            priority=r.priority,
            title=r.title,
            description=r.description,
            metric_value=r.metric_value,
            target_value=r.target_value,
        )
        for r in recommendations
    ]


@router.get("/dashboard", response_model=QualityDashboardResponse)
def get_dashboard_data(
    trend_days: int = Query(default=7, ge=1, le=30, description="Days of trend data"),
    top_contacts_limit: int = Query(default=10, ge=1, le=50, description="Number of top contacts"),
) -> QualityDashboardResponse:
    """Get all data needed for the quality dashboard.

    Combines multiple endpoints into a single response for efficient
    dashboard rendering.

    Args:
        trend_days: Number of days of trend data (default: 7)
        top_contacts_limit: Number of top contacts to include (default: 10)

    Returns:
        Complete dashboard data in a single response
    """
    metrics = get_quality_metrics()

    # Get summary
    summary = metrics.get_summary()

    # Get trends
    trends = metrics.get_trends(days=trend_days)

    # Get top contacts
    contacts = metrics.get_contact_quality()[:top_contacts_limit]

    # Get time of day data
    time_data = metrics.get_time_of_day_quality()

    # Get intent data
    intent_data = metrics.get_intent_quality()

    # Get conversation type data
    conv_type_data = metrics.get_conversation_type_quality()

    # Get recommendations
    recommendations = metrics.get_recommendations()

    return QualityDashboardResponse(
        summary=QualitySummaryResponse(**summary),
        trends=[TrendDataPoint(**t) for t in trends],
        top_contacts=[
            ContactQualityResponse(
                contact_id=c.contact_id,
                total_responses=c.total_responses,
                template_responses=c.template_responses,
                model_responses=c.model_responses,
                acceptance_rate=c.acceptance_rate,
                avg_hhem_score=c.avg_hhem_score,
                avg_edit_distance=c.avg_edit_distance,
                avg_latency_ms=c.avg_latency_ms,
            )
            for c in contacts
        ],
        time_of_day=[
            TimeOfDayQualityResponse(
                hour=t.hour,
                total_responses=t.total_responses,
                template_hit_rate=t.template_hit_rate,
                acceptance_rate=t.acceptance_rate,
                avg_latency_ms=t.avg_latency_ms,
            )
            for t in time_data
        ],
        by_intent=[
            IntentQualityResponse(
                intent=i.intent.value,
                total_responses=i.total_responses,
                template_hit_rate=i.template_hit_rate,
                acceptance_rate=i.acceptance_rate,
                avg_hhem_score=i.avg_hhem_score,
                avg_latency_ms=i.avg_latency_ms,
            )
            for i in intent_data
        ],
        by_conversation_type=conv_type_data,
        recommendations=[
            RecommendationResponse(
                category=r.category,
                priority=r.priority,
                title=r.title,
                description=r.description,
                metric_value=r.metric_value,
                target_value=r.target_value,
            )
            for r in recommendations
        ],
    )


@router.post("/record/response")
def record_response(request: RecordResponseRequest) -> dict[str, str]:
    """Record a response generation event.

    Used to track template vs model usage, latency, and HHEM scores.

    Args:
        request: Response event details

    Returns:
        Confirmation message
    """
    metrics = get_quality_metrics()

    try:
        source = _parse_response_source(request.source)
        intent = _parse_intent(request.intent)
        conv_type = _parse_conversation_type(request.conversation_type)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    metrics.record_response(
        source=source,
        intent=intent,
        contact_id=request.contact_id,
        conversation_type=conv_type,
        latency_ms=request.latency_ms,
        hhem_score=request.hhem_score,
    )

    return {"status": "ok", "message": "Response recorded"}


@router.post("/record/acceptance")
def record_acceptance(request: RecordAcceptanceRequest) -> dict[str, str]:
    """Record user acceptance of a suggestion.

    Used to track acceptance rate and edit distance.

    Args:
        request: Acceptance event details

    Returns:
        Confirmation message
    """
    metrics = get_quality_metrics()

    try:
        status = _parse_acceptance_status(request.status)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    metrics.record_acceptance(
        contact_id=request.contact_id,
        status=status,
        edit_distance=request.edit_distance,
    )

    return {"status": "ok", "message": "Acceptance recorded"}


@router.post("/reset")
def reset_quality_metrics() -> dict[str, str]:
    """Reset all quality metrics data.

    Clears all recorded events, snapshots, and counters.
    Use this to start fresh quality tracking.

    Returns:
        Confirmation message
    """
    metrics = get_quality_metrics()
    metrics.reset()
    return {"status": "ok", "message": "Quality metrics reset successfully"}
