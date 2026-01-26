"""Feedback API endpoints for tracking user interactions with suggestions.

Provides endpoints for:
- Recording feedback when users send, edit, or dismiss suggestions
- Retrieving aggregate feedback statistics
- Getting suggested improvements based on feedback patterns
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from jarvis.evaluation import (
    EvaluationResult,
    FeedbackAction,
    FeedbackEntry,
    get_feedback_store,
    get_response_evaluator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


# =============================================================================
# Pydantic Schemas
# =============================================================================


class EvaluationScores(BaseModel):
    """Evaluation scores for a response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tone_score": 0.85,
                "relevance_score": 0.78,
                "naturalness_score": 0.92,
                "length_score": 0.75,
                "overall_score": 0.82,
            }
        }
    )

    tone_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the response matches conversation tone (0-1)",
        examples=[0.85],
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity to recent context (0-1)",
        examples=[0.78],
    )
    naturalness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How natural the response sounds (0-1)",
        examples=[0.92],
    )
    length_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How appropriate the length is (0-1)",
        examples=[0.75],
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted average of all scores (0-1)",
        examples=[0.82],
    )


class RecordFeedbackRequest(BaseModel):
    """Request to record feedback on a suggestion.

    Example:
        ```json
        {
            "action": "sent",
            "suggestion_text": "Sounds great! What time works for you?",
            "chat_id": "chat123456789",
            "context_messages": ["Hey, want to grab lunch?", "Are you free tomorrow?"],
            "edited_text": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "sent",
                "suggestion_text": "Sounds great! What time works for you?",
                "chat_id": "chat123456789",
                "context_messages": [
                    "Hey, want to grab lunch?",
                    "Are you free tomorrow?",
                ],
                "edited_text": None,
            }
        }
    )

    action: str = Field(
        ...,
        description="Feedback action: 'sent', 'edited', 'dismissed', or 'copied'",
        examples=["sent", "edited", "dismissed", "copied"],
    )
    suggestion_text: str = Field(
        ...,
        min_length=1,
        description="The original suggestion text",
        examples=["Sounds great! What time works for you?"],
    )
    chat_id: str = Field(
        ...,
        min_length=1,
        description="Conversation ID where the suggestion was made",
        examples=["chat123456789"],
    )
    context_messages: list[str] = Field(
        default_factory=list,
        description="Recent messages from the conversation for context",
        examples=[["Hey, want to grab lunch?", "Are you free tomorrow?"]],
    )
    edited_text: str | None = Field(
        default=None,
        description="The edited text (required if action is 'edited')",
        examples=["Yes, sounds great! What time?"],
    )
    include_evaluation: bool = Field(
        default=True,
        description="Whether to compute and store evaluation scores",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata to store with feedback",
    )


class RecordFeedbackResponse(BaseModel):
    """Response after recording feedback.

    Example:
        ```json
        {
            "success": true,
            "suggestion_id": "a1b2c3d4e5f6",
            "evaluation": {
                "tone_score": 0.85,
                "relevance_score": 0.78,
                "naturalness_score": 0.92,
                "length_score": 0.75,
                "overall_score": 0.82
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "suggestion_id": "a1b2c3d4e5f6",
                "evaluation": {
                    "tone_score": 0.85,
                    "relevance_score": 0.78,
                    "naturalness_score": 0.92,
                    "length_score": 0.75,
                    "overall_score": 0.82,
                },
            }
        }
    )

    success: bool = Field(..., description="Whether feedback was recorded successfully")
    suggestion_id: str = Field(
        ...,
        description="Unique ID for the suggestion (hash of text)",
        examples=["a1b2c3d4e5f6"],
    )
    evaluation: EvaluationScores | None = Field(
        default=None,
        description="Evaluation scores if computed",
    )


class FeedbackStatsResponse(BaseModel):
    """Aggregate feedback statistics.

    Example:
        ```json
        {
            "total_feedback": 150,
            "sent_unchanged": 80,
            "edited": 45,
            "dismissed": 20,
            "copied": 5,
            "acceptance_rate": 0.55,
            "edit_rate": 0.31,
            "avg_evaluation_scores": {
                "tone_score": 0.82,
                "relevance_score": 0.75,
                "naturalness_score": 0.88,
                "length_score": 0.72,
                "overall_score": 0.79
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_feedback": 150,
                "sent_unchanged": 80,
                "edited": 45,
                "dismissed": 20,
                "copied": 5,
                "acceptance_rate": 0.55,
                "edit_rate": 0.31,
                "avg_evaluation_scores": {
                    "tone_score": 0.82,
                    "relevance_score": 0.75,
                    "naturalness_score": 0.88,
                    "length_score": 0.72,
                    "overall_score": 0.79,
                },
            }
        }
    )

    total_feedback: int = Field(
        ...,
        ge=0,
        description="Total number of feedback entries",
        examples=[150],
    )
    sent_unchanged: int = Field(
        ...,
        ge=0,
        description="Count of suggestions sent without edits (implicit positive)",
        examples=[80],
    )
    edited: int = Field(
        ...,
        ge=0,
        description="Count of suggestions that were edited before sending",
        examples=[45],
    )
    dismissed: int = Field(
        ...,
        ge=0,
        description="Count of suggestions that were dismissed (implicit negative)",
        examples=[20],
    )
    copied: int = Field(
        ...,
        ge=0,
        description="Count of suggestions that were copied but not sent yet",
        examples=[5],
    )
    acceptance_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of suggestions sent unchanged (sent / total actioned)",
        examples=[0.55],
    )
    edit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rate of suggestions edited before sending",
        examples=[0.31],
    )
    avg_evaluation_scores: dict[str, float] | None = Field(
        default=None,
        description="Average evaluation scores across all feedback with evaluations",
    )


class ImprovementSuggestion(BaseModel):
    """A suggested improvement based on feedback patterns.

    Example:
        ```json
        {
            "type": "length",
            "suggestion": "Generate shorter responses",
            "detail": "Users typically shorten suggestions by 30%",
            "confidence": 0.85
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "length",
                "suggestion": "Generate shorter responses",
                "detail": "Users typically shorten suggestions by 30%",
                "confidence": 0.85,
            }
        }
    )

    type: str = Field(
        ...,
        description="Type of improvement: 'length', 'tone', or 'vocabulary'",
        examples=["length", "tone", "vocabulary"],
    )
    suggestion: str = Field(
        ...,
        description="The improvement suggestion",
        examples=["Generate shorter responses"],
    )
    detail: str = Field(
        ...,
        description="Additional detail about the pattern",
        examples=["Users typically shorten suggestions by 30%"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this suggestion based on data",
        examples=[0.85],
    )


class ImprovementsResponse(BaseModel):
    """Response with suggested improvements.

    Example:
        ```json
        {
            "improvements": [
                {
                    "type": "length",
                    "suggestion": "Generate shorter responses",
                    "detail": "Users typically shorten suggestions by 30%",
                    "confidence": 0.85
                }
            ],
            "based_on_entries": 100
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "improvements": [
                    {
                        "type": "length",
                        "suggestion": "Generate shorter responses",
                        "detail": "Users typically shorten suggestions by 30%",
                        "confidence": 0.85,
                    },
                    {
                        "type": "tone",
                        "suggestion": "Use more casual language",
                        "detail": "Users often make responses less formal",
                        "confidence": 0.72,
                    },
                ],
                "based_on_entries": 100,
            }
        }
    )

    improvements: list[ImprovementSuggestion] = Field(
        ...,
        description="List of suggested improvements",
    )
    based_on_entries: int = Field(
        ...,
        ge=0,
        description="Number of feedback entries analyzed",
        examples=[100],
    )


class FeedbackEntryResponse(BaseModel):
    """A single feedback entry.

    Example:
        ```json
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "action": "edited",
            "suggestion_id": "a1b2c3d4e5f6",
            "suggestion_text": "Sounds great! What time?",
            "edited_text": "Yes! What time works?",
            "chat_id": "chat123456789"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "action": "edited",
                "suggestion_id": "a1b2c3d4e5f6",
                "suggestion_text": "Sounds great! What time?",
                "edited_text": "Yes! What time works?",
                "chat_id": "chat123456789",
                "evaluation": None,
            }
        }
    )

    timestamp: str = Field(
        ...,
        description="When the feedback was recorded (ISO format)",
        examples=["2024-01-15T10:30:00Z"],
    )
    action: str = Field(
        ...,
        description="The feedback action type",
        examples=["sent", "edited", "dismissed"],
    )
    suggestion_id: str = Field(
        ...,
        description="Unique ID for the suggestion",
        examples=["a1b2c3d4e5f6"],
    )
    suggestion_text: str = Field(
        ...,
        description="The original suggestion text",
    )
    edited_text: str | None = Field(
        default=None,
        description="The edited text if action was 'edited'",
    )
    chat_id: str = Field(
        ...,
        description="Conversation ID",
    )
    evaluation: EvaluationScores | None = Field(
        default=None,
        description="Evaluation scores if available",
    )


class RecentFeedbackResponse(BaseModel):
    """Response with recent feedback entries."""

    entries: list[FeedbackEntryResponse] = Field(
        ...,
        description="List of recent feedback entries",
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of feedback entries in store",
    )


class EvaluateResponseRequest(BaseModel):
    """Request to evaluate a response.

    Example:
        ```json
        {
            "response": "Sounds great! What time works for you?",
            "context_messages": ["Hey, want to grab lunch?", "Are you free tomorrow?"],
            "user_messages": ["Sure!", "Okay", "Thanks!"]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Sounds great! What time works for you?",
                "context_messages": [
                    "Hey, want to grab lunch?",
                    "Are you free tomorrow?",
                ],
                "user_messages": ["Sure!", "Okay", "Thanks!"],
            }
        }
    )

    response: str = Field(
        ...,
        min_length=1,
        description="The response to evaluate",
    )
    context_messages: list[str] = Field(
        ...,
        min_length=1,
        description="Recent messages from the conversation",
    )
    user_messages: list[str] | None = Field(
        default=None,
        description="User's own messages (for length comparison)",
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/response",
    response_model=RecordFeedbackResponse,
    responses={
        200: {"description": "Feedback recorded successfully"},
        400: {"description": "Invalid action type"},
    },
)
def record_feedback(request: RecordFeedbackRequest) -> RecordFeedbackResponse:
    """Record feedback when a user interacts with a suggestion.

    Use this endpoint to track:
    - **sent**: User sent the suggestion unchanged (implicit positive signal)
    - **edited**: User edited the suggestion before sending (capture for learning)
    - **dismissed**: User dismissed the suggestion (implicit negative signal)
    - **copied**: User copied the suggestion (partial positive signal)

    The endpoint optionally computes evaluation scores for the suggestion
    based on tone consistency, relevance, naturalness, and length appropriateness.
    """
    # Validate action
    try:
        action = FeedbackAction(request.action)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {request.action}. Must be one of: sent, edited, dismissed, copied",
        )

    # Validate edited text for edit actions
    if action == FeedbackAction.EDITED and not request.edited_text:
        raise HTTPException(
            status_code=400,
            detail="edited_text is required when action is 'edited'",
        )

    # Compute evaluation if requested
    evaluation: EvaluationResult | None = None
    if request.include_evaluation and request.context_messages:
        try:
            evaluator = get_response_evaluator()
            evaluation = evaluator.evaluate(
                response=request.suggestion_text,
                context_messages=request.context_messages,
                user_messages=None,  # Could be added to request if needed
            )
        except Exception:
            logger.exception("Error computing evaluation")

    # Record feedback
    store = get_feedback_store()
    entry = store.record_feedback(
        action=action,
        suggestion_text=request.suggestion_text,
        chat_id=request.chat_id,
        context_messages=request.context_messages,
        edited_text=request.edited_text,
        evaluation=evaluation,
        metadata=request.metadata,
    )

    # Build response
    eval_response = None
    if evaluation:
        eval_response = EvaluationScores(
            tone_score=evaluation.tone_score,
            relevance_score=evaluation.relevance_score,
            naturalness_score=evaluation.naturalness_score,
            length_score=evaluation.length_score,
            overall_score=evaluation.overall_score,
        )

    return RecordFeedbackResponse(
        success=True,
        suggestion_id=entry.suggestion_id,
        evaluation=eval_response,
    )


@router.get(
    "/stats",
    response_model=FeedbackStatsResponse,
    responses={
        200: {"description": "Feedback statistics retrieved successfully"},
    },
)
def get_feedback_stats() -> FeedbackStatsResponse:
    """Get aggregate feedback statistics.

    Returns metrics including:
    - Total feedback count by action type
    - Acceptance rate (suggestions sent unchanged)
    - Edit rate (suggestions modified before sending)
    - Average evaluation scores across all feedback

    Use these metrics to understand how well AI suggestions are meeting user needs.
    """
    store = get_feedback_store()
    stats = store.get_stats()

    # Convert avg_evaluation_scores dict to proper format
    avg_scores = stats.get("avg_evaluation_scores")
    if avg_scores:
        avg_scores = {
            "tone_score": round(avg_scores["tone_score"], 3),
            "relevance_score": round(avg_scores["relevance_score"], 3),
            "naturalness_score": round(avg_scores["naturalness_score"], 3),
            "length_score": round(avg_scores["length_score"], 3),
            "overall_score": round(avg_scores["overall_score"], 3),
        }

    return FeedbackStatsResponse(
        total_feedback=stats["total_feedback"],
        sent_unchanged=stats["sent_unchanged"],
        edited=stats["edited"],
        dismissed=stats["dismissed"],
        copied=stats["copied"],
        acceptance_rate=stats["acceptance_rate"],
        edit_rate=stats["edit_rate"],
        avg_evaluation_scores=avg_scores,
    )


@router.get(
    "/improvements",
    response_model=ImprovementsResponse,
    responses={
        200: {"description": "Improvement suggestions retrieved successfully"},
    },
)
def get_improvements(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of improvement suggestions to return",
    ),
) -> ImprovementsResponse:
    """Get suggested improvements based on feedback patterns.

    Analyzes patterns in user edits to identify areas for improvement:
    - **Length**: Are users making responses shorter or longer?
    - **Tone**: Are users adjusting formality levels?
    - **Vocabulary**: What words do users frequently add or remove?

    Use these insights to tune prompt engineering or model parameters.
    """
    store = get_feedback_store()
    improvements = store.get_improvements(limit=limit)
    stats = store.get_stats()

    improvement_list = [
        ImprovementSuggestion(
            type=imp["type"],
            suggestion=imp["suggestion"],
            detail=imp["detail"],
            confidence=imp["confidence"],
        )
        for imp in improvements
    ]

    return ImprovementsResponse(
        improvements=improvement_list,
        based_on_entries=stats["edited"],
    )


@router.get(
    "/recent",
    response_model=RecentFeedbackResponse,
    responses={
        200: {"description": "Recent feedback entries retrieved successfully"},
    },
)
def get_recent_feedback(
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of entries to return",
    ),
) -> RecentFeedbackResponse:
    """Get recent feedback entries.

    Returns the most recent feedback entries in reverse chronological order.
    Useful for debugging and monitoring feedback patterns in real-time.
    """
    store = get_feedback_store()
    entries = store.get_recent_entries(limit=limit)
    stats = store.get_stats()

    def entry_to_response(entry: FeedbackEntry) -> FeedbackEntryResponse:
        eval_scores = None
        if entry.evaluation:
            eval_scores = EvaluationScores(
                tone_score=entry.evaluation.tone_score,
                relevance_score=entry.evaluation.relevance_score,
                naturalness_score=entry.evaluation.naturalness_score,
                length_score=entry.evaluation.length_score,
                overall_score=entry.evaluation.overall_score,
            )

        return FeedbackEntryResponse(
            timestamp=entry.timestamp.isoformat(),
            action=entry.action.value,
            suggestion_id=entry.suggestion_id,
            suggestion_text=entry.suggestion_text,
            edited_text=entry.edited_text,
            chat_id=entry.chat_id,
            evaluation=eval_scores,
        )

    return RecentFeedbackResponse(
        entries=[entry_to_response(e) for e in entries],
        total_count=stats["total_feedback"],
    )


@router.post(
    "/evaluate",
    response_model=EvaluationScores,
    responses={
        200: {"description": "Response evaluated successfully"},
    },
)
def evaluate_response(request: EvaluateResponseRequest) -> EvaluationScores:
    """Evaluate a response against quality metrics.

    Computes scores for:
    - **Tone**: How well the response matches the conversation's historical tone
    - **Relevance**: Semantic similarity between response and recent context
    - **Naturalness**: How natural the response sounds (perplexity-based)
    - **Length**: How appropriate the length is compared to user's typical messages

    Use this endpoint to preview evaluation scores before recording feedback,
    or to evaluate responses from other sources.
    """
    evaluator = get_response_evaluator()
    result = evaluator.evaluate(
        response=request.response,
        context_messages=request.context_messages,
        user_messages=request.user_messages,
    )

    return EvaluationScores(
        tone_score=result.tone_score,
        relevance_score=result.relevance_score,
        naturalness_score=result.naturalness_score,
        length_score=result.length_score,
        overall_score=result.overall_score,
    )
