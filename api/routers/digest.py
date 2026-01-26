"""Digest API endpoints.

Provides endpoints for generating daily/weekly digests and managing digest preferences.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_imessage_reader
from api.schemas import (
    ActionItemResponse,
    DigestExportRequest,
    DigestExportResponse,
    DigestFormatEnum,
    DigestGenerateRequest,
    DigestPeriodEnum,
    DigestPreferencesResponse,
    DigestPreferencesUpdateRequest,
    DigestResponse,
    GroupHighlightResponse,
    MessageStatsResponse,
    UnansweredConversationResponse,
)
from integrations.imessage import ChatDBReader
from jarvis.config import get_config, save_config
from jarvis.digest import (
    Digest,
    DigestFormat,
    DigestGenerator,
    DigestPeriod,
    export_digest_html,
    export_digest_markdown,
    get_digest_filename,
)

router = APIRouter(prefix="/digest", tags=["digest"])


def _digest_to_response(digest: Digest) -> DigestResponse:
    """Convert internal Digest to API response model."""
    return DigestResponse(
        period=DigestPeriodEnum(digest.period.value),
        generated_at=digest.generated_at,
        start_date=digest.start_date,
        end_date=digest.end_date,
        needs_attention=[
            UnansweredConversationResponse(
                chat_id=c.chat_id,
                display_name=c.display_name,
                participants=c.participants,
                unanswered_count=c.unanswered_count,
                last_message_date=c.last_message_date,
                last_message_preview=c.last_message_preview,
                is_group=c.is_group,
            )
            for c in digest.needs_attention
        ],
        highlights=[
            GroupHighlightResponse(
                chat_id=h.chat_id,
                display_name=h.display_name,
                participants=h.participants,
                message_count=h.message_count,
                active_participants=h.active_participants,
                top_topics=h.top_topics,
                last_activity=h.last_activity,
            )
            for h in digest.highlights
        ],
        action_items=[
            ActionItemResponse(
                text=a.text,
                chat_id=a.chat_id,
                conversation_name=a.conversation_name,
                sender=a.sender,
                date=a.date,
                message_id=a.message_id,
                item_type=a.item_type,
            )
            for a in digest.action_items
        ],
        stats=MessageStatsResponse(
            total_sent=digest.stats.total_sent,
            total_received=digest.stats.total_received,
            total_messages=digest.stats.total_messages,
            active_conversations=digest.stats.active_conversations,
            most_active_conversation=digest.stats.most_active_conversation,
            most_active_count=digest.stats.most_active_count,
            avg_messages_per_day=digest.stats.avg_messages_per_day,
            busiest_hour=digest.stats.busiest_hour,
            hourly_distribution={str(k): v for k, v in digest.stats.hourly_distribution.items()},
        ),
    )


@router.post("/generate", response_model=DigestResponse)
def generate_digest(
    request: DigestGenerateRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DigestResponse:
    """Generate a digest for the specified period.

    Generates a comprehensive digest including:
    - Conversations with unanswered messages ("needs attention")
    - Highlights from active group chats
    - Detected action items (tasks, questions, events, reminders)
    - Message volume statistics

    Args:
        request: Digest generation options.

    Returns:
        DigestResponse containing all digest sections.
    """
    try:
        generator = DigestGenerator(reader)
        period = DigestPeriod(request.period.value)
        digest = generator.generate(period=period, end_date=request.end_date)
        return _digest_to_response(digest)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate digest: {e}",
        ) from e


@router.post("/export", response_model=DigestExportResponse)
def export_digest(
    request: DigestExportRequest,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DigestExportResponse:
    """Generate and export a digest in the specified format.

    Exports the digest as Markdown or HTML for viewing or sharing.

    Args:
        request: Export options including format and period.

    Returns:
        DigestExportResponse with exported content.
    """
    try:
        generator = DigestGenerator(reader)
        period = DigestPeriod(request.period.value)
        digest = generator.generate(period=period, end_date=request.end_date)

        # Export based on format
        if request.format == DigestFormatEnum.HTML:
            exported_data = export_digest_html(digest)
            export_format = DigestFormat.HTML
        else:
            exported_data = export_digest_markdown(digest)
            export_format = DigestFormat.MARKDOWN

        filename = get_digest_filename(
            period=period,
            format=export_format,
            date=digest.end_date,
        )

        return DigestExportResponse(
            success=True,
            format=request.format.value,
            filename=filename,
            data=exported_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export digest: {e}",
        ) from e


@router.get("/preferences", response_model=DigestPreferencesResponse)
def get_digest_preferences() -> DigestPreferencesResponse:
    """Get current digest preferences.

    Returns:
        DigestPreferencesResponse with current settings.
    """
    try:
        config = get_config()
        digest_config = config.digest

        return DigestPreferencesResponse(
            enabled=digest_config.enabled,
            schedule=digest_config.schedule,
            preferred_time=digest_config.preferred_time,
            include_action_items=digest_config.include_action_items,
            include_stats=digest_config.include_stats,
            max_conversations=digest_config.max_conversations,
            export_format=digest_config.export_format,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get digest preferences: {e}",
        ) from e


@router.put("/preferences", response_model=DigestPreferencesResponse)
def update_digest_preferences(
    request: DigestPreferencesUpdateRequest,
) -> DigestPreferencesResponse:
    """Update digest preferences.

    Only provided fields are updated. Others remain unchanged.

    Args:
        request: Fields to update.

    Returns:
        DigestPreferencesResponse with updated settings.
    """
    try:
        config = get_config()
        digest_config = config.digest

        # Update only provided fields
        if request.enabled is not None:
            digest_config.enabled = request.enabled
        if request.schedule is not None:
            if request.schedule not in ("daily", "weekly"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid schedule. Must be 'daily' or 'weekly'.",
                )
            digest_config.schedule = request.schedule  # type: ignore[assignment]
        if request.preferred_time is not None:
            # Validate time format
            import re

            if not re.match(r"^\d{2}:\d{2}$", request.preferred_time):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid time format. Use HH:MM.",
                )
            digest_config.preferred_time = request.preferred_time
        if request.include_action_items is not None:
            digest_config.include_action_items = request.include_action_items
        if request.include_stats is not None:
            digest_config.include_stats = request.include_stats
        if request.max_conversations is not None:
            digest_config.max_conversations = request.max_conversations
        if request.export_format is not None:
            if request.export_format not in ("markdown", "html"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid format. Must be 'markdown' or 'html'.",
                )
            digest_config.export_format = request.export_format  # type: ignore[assignment]

        # Save configuration
        save_config(config)

        return DigestPreferencesResponse(
            enabled=digest_config.enabled,
            schedule=digest_config.schedule,
            preferred_time=digest_config.preferred_time,
            include_action_items=digest_config.include_action_items,
            include_stats=digest_config.include_stats,
            max_conversations=digest_config.max_conversations,
            export_format=digest_config.export_format,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update digest preferences: {e}",
        ) from e


@router.get("/daily", response_model=DigestResponse)
def get_daily_digest(
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DigestResponse:
    """Get a daily digest (convenience endpoint).

    Generates a digest for the past 24 hours.

    Returns:
        DigestResponse with daily digest.
    """
    try:
        generator = DigestGenerator(reader)
        digest = generator.generate(period=DigestPeriod.DAILY)
        return _digest_to_response(digest)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate daily digest: {e}",
        ) from e


@router.get("/weekly", response_model=DigestResponse)
def get_weekly_digest(
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> DigestResponse:
    """Get a weekly digest (convenience endpoint).

    Generates a digest for the past 7 days.

    Returns:
        DigestResponse with weekly digest.
    """
    try:
        generator = DigestGenerator(reader)
        digest = generator.generate(period=DigestPeriod.WEEKLY)
        return _digest_to_response(digest)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate weekly digest: {e}",
        ) from e
