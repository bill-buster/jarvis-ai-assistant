"""Relationship learning API endpoints.

Provides endpoints for managing relationship profiles that enable
personalized reply generation based on communication patterns with each contact.

Endpoints:
- GET /relationships/{contact_id} - Get relationship profile
- GET /relationships/{contact_id}/style-guide - Get natural language style description
- POST /relationships/{contact_id}/refresh - Rebuild profile from message history
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    ErrorResponse,
    RefreshProfileRequest,
    RefreshProfileResponse,
    RelationshipProfileResponse,
    ResponsePatternsResponse,
    StyleGuideResponse,
    ToneProfileResponse,
    TopicDistributionResponse,
)
from integrations.imessage import ChatDBReader
from jarvis.relationships import (
    RelationshipProfile,
    build_relationship_profile,
    generate_style_guide,
    get_voice_guidance,
    load_profile,
    profile_needs_refresh,
    save_profile,
)

router = APIRouter(prefix="/relationships", tags=["relationships"])


def _profile_to_response(profile: RelationshipProfile) -> RelationshipProfileResponse:
    """Convert internal RelationshipProfile to API response model."""
    return RelationshipProfileResponse(
        contact_id=profile.contact_id,
        contact_name=profile.contact_name,
        tone_profile=ToneProfileResponse(
            formality_score=profile.tone_profile.formality_score,
            emoji_frequency=profile.tone_profile.emoji_frequency,
            exclamation_frequency=profile.tone_profile.exclamation_frequency,
            question_frequency=profile.tone_profile.question_frequency,
            avg_message_length=profile.tone_profile.avg_message_length,
            uses_caps=profile.tone_profile.uses_caps,
        ),
        topic_distribution=TopicDistributionResponse(
            topics=profile.topic_distribution.topics,
            top_topics=profile.topic_distribution.top_topics,
        ),
        response_patterns=ResponsePatternsResponse(
            avg_response_time_minutes=profile.response_patterns.avg_response_time_minutes,
            typical_response_length=profile.response_patterns.typical_response_length,
            greeting_style=profile.response_patterns.greeting_style,
            signoff_style=profile.response_patterns.signoff_style,
            common_phrases=profile.response_patterns.common_phrases,
        ),
        message_count=profile.message_count,
        last_updated=profile.last_updated,
        version=profile.version,
    )


def _get_contact_name(reader: ChatDBReader, chat_id: str) -> str | None:
    """Get display name for a contact from conversation metadata."""
    try:
        conversations = reader.get_conversations(limit=100)
        for conv in conversations:
            if conv.chat_id == chat_id:
                return conv.display_name
        return None
    except Exception:
        return None


@router.get(
    "/{contact_id}",
    response_model=RelationshipProfileResponse,
    response_description="Relationship profile for the contact",
    summary="Get relationship profile",
    responses={
        200: {
            "description": "Profile retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "contact_id": "a1b2c3d4e5f6g7h8",
                        "contact_name": "John Doe",
                        "tone_profile": {
                            "formality_score": 0.3,
                            "emoji_frequency": 1.5,
                            "exclamation_frequency": 0.8,
                            "question_frequency": 0.2,
                            "avg_message_length": 45.5,
                            "uses_caps": False,
                        },
                        "topic_distribution": {
                            "topics": {"scheduling": 0.35, "food": 0.25},
                            "top_topics": ["scheduling", "food"],
                        },
                        "response_patterns": {
                            "avg_response_time_minutes": 15.5,
                            "typical_response_length": "medium",
                            "greeting_style": ["hey", "hi"],
                            "signoff_style": ["thanks"],
                            "common_phrases": ["sounds good"],
                        },
                        "message_count": 250,
                        "last_updated": "2024-01-15T10:30:00",
                        "version": "1.0.0",
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "No profile exists for this contact",
            "model": ErrorResponse,
        },
    },
)
def get_relationship_profile(
    contact_id: str,
    auto_build: bool = Query(
        default=True,
        description="Automatically build profile if none exists",
    ),
    message_limit: int = Query(
        default=500,
        ge=50,
        le=2000,
        description="Maximum messages to analyze (if auto-building)",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> RelationshipProfileResponse:
    """Get the relationship profile for a contact.

    Returns the analyzed communication patterns, topic distribution, and
    response behaviors learned from message history with this contact.

    The contact_id can be:
    - A chat_id (e.g., "chat123456789")
    - A phone number (e.g., "+15551234567")
    - An email address (e.g., "john@example.com")

    If no profile exists and auto_build is True, a new profile will be
    built from the contact's message history.

    **Profile Contents:**
    - **tone_profile**: Formality level, emoji usage, punctuation patterns
    - **topic_distribution**: Common conversation topics (scheduling, food, work, etc.)
    - **response_patterns**: Response time, message length, common phrases

    **Example Response:**
    ```json
    {
        "contact_id": "a1b2c3d4e5f6g7h8",
        "contact_name": "John Doe",
        "tone_profile": {
            "formality_score": 0.3,
            "emoji_frequency": 1.5
        },
        "topic_distribution": {
            "topics": {"scheduling": 0.35},
            "top_topics": ["scheduling"]
        },
        "response_patterns": {
            "typical_response_length": "medium"
        },
        "message_count": 250
    }
    ```

    Args:
        contact_id: Contact identifier (chat_id, phone, or email)
        auto_build: Whether to build profile if none exists
        message_limit: Maximum messages to analyze for auto-build

    Returns:
        RelationshipProfileResponse with communication patterns

    Raises:
        HTTPException 404: No profile exists and auto_build is False
        HTTPException 403: Full Disk Access permission not granted
    """
    # Try to load existing profile
    profile = load_profile(contact_id)

    if profile is not None:
        return _profile_to_response(profile)

    # No profile exists - build one if auto_build is enabled
    if not auto_build:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ProfileNotFound",
                "message": f"No relationship profile exists for contact: {contact_id}",
                "hint": "Use POST /relationships/{contact_id}/refresh to build a profile, "
                "or set auto_build=true",
            },
        )

    # Fetch messages and build profile
    try:
        messages = reader.get_messages(chat_id=contact_id, limit=message_limit)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "MessageFetchError",
                "message": f"Could not fetch messages for contact: {contact_id}",
                "cause": str(e),
            },
        )

    if not messages:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NoMessages",
                "message": f"No messages found for contact: {contact_id}",
            },
        )

    # Get contact name
    contact_name = _get_contact_name(reader, contact_id)

    # Build and save profile
    profile = build_relationship_profile(contact_id, messages, contact_name)
    save_profile(profile)

    return _profile_to_response(profile)


@router.get(
    "/{contact_id}/style-guide",
    response_model=StyleGuideResponse,
    response_description="Natural language style guide for the contact",
    summary="Get style guide for contact",
    responses={
        200: {
            "description": "Style guide generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "contact_id": "a1b2c3d4e5f6g7h8",
                        "contact_name": "John Doe",
                        "style_guide": "Keep it casual, use emojis, brief messages.",
                        "voice_guidance": {
                            "formality": "casual",
                            "use_emojis": True,
                            "emoji_level": "high",
                            "message_length": "short",
                        },
                    }
                }
            },
        },
        403: {
            "description": "Full Disk Access not granted",
            "model": ErrorResponse,
        },
        404: {
            "description": "No profile exists for this contact",
            "model": ErrorResponse,
        },
    },
)
def get_style_guide(
    contact_id: str,
    auto_build: bool = Query(
        default=True,
        description="Automatically build profile if none exists",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> StyleGuideResponse:
    """Get a natural language style guide for communicating with a contact.

    Returns a human-readable description of how you typically communicate
    with this contact, plus structured guidance parameters that can be used
    in prompt building.

    The style guide describes:
    - Tone and formality level
    - Emoji usage patterns
    - Message length preferences
    - Common greetings and sign-offs
    - Typical conversation topics

    **Example Response:**
    ```json
    {
        "contact_id": "a1b2c3d4e5f6g7h8",
        "contact_name": "John Doe",
        "style_guide": "Keep it casual, use emojis, keep messages brief.",
        "voice_guidance": {
            "formality": "casual",
            "use_emojis": true,
            "emoji_level": "high",
            "message_length": "short",
            "common_greetings": ["hey", "hi"],
            "top_topics": ["scheduling", "food"]
        }
    }
    ```

    Args:
        contact_id: Contact identifier (chat_id, phone, or email)
        auto_build: Whether to build profile if none exists

    Returns:
        StyleGuideResponse with natural language guidance

    Raises:
        HTTPException 404: No profile exists and auto_build is False
        HTTPException 403: Full Disk Access permission not granted
    """
    # Try to load existing profile
    profile = load_profile(contact_id)

    if profile is None:
        if not auto_build:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "ProfileNotFound",
                    "message": f"No relationship profile exists for contact: {contact_id}",
                    "hint": "Use POST /relationships/{contact_id}/refresh to build a profile",
                },
            )

        # Build profile automatically
        try:
            messages = reader.get_messages(chat_id=contact_id, limit=500)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "MessageFetchError",
                    "message": f"Could not fetch messages for contact: {contact_id}",
                    "cause": str(e),
                },
            )

        if not messages:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NoMessages",
                    "message": f"No messages found for contact: {contact_id}",
                },
            )

        contact_name = _get_contact_name(reader, contact_id)
        profile = build_relationship_profile(contact_id, messages, contact_name)
        save_profile(profile)

    # Generate style guide
    style_guide = generate_style_guide(profile)
    voice_guidance = get_voice_guidance(profile)

    return StyleGuideResponse(
        contact_id=profile.contact_id,
        contact_name=profile.contact_name,
        style_guide=style_guide,
        voice_guidance=voice_guidance,
    )


@router.post(
    "/{contact_id}/refresh",
    response_model=RefreshProfileResponse,
    response_description="Result of profile refresh operation",
    summary="Refresh relationship profile",
    responses={
        200: {
            "description": "Profile refreshed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "messages_analyzed": 500,
                        "previous_message_count": 250,
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
def refresh_relationship_profile(
    contact_id: str,
    request: RefreshProfileRequest | None = None,
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> RefreshProfileResponse:
    """Rebuild the relationship profile from message history.

    Analyzes the contact's message history to build or update their
    communication profile. This is useful when:
    - The profile is outdated
    - You want to analyze more messages
    - The contact's communication style has changed

    The profile is saved automatically after building.

    **Profile Analysis:**
    - **tone_profile**: Analyzes formality, emoji usage, punctuation patterns
    - **topic_distribution**: Identifies common conversation topics
    - **response_patterns**: Calculates response times, message lengths, common phrases

    **Example Request:**
    ```json
    {
        "message_limit": 1000,
        "force_refresh": true
    }
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "profile": {...},
        "messages_analyzed": 500,
        "previous_message_count": 250
    }
    ```

    Args:
        contact_id: Contact identifier (chat_id, phone, or email)
        request: Optional refresh parameters

    Returns:
        RefreshProfileResponse with the new profile

    Raises:
        HTTPException 403: Full Disk Access permission not granted
    """
    # Set defaults if no request body
    if request is None:
        request = RefreshProfileRequest()

    # Check existing profile
    existing_profile = load_profile(contact_id)
    previous_count = existing_profile.message_count if existing_profile else None

    # Check if refresh is needed (unless force_refresh)
    if existing_profile and not request.force_refresh:
        if not profile_needs_refresh(existing_profile):
            return RefreshProfileResponse(
                success=True,
                profile=_profile_to_response(existing_profile),
                messages_analyzed=existing_profile.message_count,
                previous_message_count=previous_count,
                error=None,
            )

    # Fetch messages
    try:
        messages = reader.get_messages(chat_id=contact_id, limit=request.message_limit)
    except Exception as e:
        return RefreshProfileResponse(
            success=False,
            profile=None,
            messages_analyzed=0,
            previous_message_count=previous_count,
            error=f"Could not fetch messages: {e}",
        )

    if not messages:
        return RefreshProfileResponse(
            success=False,
            profile=None,
            messages_analyzed=0,
            previous_message_count=previous_count,
            error="No messages found for this contact",
        )

    # Get contact name
    contact_name = _get_contact_name(reader, contact_id)

    # Build new profile
    profile = build_relationship_profile(contact_id, messages, contact_name)

    # Save profile
    if not save_profile(profile):
        return RefreshProfileResponse(
            success=False,
            profile=_profile_to_response(profile),
            messages_analyzed=len(messages),
            previous_message_count=previous_count,
            error="Failed to save profile to disk",
        )

    return RefreshProfileResponse(
        success=True,
        profile=_profile_to_response(profile),
        messages_analyzed=len(messages),
        previous_message_count=previous_count,
        error=None,
    )


@router.delete(
    "/{contact_id}",
    response_description="Confirmation of profile deletion",
    summary="Delete relationship profile",
    responses={
        200: {
            "description": "Profile deleted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Profile deleted for contact",
                    }
                }
            },
        },
    },
)
def delete_relationship_profile(
    contact_id: str,
) -> dict[str, str | bool]:
    """Delete the relationship profile for a contact.

    Removes the stored profile from disk. The profile can be rebuilt
    later using the refresh endpoint.

    Args:
        contact_id: Contact identifier (chat_id, phone, or email)

    Returns:
        Confirmation of deletion
    """
    from jarvis.relationships import delete_profile

    success = delete_profile(contact_id)

    return {
        "success": success,
        "message": "Profile deleted for contact" if success else "No profile found to delete",
        "contact_id": contact_id,
    }
