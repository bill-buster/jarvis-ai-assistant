"""Smart reply suggestions API endpoint.

Provides quick reply suggestions based on the last received message using
pattern matching. This is a lightweight, fast alternative to the AI-powered
draft generation that works without loading the language model.

Unlike /drafts/reply which uses the MLX model, this endpoint uses pre-defined
response patterns for common scenarios, making it extremely fast (~1ms).
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from api.ratelimit import RATE_LIMIT_READ, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


class SuggestionRequest(BaseModel):
    """Request for smart reply suggestions.

    Example:
        ```json
        {
            "last_message": "Are you free for dinner?",
            "num_suggestions": 3
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "last_message": "Are you free for dinner?",
                "num_suggestions": 3,
            }
        }
    )

    last_message: str = Field(
        ...,
        min_length=1,
        description="The last received message to generate suggestions for",
        examples=["Are you free for dinner?", "Thanks for your help!"],
    )
    num_suggestions: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of suggestions to return (1-5)",
    )


class Suggestion(BaseModel):
    """A single reply suggestion.

    Example:
        ```json
        {
            "text": "Sounds good!",
            "score": 0.95
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Sounds good!",
                "score": 0.95,
            }
        }
    )

    text: str = Field(
        ...,
        description="The suggested reply text",
        examples=["Sounds good!", "Sure!", "What time?"],
    )
    score: float = Field(
        ...,
        description="Relevance score (0.0 to 1.0) - higher is more relevant",
        examples=[0.95, 0.8, 0.5],
        ge=0.0,
        le=1.0,
    )


class SuggestionResponse(BaseModel):
    """Response containing reply suggestions.

    Example:
        ```json
        {
            "suggestions": [
                {"text": "Sounds good!", "score": 0.95},
                {"text": "What time?", "score": 0.85},
                {"text": "I'm in!", "score": 0.75}
            ]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestions": [
                    {"text": "Sounds good!", "score": 0.95},
                    {"text": "What time?", "score": 0.85},
                    {"text": "I'm in!", "score": 0.75},
                ]
            }
        }
    )

    suggestions: list[Suggestion] = Field(
        ...,
        description="List of reply suggestions sorted by relevance score",
    )


# Pre-defined response patterns
# Each tuple: (keywords_to_match, response_text, base_score)
RESPONSE_PATTERNS: list[tuple[list[str], str, float]] = [
    # Questions about time/scheduling
    (["what time", "when", "what's a good time"], "What time works for you?", 0.9),
    (["are you free", "you free", "available"], "Let me check my schedule!", 0.9),
    (["can we meet", "let's meet", "want to meet"], "Sure! When works for you?", 0.85),
    # Affirmative responses
    (["sounds good", "that works", "perfect"], "Sounds good!", 0.95),
    (["ok", "okay", "k", "kk"], "Got it!", 0.9),
    (["yes", "yeah", "yep", "yup"], "Great!", 0.85),
    (["sure", "of course"], "Sure thing!", 0.85),
    # Location questions
    (["where", "location", "address", "place"], "Where should we meet?", 0.85),
    # Gratitude
    (["thank", "thanks", "ty", "thx"], "You're welcome!", 0.95),
    (["appreciate"], "Happy to help!", 0.85),
    # Running late / timing
    (["running late", "gonna be late", "be there soon"], "No worries, take your time!", 0.95),
    (["omw", "on my way", "heading"], "See you soon!", 0.9),
    # Social invitations
    (["wanna hang", "want to hang", "free tonight"], "I'm down! What's the plan?", 0.85),
    (["dinner", "lunch", "food", "eat"], "I'm in! Where were you thinking?", 0.85),
    (["coffee", "drinks", "grab a"], "Sounds great! When?", 0.85),
    # Acknowledgments
    (["got it", "understood", "makes sense"], "Perfect!", 0.9),
    (["see you", "talk later", "ttyl", "bye"], "See you!", 0.9),
    # Excitement / agreement
    (["excited", "can't wait", "looking forward"], "Me too!", 0.9),
    (["same", "me too", "agreed"], "Right?!", 0.85),
    # Generic affirmative
    (["nice", "cool", "awesome", "great", "amazing"], "Thanks!", 0.75),
    # Fallback suggestions (lower scores, always available)
    ([], "Sounds good!", 0.3),
    ([], "Got it!", 0.25),
    ([], "Thanks!", 0.2),
]


def _compute_match_score(message: str, keywords: list[str], base_score: float) -> float:
    """Compute how well a message matches the keywords.

    Args:
        message: The incoming message (lowercase)
        keywords: Keywords to look for
        base_score: Base score if any keyword matches

    Returns:
        Score from 0 to base_score based on match quality
    """
    if not keywords:
        # Fallback patterns return their base score directly
        return base_score

    message_lower = message.lower()

    # Check for exact phrase matches first (highest score)
    for keyword in keywords:
        if keyword in message_lower:
            return base_score

    # Check for word-level matches
    message_words = set(re.findall(r"\b\w+\b", message_lower))
    for keyword in keywords:
        keyword_words = set(re.findall(r"\b\w+\b", keyword.lower()))
        if keyword_words & message_words:
            return base_score * 0.7  # Partial match

    return 0.0


@router.post(
    "",
    response_model=SuggestionResponse,
    response_model_exclude_unset=True,
    response_description="Quick reply suggestions ranked by relevance",
    summary="Get smart reply suggestions",
    responses={
        200: {
            "description": "Suggestions generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "suggestions": [
                            {"text": "I'm in! Where were you thinking?", "score": 0.85},
                            {"text": "Sounds good!", "score": 0.75},
                            {"text": "Got it!", "score": 0.25},
                        ]
                    }
                }
            },
        },
        429: {
            "description": "Rate limit exceeded",
        },
    },
)
@limiter.limit(RATE_LIMIT_READ)
async def get_suggestions(
    suggestion_request: SuggestionRequest, request: Request
) -> SuggestionResponse:
    """Get smart reply suggestions based on the last message.

    Returns contextually appropriate quick replies ranked by relevance.
    This endpoint uses pattern matching rather than AI generation,
    making it extremely fast (typically < 1ms).

    **When to Use This vs /drafts/reply:**
    - Use `/suggestions` for quick, common responses (fast, no model load)
    - Use `/drafts/reply` for contextual, AI-generated replies (slower, better quality)

    **How Scoring Works:**
    - Score 0.9-1.0: Strong keyword match (e.g., "thanks" → "You're welcome!")
    - Score 0.7-0.9: Partial word match
    - Score 0.3 or below: Generic fallback suggestions

    **Supported Patterns:**
    - Time/scheduling: "what time", "are you free", "when"
    - Affirmative: "sounds good", "yes", "okay"
    - Gratitude: "thanks", "thank you", "appreciate"
    - Social: "dinner", "lunch", "coffee", "drinks"
    - Running late: "omw", "on my way", "running late"
    - Goodbyes: "see you", "bye", "ttyl"

    **Example Request:**
    ```json
    {
        "last_message": "Want to grab dinner tonight?",
        "num_suggestions": 3
    }
    ```

    **Example Response:**
    ```json
    {
        "suggestions": [
            {"text": "I'm in! Where were you thinking?", "score": 0.85},
            {"text": "Sounds good!", "score": 0.3},
            {"text": "Got it!", "score": 0.25}
        ]
    }
    ```

    Args:
        suggestion_request: SuggestionRequest with last_message and num_suggestions
        request: FastAPI request object (for rate limiting)

    Returns:
        SuggestionResponse with ranked list of suggestions
    """
    message = suggestion_request.last_message.strip()
    if not message:
        return SuggestionResponse(suggestions=[])

    # Score all patterns
    scored: list[tuple[str, float]] = []
    seen_responses: set[str] = set()

    for keywords, response, base_score in RESPONSE_PATTERNS:
        if response in seen_responses:
            continue

        score = _compute_match_score(message, keywords, base_score)
        if score > 0:
            scored.append((response, score))
            seen_responses.add(response)

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    suggestions = [
        Suggestion(text=text, score=score)
        for text, score in scored[: suggestion_request.num_suggestions]
    ]

    logger.debug("Generated %d suggestions for message", len(suggestions))
    return SuggestionResponse(suggestions=suggestions)
