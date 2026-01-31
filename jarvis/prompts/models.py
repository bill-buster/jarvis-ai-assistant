"""Data models for prompts module.

This module contains all dataclass definitions used by the prompts system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

# Version info used by models
PROMPT_VERSION = "1.0.0"
PROMPT_LAST_UPDATED = "2026-01-26"

# Token limit guidance for small models
MAX_PROMPT_TOKENS = 1500  # Reserve space for generation
MAX_CONTEXT_CHARS = 4000  # Approximate, ~4 chars per token


@dataclass
class PromptMetadata:
    """Metadata for a prompt or example set.

    Attributes:
        name: Human-readable name for the prompt/example set
        version: Semantic version string (e.g., "1.0.0")
        last_updated: ISO date string of last update
        description: Brief description of the prompt's purpose
    """

    name: str
    version: str = PROMPT_VERSION
    last_updated: str = PROMPT_LAST_UPDATED
    description: str = ""


@dataclass
class FewShotExample:
    """A few-shot example for prompt engineering.

    Attributes:
        context: The conversation context
        output: The expected output/reply
        tone: The tone of the example (casual/professional)
    """

    context: str
    output: str
    tone: Literal["casual", "professional"] = "casual"


@dataclass
class PromptTemplate:
    """A prompt template with placeholders.

    Attributes:
        name: Template identifier
        system_message: Role/context for the model
        template: Format string with {placeholders}
        max_output_tokens: Suggested max tokens for response
    """

    name: str
    system_message: str
    template: str
    max_output_tokens: int = 100


@dataclass
class UserStyleAnalysis:
    """Analysis of user's texting style from message examples.

    Attributes:
        avg_length: Average message length in characters
        min_length: Minimum message length seen
        max_length: Maximum message length seen
        formality: Detected formality level
        uses_lowercase: Whether user typically uses lowercase
        uses_abbreviations: Whether user uses text abbreviations (u, ur, gonna)
        uses_minimal_punctuation: Whether user avoids excessive punctuation
        common_abbreviations: List of abbreviations user commonly uses
        emoji_frequency: Emojis per message
        exclamation_frequency: Exclamation marks per message
    """

    avg_length: float = 50.0
    min_length: int = 0
    max_length: int = 200
    formality: Literal["formal", "casual", "very_casual"] = "casual"
    uses_lowercase: bool = False
    uses_abbreviations: bool = False
    uses_minimal_punctuation: bool = False
    common_abbreviations: list[str] = field(default_factory=list)
    emoji_frequency: float = 0.0
    exclamation_frequency: float = 0.0
