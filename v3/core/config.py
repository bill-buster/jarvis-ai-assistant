"""Configuration settings for JARVIS v3.

Uses Pydantic for type-safe settings management.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptStrategy(str, Enum):
    """Prompt strategy for reply generation.

    - legacy: Few-shot examples followed by "them: {message}\nme:" completion
    - conversation: Simple conversation transcript with "me:" completion
    """
    LEGACY = "legacy"
    CONVERSATION = "conversation"


class GenerationSettings(BaseSettings):
    """Settings for reply generation."""

    # Model settings
    model_name: str = "lfm2.5-1.2b"
    max_tokens: int = 50  # Increased from 30 for better responses

    # Prompt strategy: "legacy" (few-shot) or "conversation" (natural continuation)
    prompt_strategy: PromptStrategy = PromptStrategy.LEGACY

    # Style hint for conversation prompt strategy
    conversation_style_hint: str = "brief, casual"

    # Generation parameters
    temperature_scale: list[float] = [0.2, 0.4, 0.6, 0.8, 0.9]

    # Confidence thresholds
    template_confidence: float = 0.7
    past_reply_confidence: float = 0.75

    # RAG weights
    same_convo_weight: float = 0.6
    cross_convo_weight: float = 0.4
    min_similarity_threshold: float = 0.55


class EmbeddingSettings(BaseSettings):
    """Settings for message embeddings and FAISS indices."""

    # Paths - use ~/.jarvis for persistent data across versions
    data_dir: Path = Path.home() / ".jarvis"
    db_path: Path = data_dir / "embeddings.db"
    faiss_cache_dir: Path = data_dir / "faiss_indices"

    # Search parameters
    batch_size: int = 100
    min_text_length: int = 3
    max_faiss_cache_size: int = 50

    # HNSW parameters
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # Time-weighting for past replies
    use_time_weighting: bool = True
    recency_weight: float = 0.15
    time_window_boost: float = 0.1
    day_type_boost: float = 0.05
    max_age_days: int = 365


class APISettings(BaseSettings):
    """Settings for the FastAPI server."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS
    allow_origins: list[str] = ["*"]

    # Default pagination limits
    default_conversation_limit: int = 50
    default_message_limit: int = 50
    generation_context_limit: int = 30


class Settings(BaseSettings):
    """Global settings object."""

    version: str = "3.0.0"
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    api: APISettings = Field(default_factory=APISettings)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="JARVIS_",
    )


# Global settings instance
settings = Settings()
