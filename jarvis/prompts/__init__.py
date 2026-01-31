"""Prompt templates and builders for JARVIS.

This package provides well-engineered prompts optimized for small local LLMs
with clear structure, few-shot examples, and tone-aware generation.

This is the SINGLE SOURCE OF TRUTH for all prompts in the JARVIS system.

During the refactoring transition, this package re-exports from jarvis._prompts
(the legacy module) to maintain backward compatibility while the package structure
is being set up.
"""

# During transition, import everything from the legacy prompts module
# This ensures existing code continues to work during migration
from jarvis._prompts import (
    PROMPT_LAST_UPDATED,
    PROMPT_VERSION,
    FewShotExample,
    PromptMetadata,
    PromptTemplate,
    UserStyleAnalysis,
    # Reply examples
    CASUAL_REPLY_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    REPLY_EXAMPLES,
    # Summary examples
    SUMMARIZATION_EXAMPLES,
    SUMMARY_EXAMPLES,
    # Search examples
    SEARCH_ANSWER_EXAMPLES,
    # Thread examples
    LOGISTICS_THREAD_EXAMPLES,
    EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    PLANNING_THREAD_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    QUICK_EXCHANGE_THREAD_EXAMPLES,
    THREAD_EXAMPLES,
    # Templates
    REPLY_TEMPLATE,
    SUMMARY_TEMPLATE,
    SEARCH_ANSWER_TEMPLATE,
    THREADED_REPLY_TEMPLATE,
    RAG_REPLY_TEMPLATE,
    # Builders
    build_reply_prompt,
    build_summary_prompt,
    build_search_answer_prompt,
    build_threaded_reply_prompt,
    build_rag_reply_prompt,
    build_rag_reply_prompt_from_embeddings,
    # Style analysis
    detect_tone,
    analyze_user_style,
    build_style_instructions,
    # Registry
    PromptRegistry,
    get_prompt_registry,
    reset_prompt_registry,
    API_REPLY_EXAMPLES_METADATA,
    API_SUMMARY_EXAMPLES_METADATA,
    # Utilities
    estimate_tokens,
    is_within_token_limit,
    # Helper functions
    _format_examples,
    _format_summary_examples,
    _format_search_examples,
    _truncate_context,
    _get_thread_examples,
    _format_thread_context,
    _get_length_guidance,
    _get_additional_instructions,
    _format_similar_exchanges,
    _format_relationship_context,
)

# Import submodules - these import from _prompts during transition
from jarvis.prompts import models
from jarvis.prompts import data
from jarvis.prompts import builders
from jarvis.prompts import style
from jarvis.prompts import registry
from jarvis.prompts import utils

__all__ = [
    # Constants
    "PROMPT_VERSION",
    "PROMPT_LAST_UPDATED",
    # Models
    "PromptMetadata",
    "FewShotExample",
    "PromptTemplate",
    "UserStyleAnalysis",
    # Reply examples
    "CASUAL_REPLY_EXAMPLES",
    "PROFESSIONAL_REPLY_EXAMPLES",
    "REPLY_EXAMPLES",
    # Summary examples
    "SUMMARIZATION_EXAMPLES",
    "SUMMARY_EXAMPLES",
    # Search examples
    "SEARCH_ANSWER_EXAMPLES",
    # Thread examples
    "LOGISTICS_THREAD_EXAMPLES",
    "EMOTIONAL_SUPPORT_THREAD_EXAMPLES",
    "PLANNING_THREAD_EXAMPLES",
    "CATCHING_UP_THREAD_EXAMPLES",
    "QUICK_EXCHANGE_THREAD_EXAMPLES",
    "THREAD_EXAMPLES",
    # Templates
    "REPLY_TEMPLATE",
    "SUMMARY_TEMPLATE",
    "SEARCH_ANSWER_TEMPLATE",
    "THREADED_REPLY_TEMPLATE",
    "RAG_REPLY_TEMPLATE",
    # Builders
    "build_reply_prompt",
    "build_summary_prompt",
    "build_search_answer_prompt",
    "build_threaded_reply_prompt",
    "build_rag_reply_prompt",
    "build_rag_reply_prompt_from_embeddings",
    # Style
    "detect_tone",
    "analyze_user_style",
    "build_style_instructions",
    # Registry
    "PromptRegistry",
    "get_prompt_registry",
    "reset_prompt_registry",
    "API_REPLY_EXAMPLES_METADATA",
    "API_SUMMARY_EXAMPLES_METADATA",
    # Utils
    "estimate_tokens",
    "is_within_token_limit",
    # Helper functions
    "_format_examples",
    "_format_summary_examples",
    "_format_search_examples",
    "_truncate_context",
    "_get_thread_examples",
    "_format_thread_context",
    "_get_length_guidance",
    "_get_additional_instructions",
    "_format_similar_exchanges",
    "_format_relationship_context",
    # Submodules
    "models",
    "data",
    "builders",
    "style",
    "registry",
    "utils",
]
