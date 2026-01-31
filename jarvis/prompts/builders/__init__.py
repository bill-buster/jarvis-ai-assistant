"""Prompt builders module.

This module contains functions for building prompts from templates and examples.
During the refactoring transition, builders are imported from the original prompts.py.
"""

# During transition, import from main prompts module
# This will be replaced with local definitions as content is migrated
from jarvis.prompts import (
    build_reply_prompt,
    build_summary_prompt,
    build_search_answer_prompt,
    build_threaded_reply_prompt,
    build_rag_reply_prompt,
    build_rag_reply_prompt_from_embeddings,
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

__all__ = [
    # Main builders
    "build_reply_prompt",
    "build_summary_prompt",
    "build_search_answer_prompt",
    "build_threaded_reply_prompt",
    "build_rag_reply_prompt",
    "build_rag_reply_prompt_from_embeddings",
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
]
