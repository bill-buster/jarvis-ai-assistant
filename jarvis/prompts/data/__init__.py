"""Prompt examples data module.

This module exports all few-shot examples and templates used by the prompts system.
During the refactoring transition, examples are imported from the original prompts.py.
"""

# During transition, import from legacy prompts module
# This will be replaced with local definitions as content is migrated
from jarvis._prompts import (
    CASUAL_REPLY_EXAMPLES,
    PROFESSIONAL_REPLY_EXAMPLES,
    SUMMARIZATION_EXAMPLES,
    SEARCH_ANSWER_EXAMPLES,
    LOGISTICS_THREAD_EXAMPLES,
    EMOTIONAL_SUPPORT_THREAD_EXAMPLES,
    PLANNING_THREAD_EXAMPLES,
    CATCHING_UP_THREAD_EXAMPLES,
    QUICK_EXCHANGE_THREAD_EXAMPLES,
    THREAD_EXAMPLES,
    REPLY_TEMPLATE,
    SUMMARY_TEMPLATE,
    SEARCH_ANSWER_TEMPLATE,
    THREADED_REPLY_TEMPLATE,
    RAG_REPLY_TEMPLATE,
)

__all__ = [
    # Reply examples
    "CASUAL_REPLY_EXAMPLES",
    "PROFESSIONAL_REPLY_EXAMPLES",
    # Other examples
    "SUMMARIZATION_EXAMPLES",
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
]
