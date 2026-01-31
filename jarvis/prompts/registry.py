"""Prompt registry module.

This module contains the PromptRegistry class for dynamic prompt management.
During the refactoring transition, the registry is imported from the original prompts.py.
"""

# During transition, import from main prompts module
# This will be replaced with local definitions as content is migrated
from jarvis.prompts import (
    PromptRegistry,
    get_prompt_registry,
    reset_prompt_registry,
    API_REPLY_EXAMPLES_METADATA,
    API_SUMMARY_EXAMPLES_METADATA,
)

__all__ = [
    "PromptRegistry",
    "get_prompt_registry",
    "reset_prompt_registry",
    "API_REPLY_EXAMPLES_METADATA",
    "API_SUMMARY_EXAMPLES_METADATA",
]
