"""Prompt utilities module.

This module contains utility functions for token estimation and validation.
During the refactoring transition, utilities are imported from the original prompts.py.
"""

# During transition, import from main prompts module
# This will be replaced with local definitions as content is migrated
from jarvis.prompts import (
    estimate_tokens,
    is_within_token_limit,
)

__all__ = [
    "estimate_tokens",
    "is_within_token_limit",
]
