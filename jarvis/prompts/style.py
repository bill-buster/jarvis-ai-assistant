"""Style analysis module for prompts.

This module contains functions for analyzing user texting style and building
style instructions for prompts.
During the refactoring transition, functions are imported from the original prompts.py.
"""

# During transition, import from main prompts module
# This will be replaced with local definitions as content is migrated
from jarvis.prompts import (
    detect_tone,
    analyze_user_style,
    build_style_instructions,
)

__all__ = [
    "detect_tone",
    "analyze_user_style",
    "build_style_instructions",
]
