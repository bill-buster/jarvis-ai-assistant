"""Argument parser for JARVIS CLI.

This module will eventually contain all argument parsing logic for the CLI.
During the refactoring transition, the parser is imported from the main cli module.
"""

# During transition, import from main cli module
# This will be moved here as part of the refactoring
from jarvis.cli import create_parser

__all__ = ["create_parser"]
