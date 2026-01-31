"""JARVIS CLI package.

This package provides the command-line interface for JARVIS,
organized into separate command modules for better maintainability.

Note: During the refactoring transition, the main CLI implementation
remains in jarvis/_cli_main.py. This package provides the structure for
the eventual migration of commands to individual modules.
"""

# Re-export all CLI functions from the main module
from jarvis._cli_main import (
    cmd_batch,
    cmd_benchmark,
    cmd_chat,
    cmd_db,
    cmd_examples,
    cmd_export,
    cmd_health,
    cmd_mcp_serve,
    cmd_reply,
    cmd_search_messages,
    cmd_search_semantic,
    cmd_serve,
    cmd_summarize,
    cmd_tasks,
    cmd_version,
    create_parser,
    main,
    run,
)
from jarvis.cli.utils import (
    ARGCOMPLETE_AVAILABLE,
    _format_jarvis_error,
    _parse_date,
    cleanup,
    console,
    logger,
    print_feature_status_table,
    run_with_error_handling,
    setup_logging,
)

__all__ = [
    # Commands
    "cmd_batch",
    "cmd_benchmark",
    "cmd_chat",
    "cmd_db",
    "cmd_examples",
    "cmd_export",
    "cmd_health",
    "cmd_mcp_serve",
    "cmd_reply",
    "cmd_search_messages",
    "cmd_search_semantic",
    "cmd_serve",
    "cmd_summarize",
    "cmd_tasks",
    "cmd_version",
    # Core
    "create_parser",
    "main",
    "run",
    # Utils
    "ARGCOMPLETE_AVAILABLE",
    "cleanup",
    "console",
    "_format_jarvis_error",
    "logger",
    "_parse_date",
    "print_feature_status_table",
    "run_with_error_handling",
    "setup_logging",
]
