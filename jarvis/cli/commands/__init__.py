"""CLI commands for JARVIS.

This package contains individual command modules for the JARVIS CLI.
Each command is implemented in its own module for better maintainability.

Available commands:
- chat: Interactive chat mode
- reply: Generate reply suggestions
- summarize: Summarize conversations
- search: Search iMessage conversations
- health: System health check
- benchmark: Performance benchmarks
- version: Show version info
- export: Export conversations
- batch: Batch operations
- tasks: Task management
- serve: Start API server
- db: Database operations
"""

# Import all command functions
# These will be populated as we extract commands from cli.py
# For now, they remain in the main cli.py module

__all__ = [
    "cmd_chat",
    "cmd_reply",
    "cmd_summarize",
    "cmd_search_messages",
    "cmd_search_semantic",
    "cmd_health",
    "cmd_benchmark",
    "cmd_version",
    "cmd_export",
    "cmd_batch",
    "cmd_tasks",
    "cmd_serve",
    "cmd_mcp_serve",
    "cmd_db",
    "cmd_examples",
]
