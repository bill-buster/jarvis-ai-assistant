"""Shared utilities for JARVIS CLI commands.

Common functions used across multiple CLI commands.
"""

import logging
import sys
from datetime import UTC, datetime
from typing import NoReturn

# Optional argcomplete support for shell completion
try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

from rich.console import Console
from rich.text import Text

from contracts.health import FeatureState
from jarvis.errors import (
    ConfigurationError,
    JarvisError,
    ModelError,
    ResourceError,
    iMessageAccessError,
    iMessageError,
)
from jarvis.system import _check_imessage_access

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a datetime object.

    Supports formats:
    - YYYY-MM-DD (e.g., 2024-01-15)
    - YYYY-MM-DD HH:MM (e.g., 2024-01-15 14:30)

    Args:
        date_str: Date string to parse.

    Returns:
        datetime object with UTC timezone, or None if parsing fails.
    """
    formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    logger.warning(f"Could not parse date: {date_str}")
    return None


def _format_jarvis_error(error: JarvisError) -> None:
    """Format and display a JARVIS error with helpful suggestions.

    Args:
        error: The JARVIS error to display.
    """
    # Main error message
    console.print(f"[red]Error: {error.message}[/red]")

    # Provide type-specific guidance
    if isinstance(error, iMessageAccessError):
        # Show permission instructions if available
        if error.details.get("requires_permission"):
            instructions = error.details.get("permission_instructions", [])
            if instructions:
                console.print("\n[yellow]To fix this:[/yellow]")
                for i, instruction in enumerate(instructions, 1):
                    console.print(f"  {i}. {instruction}")
        else:
            console.print(
                "[yellow]Grant Full Disk Access in System Settings > Privacy & Security.[/yellow]"
            )
    elif isinstance(error, iMessageError):
        console.print("[yellow]Check that iMessage is accessible and try again.[/yellow]")
    elif isinstance(error, ModelError):
        if error.details.get("available_mb") and error.details.get("required_mb"):
            console.print(
                f"[yellow]Available: {error.details['available_mb']} MB, "
                f"Required: {error.details['required_mb']} MB[/yellow]"
            )
        console.print("[yellow]Try closing other applications to free memory.[/yellow]")
    elif isinstance(error, ResourceError):
        if error.details.get("resource_type") == "memory":
            console.print("[yellow]Close other applications to free up memory.[/yellow]")
        elif error.details.get("resource_type") == "disk":
            console.print("[yellow]Free up disk space and try again.[/yellow]")
    elif isinstance(error, ConfigurationError):
        if error.details.get("config_path"):
            console.print(f"[yellow]Config file: {error.details['config_path']}[/yellow]")
        console.print("[yellow]Try running 'jarvis health' to diagnose the issue.[/yellow]")

    # Log with details for debugging
    logger.debug(
        "JarvisError details - code=%s, details=%s, cause=%s",
        error.code.value,
        error.details,
        error.cause,
    )


def cleanup() -> None:
    """Clean up system resources."""
    from core.health import reset_degradation_controller
    from core.memory import reset_memory_controller

    try:
        # Reset singletons to free resources
        reset_memory_controller()
        reset_degradation_controller()
    except Exception as e:
        logger.debug("Error resetting controllers during cleanup: %s", e)

    # Unload models (separate try block so we attempt all cleanup steps)
    try:
        from models import reset_generator

        reset_generator()
    except ImportError:
        # Models module not available, nothing to clean up
        pass
    except Exception as e:
        logger.debug("Error resetting generator during cleanup: %s", e)


def run_with_error_handling(main_func, argv: list[str] | None = None) -> NoReturn:
    """Run main function with error handling and cleanup.

    Args:
        main_func: The main function to run (should return exit code)
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        NoReturn - always calls sys.exit()
    """
    try:
        exit_code = main_func(argv)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        exit_code = 130
    except JarvisError as e:
        _format_jarvis_error(e)
        logger.exception("JARVIS error")
        exit_code = 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected error")
        exit_code = 1
    finally:
        cleanup()

    sys.exit(exit_code)


def print_feature_status_table(health: dict[str, FeatureState]) -> None:
    """Print a table showing feature status.

    Args:
        health: Dictionary mapping feature names to their states.
    """
    from rich.table import Table

    feature_table = Table(title="Feature Status")
    feature_table.add_column("Feature", style="bold")
    feature_table.add_column("Status")
    feature_table.add_column("Details")

    status_colors = {
        FeatureState.HEALTHY: "green",
        FeatureState.DEGRADED: "yellow",
        FeatureState.FAILED: "red",
    }

    from jarvis.system import FEATURE_CHAT, FEATURE_IMESSAGE

    for feature_name, feature_state in health.items():
        color = status_colors.get(feature_state, "white")
        status = Text(feature_state.value, style=color)

        # Get additional details
        details = ""
        if feature_name == FEATURE_IMESSAGE:
            details = "Full Disk Access required" if not _check_imessage_access() else "OK"
        elif feature_name == FEATURE_CHAT:
            details = "OK"

        feature_table.add_row(feature_name, status, details)

    console.print(feature_table)
