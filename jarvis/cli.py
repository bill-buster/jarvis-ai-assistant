"""JARVIS CLI - Command-line interface for the JARVIS AI assistant.

Provides commands for chat, iMessage search, health monitoring, and benchmarking.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Any, NoReturn

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contracts.health import FeatureState
from contracts.models import GenerationRequest
from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller
from jarvis.system import (
    DEFAULT_MAX_TOKENS,
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    MESSAGE_PREVIEW_LENGTH,
    _check_imessage_access,
    initialize_system,
)

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


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat mode.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    console.print(
        Panel(
            "[bold green]JARVIS Chat[/bold green]\n"
            "Type your message and press Enter. Type 'quit' or 'exit' to leave.",
            title="Chat Mode",
        )
    )

    # Check memory mode
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()
    console.print(f"[dim]Operating in {state.current_mode.value} mode[/dim]\n")

    try:
        from models import get_generator
    except ImportError:
        console.print("[red]Error: Model system not available.[/red]")
        return 1

    generator = get_generator()
    deg_controller = get_degradation_controller()

    # Define response generator outside the loop to avoid function redefinition
    def generate_response(prompt: str) -> str:
        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],
            few_shot_examples=[],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.7,
        )
        response = generator.generate(request)
        return response.text

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        try:
            result = deg_controller.execute(
                FEATURE_CHAT,
                generate_response,
                user_input,
            )
            console.print(f"[bold green]JARVIS:[/bold green] {result}\n")
        except Exception as e:
            logger.exception("Chat error")
            console.print(f"[red]Error: {e}[/red]\n")

    return 0


def cmd_search_messages(args: argparse.Namespace) -> int:
    """Search iMessage conversations.

    Args:
        args: Parsed arguments with 'query' attribute.

    Returns:
        Exit code.
    """
    query = args.query
    limit = args.limit

    # Parse optional filter arguments
    start_date = _parse_date(args.start_date) if args.start_date else None
    end_date = _parse_date(args.end_date) if args.end_date else None
    sender = args.sender
    has_attachment = args.has_attachment

    console.print(f"[bold]Searching messages for:[/bold] {query}\n")

    # Show active filters
    filters_active = []
    if start_date:
        filters_active.append(f"after {start_date.strftime('%Y-%m-%d')}")
    if end_date:
        filters_active.append(f"before {end_date.strftime('%Y-%m-%d')}")
    if sender:
        filters_active.append(f"from {sender}")
    if has_attachment is not None:
        filters_active.append("with attachments" if has_attachment else "without attachments")

    if filters_active:
        console.print(f"[dim]Filters: {', '.join(filters_active)}[/dim]\n")

    deg_controller = get_degradation_controller()

    def search_messages(search_query: str) -> list[Any]:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            if not reader.check_access():
                raise PermissionError("Cannot access iMessage database")
            return reader.search(
                search_query,
                limit=limit,
                after=start_date,
                before=end_date,
                sender=sender,
                has_attachments=has_attachment,
            )

    try:
        messages = deg_controller.execute(FEATURE_IMESSAGE, search_messages, query)

        if not messages:
            console.print("[yellow]No messages found.[/yellow]")
            return 0

        # Display results
        table = Table(title=f"Search Results ({len(messages)} messages)")
        table.add_column("Date", style="dim")
        table.add_column("Sender")
        table.add_column("Message")

        for msg in messages[:limit]:
            date_str = msg.date.strftime("%Y-%m-%d %H:%M") if msg.date else "Unknown"
            display_sender = "Me" if msg.is_from_me else (msg.sender or "Unknown")
            if len(msg.text) > MESSAGE_PREVIEW_LENGTH:
                text = msg.text[:MESSAGE_PREVIEW_LENGTH] + "..."
            else:
                text = msg.text
            table.add_row(date_str, display_sender, text)

        console.print(table)
        return 0

    except PermissionError as e:
        console.print(f"[red]Permission error: {e}[/red]")
        console.print(
            "[yellow]Grant Full Disk Access to your terminal in "
            "System Settings > Privacy & Security.[/yellow]"
        )
        return 1
    except Exception as e:
        logger.exception("Search error")
        console.print(f"[red]Error searching messages: {e}[/red]")
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Display system health status.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    console.print(Panel("[bold]JARVIS System Health[/bold]", title="Health Check"))

    # Memory status
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()

    mem_table = Table(title="Memory Status")
    mem_table.add_column("Metric", style="bold")
    mem_table.add_column("Value")

    mem_table.add_row("Available Memory", f"{state.available_mb:.0f} MB")
    mem_table.add_row("Used Memory", f"{state.used_mb:.0f} MB")
    mem_table.add_row("Operating Mode", state.current_mode.value)
    mem_table.add_row("Pressure Level", state.pressure_level)
    mem_table.add_row("Model Loaded", "Yes" if state.model_loaded else "No")

    console.print(mem_table)
    console.print()

    # Feature health
    deg_controller = get_degradation_controller()
    health = deg_controller.get_health()

    feature_table = Table(title="Feature Status")
    feature_table.add_column("Feature", style="bold")
    feature_table.add_column("Status")
    feature_table.add_column("Details")

    status_colors = {
        FeatureState.HEALTHY: "green",
        FeatureState.DEGRADED: "yellow",
        FeatureState.FAILED: "red",
    }

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
    console.print()

    # Model status
    try:
        from models import get_generator

        generator = get_generator()
        model_loaded = generator.is_loaded()

        model_table = Table(title="Model Status")
        model_table.add_column("Metric", style="bold")
        model_table.add_column("Value")

        model_table.add_row("Loaded", "Yes" if model_loaded else "No")
        if model_loaded:
            model_table.add_row("Memory Usage", f"{generator.get_memory_usage_mb():.0f} MB")

        console.print(model_table)
    except Exception as e:
        console.print(f"[yellow]Model status unavailable: {e}[/yellow]")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks.

    Args:
        args: Parsed arguments with 'type' attribute.

    Returns:
        Exit code.
    """
    benchmark_type = args.type
    output_file = args.output

    console.print(f"[bold]Running {benchmark_type} benchmark...[/bold]\n")

    # Map benchmark types to their modules
    benchmark_modules = {
        "memory": "benchmarks.memory.run",
        "latency": "benchmarks.latency.run",
        "hhem": "benchmarks.hallucination.run",
    }

    if benchmark_type not in benchmark_modules:
        console.print(f"[red]Unknown benchmark type: {benchmark_type}[/red]")
        console.print(f"Available types: {', '.join(benchmark_modules.keys())}")
        return 1

    # Build command to run the benchmark module
    import subprocess

    cmd = [sys.executable, "-m", benchmark_modules[benchmark_type]]
    if output_file:
        cmd.extend(["--output", output_file])

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Display version information.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from jarvis import __version__

    console.print(f"JARVIS AI Assistant v{__version__}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API server.

    Args:
        args: Parsed arguments with host, port, and reload options.

    Returns:
        Exit code.
    """
    import uvicorn

    host = args.host
    port = args.port
    reload = args.reload

    console.print(
        Panel(
            f"[bold green]Starting JARVIS API Server[/bold green]\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Reload: {'Enabled' if reload else 'Disabled'}",
            title="API Server",
        )
    )

    try:
        uvicorn.run(
            "jarvis.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
        return 0
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS - Local-first AI assistant for macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jarvis chat                    Start interactive chat
  jarvis search-messages "hello" Search iMessage for "hello"
  jarvis health                  Show system health status
  jarvis benchmark memory        Run memory benchmark
  jarvis serve                   Start the API server
  jarvis serve --port 3000       Start server on custom port
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat mode",
    )
    chat_parser.set_defaults(func=cmd_chat)

    # Search messages command
    search_parser = subparsers.add_parser(
        "search-messages",
        help="Search iMessage conversations",
    )
    search_parser.add_argument(
        "query",
        help="Search query string",
    )
    search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)",
    )
    search_parser.add_argument(
        "--start-date",
        dest="start_date",
        help="Filter messages after this date (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    search_parser.add_argument(
        "--end-date",
        dest="end_date",
        help="Filter messages before this date (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    search_parser.add_argument(
        "--sender",
        help="Filter by sender phone number or email (use 'me' for your own messages)",
    )
    search_parser.add_argument(
        "--has-attachment",
        dest="has_attachment",
        action="store_true",
        default=None,
        help="Show only messages with attachments",
    )
    search_parser.add_argument(
        "--no-attachment",
        dest="has_attachment",
        action="store_false",
        help="Show only messages without attachments",
    )
    search_parser.set_defaults(func=cmd_search_messages)

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Show system health status",
    )
    health_parser.set_defaults(func=cmd_health)

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks",
    )
    bench_parser.add_argument(
        "type",
        choices=["memory", "latency", "hhem"],
        help="Benchmark type to run",
    )
    bench_parser.add_argument(
        "-o",
        "--output",
        help="Output file for results (JSON)",
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    # Version command (also accessible via --version)
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    version_parser.set_defaults(func=cmd_version)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the API server",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --version flag
    if args.version:
        return cmd_version(args)

    # Setup logging
    setup_logging(args.verbose)

    # Initialize system
    success, warnings = initialize_system()
    if not success:
        console.print("[red]Failed to initialize JARVIS system.[/red]")
        return 1

    # Show warnings
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    # Run command or show help
    if args.command is None:
        parser.print_help()
        return 0

    result: int = args.func(args)
    return result


def cleanup() -> None:
    """Clean up system resources."""
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


def run() -> NoReturn:
    """Entry point that handles cleanup and exit."""
    try:
        exit_code = main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        exit_code = 130
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected error")
        exit_code = 1
    finally:
        cleanup()

    sys.exit(exit_code)


if __name__ == "__main__":
    run()
