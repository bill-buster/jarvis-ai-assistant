"""JARVIS CLI - Command-line interface for the JARVIS AI assistant.

Provides commands for chat, iMessage search, email summarization,
health monitoring, and benchmarking.
"""

import argparse
import logging
import sys
from typing import Any, NoReturn

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contracts.health import DegradationPolicy, FeatureState
from contracts.models import GenerationRequest
from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller

console = Console()
logger = logging.getLogger(__name__)

# Feature names for degradation controller
FEATURE_CHAT = "chat"
FEATURE_IMESSAGE = "imessage"
FEATURE_EMAIL = "email"


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


def initialize_system() -> tuple[bool, list[str]]:
    """Initialize JARVIS system components.

    Initializes the memory controller and degradation controller,
    registers features, and returns status.

    Returns:
        Tuple of (success, list of warnings)
    """
    warnings: list[str] = []

    # Initialize memory controller
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()

    logger.info(
        "Memory mode: %s (%.0f MB available)",
        state.current_mode.value,
        state.available_mb,
    )

    # Initialize degradation controller and register features
    deg_controller = get_degradation_controller()

    # Register chat feature
    deg_controller.register_feature(
        DegradationPolicy(
            feature_name=FEATURE_CHAT,
            health_check=lambda: True,  # Always healthy for now
            degraded_behavior=lambda prompt: _template_only_response(prompt),
            fallback_behavior=lambda prompt: _fallback_response(),
            recovery_check=lambda: True,
            max_failures=3,
        )
    )

    # Register iMessage feature
    deg_controller.register_feature(
        DegradationPolicy(
            feature_name=FEATURE_IMESSAGE,
            health_check=_check_imessage_access,
            degraded_behavior=lambda query: _imessage_degraded(query),
            fallback_behavior=lambda query: _imessage_fallback(),
            recovery_check=_check_imessage_access,
            max_failures=3,
        )
    )

    # Register email feature
    deg_controller.register_feature(
        DegradationPolicy(
            feature_name=FEATURE_EMAIL,
            health_check=_check_gmail_access,
            degraded_behavior=lambda: _email_degraded(),
            fallback_behavior=lambda: _email_fallback(),
            recovery_check=_check_gmail_access,
            max_failures=3,
        )
    )

    # Check for permission issues
    if not _check_imessage_access():
        warnings.append(
            "iMessage access unavailable. Grant Full Disk Access in "
            "System Settings > Privacy & Security > Full Disk Access."
        )

    if not _check_gmail_access():
        warnings.append(
            "Gmail integration not configured. Run 'jarvis setup-email' "
            "to configure Gmail API access."
        )

    return True, warnings


def _check_imessage_access() -> bool:
    """Check if iMessage database is accessible.

    Returns:
        True if accessible, False otherwise.
    """
    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            return reader.check_access()
    except Exception:
        return False


def _check_gmail_access() -> bool:
    """Check if Gmail API is configured.

    Returns:
        True if configured, False otherwise.
    """
    # Gmail integration is not yet implemented
    return False


def _template_only_response(prompt: str) -> str:
    """Generate response using only template matching.

    Args:
        prompt: User prompt.

    Returns:
        Template response or degraded message.
    """
    try:
        from models.templates import TemplateMatcher

        matcher = TemplateMatcher()
        match = matcher.match(prompt)
        if match:
            return match.template.response
    except Exception:
        pass
    return "I'm operating in limited mode. Please try a simpler query."


def _fallback_response() -> str:
    """Return a fallback response when chat is unavailable.

    Returns:
        Static fallback message.
    """
    return (
        "I'm currently unable to process your request. "
        "Please check system health with 'jarvis health'."
    )


def _imessage_degraded(query: str) -> list[Any]:
    """Return degraded iMessage search result.

    Args:
        query: Search query.

    Returns:
        Empty list with logged warning.
    """
    logger.warning("iMessage search running in degraded mode")
    return []


def _imessage_fallback() -> list[Any]:
    """Return fallback for iMessage when unavailable.

    Returns:
        Empty list.
    """
    return []


def _email_degraded() -> str:
    """Return degraded email summary.

    Returns:
        Degraded mode message.
    """
    return "Email summarization is running in degraded mode."


def _email_fallback() -> str:
    """Return fallback for email when unavailable.

    Returns:
        Unavailable message.
    """
    return "Email integration is not available."


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

        # Generate response using degradation controller
        def generate_response(prompt: str) -> str:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=[],
                max_tokens=200,
                temperature=0.7,
            )
            response = generator.generate(request)
            return response.text

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

    console.print(f"[bold]Searching messages for:[/bold] {query}\n")

    deg_controller = get_degradation_controller()

    def search_messages(search_query: str) -> list[Any]:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            if not reader.check_access():
                raise PermissionError("Cannot access iMessage database")
            return reader.search(search_query, limit=limit)

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
            sender = "Me" if msg.is_from_me else (msg.sender or "Unknown")
            text = msg.text[:80] + "..." if len(msg.text) > 80 else msg.text
            table.add_row(date_str, sender, text)

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


def cmd_summarize_emails(args: argparse.Namespace) -> int:
    """Summarize recent emails.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    deg_controller = get_degradation_controller()

    def summarize() -> str:
        # Gmail integration is not yet implemented
        raise NotImplementedError("Gmail integration not yet implemented")

    try:
        result = deg_controller.execute(FEATURE_EMAIL, summarize)
        console.print(f"[bold]Email Summary:[/bold]\n{result}")
        return 0
    except NotImplementedError:
        console.print(
            "[yellow]Email summarization is not yet implemented.[/yellow]\n"
            "Gmail integration (WS9) is planned for a future release."
        )
        return 0
    except Exception as e:
        logger.exception("Email summarization error")
        console.print(f"[red]Error: {e}[/red]")
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
        elif feature_name == FEATURE_EMAIL:
            details = "Not configured" if not _check_gmail_access() else "OK"
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
    search_parser.set_defaults(func=cmd_search_messages)

    # Summarize emails command
    email_parser = subparsers.add_parser(
        "summarize-emails",
        help="Summarize recent emails",
    )
    email_parser.set_defaults(func=cmd_summarize_emails)

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

        # Unload models
        try:
            from models import reset_generator

            reset_generator()
        except Exception:
            pass
    except Exception:
        pass


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
