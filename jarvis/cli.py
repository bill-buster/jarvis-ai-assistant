#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""JARVIS CLI - Command-line interface for the JARVIS AI assistant.

Provides commands for chat, iMessage search, health monitoring, and benchmarking.

Usage:
    jarvis chat                      Start interactive chat
    jarvis search-messages "query"   Search iMessage conversations
    jarvis reply John                Generate reply suggestions
    jarvis summarize Sarah           Summarize a conversation
    jarvis health                    Show system health status
    jarvis benchmark memory          Run performance benchmarks
    jarvis serve                     Start the API server
    jarvis --examples                Show detailed usage examples

For comprehensive documentation, see docs/CLI_GUIDE.md

Shell Completion:
    To enable shell completion, install argcomplete and run:

    Bash:  eval "$(register-python-argcomplete jarvis)"
    Zsh:   eval "$(register-python-argcomplete jarvis)"
    Fish:  register-python-argcomplete --shell fish jarvis | source
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Any, NoReturn

# Optional argcomplete support for shell completion
try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contracts.health import FeatureState
from contracts.models import GenerationRequest
from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller
from jarvis.context import ContextFetcher
from jarvis.db import get_db
from jarvis.errors import (
    ConfigurationError,
    JarvisError,
    ModelError,
    ResourceError,
    iMessageAccessError,
    iMessageError,
)
from jarvis.intent import IntentClassifier, IntentResult, IntentType
from jarvis.prompts import (
    REPLY_EXAMPLES,
    SUMMARY_EXAMPLES,
    build_reply_prompt,
    build_summary_prompt,
)
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

# Singleton intent classifier
_intent_classifier: IntentClassifier | None = None


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


def get_intent_classifier() -> IntentClassifier:
    """Get or create singleton intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


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


def _handle_quick_reply(prompt: str) -> str:
    """Handle quick reply using template matcher (no LLM).

    Args:
        prompt: User input.

    Returns:
        Template response or fallback.
    """
    from models.templates import TemplateMatcher

    matcher = TemplateMatcher()
    match = matcher.match(prompt)
    if match:
        return match.template.response
    return "Got it!"


def _handle_reply_intent(prompt: str, intent_result: IntentResult) -> str:
    """Handle reply intent with conversation context.

    Args:
        prompt: Original user input.
        intent_result: Classified intent with extracted params.

    Returns:
        Generated reply or error message.
    """
    from integrations.imessage import ChatDBReader
    from models import get_generator

    person_name = intent_result.extracted_params.get("person_name")

    if not _check_imessage_access():
        return (
            "I can't access your messages right now. "
            "Please grant Full Disk Access in System Settings > Privacy & Security."
        )

    try:
        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)

            # Find conversation by person name
            if person_name:
                chat_id = fetcher.find_conversation_by_name(person_name)
                if not chat_id:
                    convos = reader.get_conversations(limit=5)
                    names = [c.display_name or c.participants[0] for c in convos if c.participants]
                    return (
                        f"I couldn't find a conversation with '{person_name}'. "
                        f"Recent conversations: {', '.join(names)}"
                    )
            else:
                # Use most recent conversation
                convos = reader.get_conversations(limit=1)
                if not convos:
                    return "No conversations found."
                chat_id = convos[0].chat_id
                person_name = convos[0].display_name or (
                    convos[0].participants[0] if convos[0].participants else "Unknown"
                )

            # Get conversation context
            context = fetcher.get_reply_context(chat_id, num_messages=20)

            if not context.last_received_message:
                return f"No recent messages from {person_name} to reply to."

            # Build prompt with RAG context
            formatted_prompt = build_reply_prompt(
                context=context.formatted_context,
                last_message=context.last_received_message.text,
            )

            # Generate with context
            generator = get_generator()
            request = GenerationRequest(
                prompt=formatted_prompt,
                context_documents=[context.formatted_context],
                few_shot_examples=REPLY_EXAMPLES,
                max_tokens=150,
                temperature=0.7,
            )
            response = generator.generate(request)

            # Format output nicely
            last_msg = context.last_received_message
            msg_text = last_msg.text
            text_preview = msg_text[:100] + "..." if len(msg_text) > 100 else msg_text
            return (
                f"[Replying to {person_name}]\n"
                f'Their message: "{text_preview}"\n\n'
                f"Suggested reply: {response.text}"
            )

    except Exception as e:
        logger.exception("Error generating reply")
        return f"Error generating reply: {e}"


def _handle_summarize_intent(prompt: str, intent_result: IntentResult) -> str:
    """Handle summarize intent with conversation context.

    Args:
        prompt: Original user input.
        intent_result: Classified intent with extracted params.

    Returns:
        Generated summary or error message.
    """
    from integrations.imessage import ChatDBReader
    from models import get_generator

    person_name = intent_result.extracted_params.get("person_name")

    if not _check_imessage_access():
        return (
            "I can't access your messages right now. "
            "Please grant Full Disk Access in System Settings > Privacy & Security."
        )

    try:
        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)

            # Find conversation
            if person_name:
                chat_id = fetcher.find_conversation_by_name(person_name)
                if not chat_id:
                    convos = reader.get_conversations(limit=5)
                    names = [c.display_name or c.participants[0] for c in convos if c.participants]
                    return (
                        f"I couldn't find a conversation with '{person_name}'. "
                        f"Recent conversations: {', '.join(names)}"
                    )
            else:
                convos = reader.get_conversations(limit=1)
                if not convos:
                    return "No conversations found."
                chat_id = convos[0].chat_id
                person_name = convos[0].display_name or (
                    convos[0].participants[0] if convos[0].participants else "Unknown"
                )

            # Get summary context (more messages than reply)
            context = fetcher.get_summary_context(chat_id, num_messages=50)

            if len(context.messages) < 3:
                return f"Not enough messages with {person_name} to summarize."

            # Build summary prompt
            formatted_prompt = build_summary_prompt(context=context.formatted_context)

            # Generate summary
            generator = get_generator()
            request = GenerationRequest(
                prompt=formatted_prompt,
                context_documents=[context.formatted_context],
                few_shot_examples=SUMMARY_EXAMPLES,
                max_tokens=500,
                temperature=0.5,
            )
            response = generator.generate(request)

            # Format output
            start_date = context.date_range[0].strftime("%b %d")
            end_date = context.date_range[1].strftime("%b %d")

            return (
                f"[Summary: {person_name}]\n"
                f"Period: {start_date} - {end_date} ({len(context.messages)} messages)\n\n"
                f"{response.text}"
            )

    except Exception as e:
        logger.exception("Error generating summary")
        return f"Error generating summary: {e}"


def _handle_search_intent(prompt: str, intent_result: IntentResult) -> str:
    """Handle search intent - find specific messages.

    Args:
        prompt: Original user input.
        intent_result: Classified intent with extracted params.

    Returns:
        Search results or error message.
    """
    from integrations.imessage import ChatDBReader

    search_query = intent_result.extracted_params.get("search_query") or prompt
    person_name = intent_result.extracted_params.get("person_name")

    if not _check_imessage_access():
        return "I can't access your messages. Please grant Full Disk Access."

    try:
        with ChatDBReader() as reader:
            results = reader.search(
                query=search_query,
                limit=10,
                sender=person_name,
            )

            if not results:
                return f"No messages found matching '{search_query}'."

            # Format results
            output_lines = [f"Found {len(results)} messages matching '{search_query}':\n"]
            for msg in results[:5]:
                sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
                date = msg.date.strftime("%b %d, %H:%M")
                text = msg.text[:80] + "..." if len(msg.text) > 80 else msg.text
                output_lines.append(f"[{date}] {sender}: {text}")

            if len(results) > 5:
                output_lines.append(f"\n... and {len(results) - 5} more")

            return "\n".join(output_lines)

    except Exception as e:
        logger.exception("Error searching messages")
        return f"Error searching: {e}"


def _handle_general_intent(prompt: str) -> str:
    """Handle general questions without message context.

    Args:
        prompt: User input.

    Returns:
        Generated response.
    """
    from models import get_generator

    generator = get_generator()
    request = GenerationRequest(
        prompt=prompt,
        context_documents=[],
        few_shot_examples=[],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=0.7,
    )
    response = generator.generate(request)
    return response.text


def _generate_response_with_intent(user_input: str) -> str:
    """Generate response using intent-based routing with RAG.

    Args:
        user_input: User's input text.

    Returns:
        Generated response.
    """
    classifier = get_intent_classifier()
    intent_result = classifier.classify(user_input)

    logger.debug(
        "Intent: %s (confidence: %.2f), params: %s",
        intent_result.intent.value,
        intent_result.confidence,
        intent_result.extracted_params,
    )

    # Route based on intent
    if intent_result.intent == IntentType.QUICK_REPLY:
        return _handle_quick_reply(user_input)

    elif intent_result.intent == IntentType.REPLY:
        return _handle_reply_intent(user_input, intent_result)

    elif intent_result.intent == IntentType.SUMMARIZE:
        return _handle_summarize_intent(user_input, intent_result)

    elif intent_result.intent == IntentType.SEARCH:
        return _handle_search_intent(user_input, intent_result)

    else:  # GENERAL
        return _handle_general_intent(user_input)


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat mode with intent-aware routing.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    console.print(
        Panel(
            "[bold green]JARVIS Chat[/bold green]\n"
            "Type your message and press Enter. Type 'quit' or 'exit' to leave.\n"
            "Try: 'reply to John', 'summarize my chat with Sarah', or ask anything!",
            title="Chat Mode",
        )
    )

    # Check memory mode
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()
    console.print(f"[dim]Operating in {state.current_mode.value} mode[/dim]\n")

    try:
        from models import get_generator

        # Verify generator is available
        get_generator()
    except ImportError:
        console.print("[red]Error: Model system not available.[/red]")
        return 1

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

        try:
            result = deg_controller.execute(
                FEATURE_CHAT,
                _generate_response_with_intent,
                user_input,
            )
            console.print(f"[bold green]JARVIS:[/bold green] {result}\n")
        except JarvisError as e:
            _format_jarvis_error(e)
            console.print()  # Extra newline for spacing
        except Exception as e:
            logger.exception("Chat error")
            console.print(f"[red]Error: {e}[/red]\n")

    return 0


def cmd_reply(args: argparse.Namespace) -> int:
    """Generate reply suggestions for a conversation.

    Args:
        args: Parsed arguments with 'person' and optional 'instruction'.

    Returns:
        Exit code.
    """
    from integrations.imessage import ChatDBReader
    from models import get_generator

    person = args.person
    instruction = getattr(args, "instruction", None)

    console.print(f"[bold]Generating reply for conversation with {person}...[/bold]\n")

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    try:
        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)

            chat_id = fetcher.find_conversation_by_name(person)
            if not chat_id:
                convos = reader.get_conversations(limit=5)
                console.print(f"[yellow]Could not find conversation with '{person}'[/yellow]")
                console.print("\nRecent conversations:")
                for c in convos:
                    name = c.display_name or (c.participants[0] if c.participants else "Unknown")
                    console.print(f"  - {name}")
                return 1

            context = fetcher.get_reply_context(chat_id, num_messages=20)

            if not context.last_received_message:
                console.print(f"[yellow]No recent messages from {person} to reply to.[/yellow]")
                return 1

            # Show what we're replying to
            last_msg = context.last_received_message
            console.print(f"[dim]Last message from {person}:[/dim]")
            console.print(f'  "{last_msg.text}"\n')

            # Build prompt
            prompt_instruction = instruction or "Generate a natural, friendly reply"
            formatted_prompt = build_reply_prompt(
                context=context.formatted_context,
                last_message=last_msg.text,
                instruction=prompt_instruction,
            )

            # Generate
            console.print("[dim]Generating suggestions...[/dim]")
            generator = get_generator()

            suggestions = []
            for i in range(3):  # Generate 3 options
                request = GenerationRequest(
                    prompt=formatted_prompt,
                    context_documents=[context.formatted_context],
                    few_shot_examples=REPLY_EXAMPLES,
                    max_tokens=150,
                    temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                )
                response = generator.generate(request)
                suggestions.append(response.text)

            # Display suggestions
            console.print("\n[bold green]Suggested replies:[/bold green]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"\n  {i}. {suggestion}")

            return 0

    except JarvisError as e:
        _format_jarvis_error(e)
        return 1
    except Exception as e:
        logger.exception("Error generating reply")
        console.print(f"[red]Error: {e}[/red]")
        return 1


def cmd_summarize(args: argparse.Namespace) -> int:
    """Summarize a conversation.

    Args:
        args: Parsed arguments with 'person' and optional 'messages' count.

    Returns:
        Exit code.
    """
    from integrations.imessage import ChatDBReader
    from models import get_generator

    person = args.person
    num_messages = args.messages

    console.print(f"[bold]Summarizing conversation with {person}...[/bold]\n")

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    try:
        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)

            chat_id = fetcher.find_conversation_by_name(person)
            if not chat_id:
                convos = reader.get_conversations(limit=5)
                console.print(f"[yellow]Could not find conversation with '{person}'[/yellow]")
                console.print("\nRecent conversations:")
                for c in convos:
                    name = c.display_name or (c.participants[0] if c.participants else "Unknown")
                    console.print(f"  - {name}")
                return 1

            context = fetcher.get_summary_context(chat_id, num_messages=num_messages)

            if len(context.messages) < 3:
                console.print(
                    f"[yellow]Not enough messages to summarize "
                    f"({len(context.messages)} found).[/yellow]"
                )
                return 1

            # Show context info
            start_date = context.date_range[0].strftime("%B %d, %Y")
            end_date = context.date_range[1].strftime("%B %d, %Y")
            console.print(
                f"[dim]Analyzing {len(context.messages)} messages "
                f"from {start_date} to {end_date}[/dim]\n"
            )

            # Generate summary
            console.print("[dim]Generating summary...[/dim]")
            formatted_prompt = build_summary_prompt(context=context.formatted_context)

            generator = get_generator()
            request = GenerationRequest(
                prompt=formatted_prompt,
                context_documents=[context.formatted_context],
                few_shot_examples=SUMMARY_EXAMPLES,
                max_tokens=500,
                temperature=0.5,
            )
            response = generator.generate(request)

            # Display summary
            console.print(
                Panel(
                    response.text,
                    title=f"Summary: {person}",
                    subtitle=f"{len(context.messages)} messages | {start_date} - {end_date}",
                )
            )

            return 0

    except JarvisError as e:
        _format_jarvis_error(e)
        return 1
    except Exception as e:
        logger.exception("Error generating summary")
        console.print(f"[red]Error: {e}[/red]")
        return 1


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
    except iMessageError as e:
        _format_jarvis_error(e)
        return 1
    except JarvisError as e:
        _format_jarvis_error(e)
        return 1
    except Exception as e:
        logger.exception("Search error")
        console.print(f"[red]Error searching messages: {e}[/red]")
        return 1


def cmd_search_semantic(args: argparse.Namespace) -> int:
    """Search messages using semantic similarity.

    Args:
        args: Parsed arguments with 'query', 'limit', 'threshold', etc.

    Returns:
        Exit code.
    """
    from jarvis.semantic_search import SearchFilters, SemanticSearcher

    query = args.query
    limit = args.limit
    threshold = args.threshold

    # Parse optional filter arguments
    start_date = _parse_date(args.start_date) if args.start_date else None
    end_date = _parse_date(args.end_date) if args.end_date else None
    sender = args.sender
    chat_id = args.chat_id

    console.print(f"[bold]Semantic search for:[/bold] {query}\n")

    # Show settings
    console.print(f"[dim]Similarity threshold: {threshold}[/dim]")
    if sender:
        console.print(f"[dim]Sender filter: {sender}[/dim]")
    if chat_id:
        console.print(f"[dim]Conversation filter: {chat_id}[/dim]")
    if start_date:
        console.print(f"[dim]After: {start_date.strftime('%Y-%m-%d')}[/dim]")
    if end_date:
        console.print(f"[dim]Before: {end_date.strftime('%Y-%m-%d')}[/dim]")
    console.print()

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            # Build filters
            filters = SearchFilters(
                sender=sender,
                chat_id=chat_id,
                after=start_date,
                before=end_date,
            )

            # Create searcher
            searcher = SemanticSearcher(
                reader=reader,
                similarity_threshold=threshold,
            )

            console.print("[dim]Computing semantic embeddings...[/dim]")
            results = searcher.search(
                query=query,
                filters=filters,
                limit=limit,
                index_limit=args.index_limit,
            )

            if not results:
                console.print("[yellow]No semantically similar messages found.[/yellow]")
                console.print("\n[dim]Tips:[/dim]")
                console.print("  - Try lowering the threshold with --threshold 0.2")
                console.print("  - Use different or simpler query terms")
                console.print("  - Remove filters to search more messages")
                return 0

            # Display results
            table = Table(title=f"Semantic Search Results ({len(results)} matches)")
            table.add_column("Score", style="cyan", width=6)
            table.add_column("Date", style="dim")
            table.add_column("Sender")
            table.add_column("Message")

            for result in results:
                msg = result.message
                date_str = msg.date.strftime("%Y-%m-%d %H:%M") if msg.date else "Unknown"
                display_sender = "Me" if msg.is_from_me else (msg.sender_name or msg.sender)
                score = f"{result.similarity:.2f}"

                if len(msg.text) > MESSAGE_PREVIEW_LENGTH:
                    text = msg.text[:MESSAGE_PREVIEW_LENGTH] + "..."
                else:
                    text = msg.text

                table.add_row(score, date_str, display_sender, text)

            console.print(table)
            return 0

    except JarvisError as e:
        _format_jarvis_error(e)
        return 1
    except Exception as e:
        logger.exception("Semantic search error")
        console.print(f"[red]Error during semantic search: {e}[/red]")
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


def cmd_export(args: argparse.Namespace) -> int:
    """Export a conversation to a file.

    Args:
        args: Parsed arguments with chat_id, format, and output options.

    Returns:
        Exit code.
    """
    from pathlib import Path

    from jarvis.export import ExportFormat, export_messages, get_export_filename

    chat_id = args.chat_id
    format_str = args.format.lower()
    output_path = args.output

    # Validate format
    try:
        export_format = ExportFormat(format_str)
    except ValueError:
        console.print(f"[red]Invalid format: {format_str}[/red]")
        console.print("Supported formats: json, csv, txt")
        return 1

    console.print(f"[bold]Exporting conversation: {chat_id}[/bold]\n")

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            # Get conversation metadata
            conversations = reader.get_conversations(limit=500)
            conversation = None
            for conv in conversations:
                if conv.chat_id == chat_id:
                    conversation = conv
                    break

            if conversation is None:
                console.print(f"[red]Conversation not found: {chat_id}[/red]")
                console.print("\nAvailable conversations:")
                for c in conversations[:10]:
                    name = c.display_name or (c.participants[0] if c.participants else "Unknown")
                    console.print(f"  {c.chat_id} - {name}")
                if len(conversations) > 10:
                    console.print(f"  ... and {len(conversations) - 10} more")
                return 1

            # Get messages
            limit = args.limit
            messages = reader.get_messages(chat_id=chat_id, limit=limit)

            if not messages:
                console.print("[yellow]No messages found in this conversation.[/yellow]")
                return 1

            # Export
            console.print(f"[dim]Exporting {len(messages)} messages...[/dim]")
            exported_data = export_messages(
                messages=messages,
                format=export_format,
                conversation=conversation,
                include_attachments=args.include_attachments,
            )

            # Determine output path
            if output_path is None:
                output_path = get_export_filename(
                    format=export_format,
                    prefix="conversation",
                    chat_id=chat_id,
                )

            # Write to file
            output_file = Path(output_path)
            output_file.write_text(exported_data, encoding="utf-8")

            console.print(f"\n[green]Successfully exported to: {output_file}[/green]")
            console.print(f"Format: {format_str.upper()}")
            console.print(f"Messages: {len(messages)}")

            return 0

    except Exception as e:
        logger.exception("Export error")
        console.print(f"[red]Error exporting conversation: {e}[/red]")
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Handle batch operations.

    Args:
        args: Parsed arguments with subcommand and options.

    Returns:
        Exit code.
    """
    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

    from jarvis.tasks import (
        TaskStatus,
        TaskType,
        get_task_queue,
        get_worker,
        start_worker,
    )

    subcommand = args.batch_command
    if subcommand is None:
        console.print("[red]Error: Please specify a batch subcommand[/red]")
        console.print("Available subcommands: export, summarize")
        return 1

    # Ensure worker is running
    worker = get_worker()
    if not worker.is_running:
        start_worker()

    queue = get_task_queue()

    if subcommand == "export":
        # Get chat IDs
        if args.all:
            # Export all conversations
            console.print("[bold]Exporting all conversations...[/bold]")
            if not _check_imessage_access():
                console.print(
                    "[red]Cannot access iMessage. Grant Full Disk Access in "
                    "System Settings > Privacy & Security.[/red]"
                )
                return 1

            from integrations.imessage import ChatDBReader

            with ChatDBReader() as reader:
                conversations = reader.get_conversations(limit=args.limit or 50)
                chat_ids = [c.chat_id for c in conversations]

            if not chat_ids:
                console.print("[yellow]No conversations found.[/yellow]")
                return 1

            console.print(f"Found {len(chat_ids)} conversations")
        elif args.chats:
            chat_ids = [c.strip() for c in args.chats.split(",")]
        else:
            console.print("[red]Error: Specify --all or --chats[/red]")
            return 1

        # Create the task
        task = queue.enqueue(
            task_type=TaskType.BATCH_EXPORT,
            params={
                "chat_ids": chat_ids,
                "format": args.format or "json",
                "output_dir": args.output_dir,
            },
        )
        console.print(f"[green]Task created: {task.id}[/green]")

    elif subcommand == "summarize":
        # Get chat IDs for summarization
        if args.recent:
            # Summarize recent conversations
            console.print(f"[bold]Summarizing {args.recent} recent conversations...[/bold]")
            if not _check_imessage_access():
                console.print(
                    "[red]Cannot access iMessage. Grant Full Disk Access in "
                    "System Settings > Privacy & Security.[/red]"
                )
                return 1

            from integrations.imessage import ChatDBReader

            with ChatDBReader() as reader:
                conversations = reader.get_conversations(limit=args.recent)
                chat_ids = [c.chat_id for c in conversations]

            if not chat_ids:
                console.print("[yellow]No conversations found.[/yellow]")
                return 1

            console.print(f"Found {len(chat_ids)} conversations")
        elif args.chats:
            chat_ids = [c.strip() for c in args.chats.split(",")]
        else:
            console.print("[red]Error: Specify --recent or --chats[/red]")
            return 1

        # Create the task
        task = queue.enqueue(
            task_type=TaskType.BATCH_SUMMARIZE,
            params={
                "chat_ids": chat_ids,
                "num_messages": args.messages or 50,
            },
        )
        console.print(f"[green]Task created: {task.id}[/green]")

    else:
        console.print(f"[red]Unknown batch subcommand: {subcommand}[/red]")
        return 1

    # Monitor progress if requested
    if not args.no_wait:
        console.print("[dim]Waiting for task to complete (use --no-wait to skip)...[/dim]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress_task: TaskID = progress.add_task(
                f"[cyan]{subcommand.capitalize()}...", total=100
            )

            import time

            while True:
                task = queue.get(task.id)
                if task is None:
                    console.print("[red]Task not found[/red]")
                    return 1

                progress.update(
                    progress_task,
                    completed=task.progress.percent,
                    description=f"[cyan]{task.progress.message or subcommand.capitalize()}",
                )

                if task.is_terminal:
                    break

                time.sleep(0.5)

        # Show final status
        if task.status == TaskStatus.COMPLETED:
            console.print("\n[green]Task completed successfully![/green]")
            if task.result:
                console.print(
                    f"Processed: {task.result.items_processed}, Failed: {task.result.items_failed}"
                )
        elif task.status == TaskStatus.FAILED:
            console.print(f"\n[red]Task failed: {task.error_message}[/red]")
            return 1
        else:
            console.print(f"\n[yellow]Task status: {task.status.value}[/yellow]")

    else:
        console.print(f"\nTask ID: {task.id}")
        console.print("Use 'jarvis tasks status <id>' to check progress")

    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    """Handle task management commands.

    Args:
        args: Parsed arguments with subcommand.

    Returns:
        Exit code.
    """
    from jarvis.tasks import TaskStatus, get_task_queue

    subcommand = args.tasks_command
    if subcommand is None:
        console.print("[red]Error: Please specify a tasks subcommand[/red]")
        console.print("Available subcommands: list, status, cancel")
        return 1

    queue = get_task_queue()

    if subcommand == "list":
        # List all tasks
        status_filter = None
        if args.status:
            try:
                status_filter = TaskStatus(args.status)
            except ValueError:
                console.print(f"[red]Invalid status: {args.status}[/red]")
                console.print("Valid statuses: pending, running, completed, failed, cancelled")
                return 1

        tasks = queue.get_all(status=status_filter, limit=args.limit or 20)

        if not tasks:
            console.print("[dim]No tasks found[/dim]")
            return 0

        table = Table(title="Tasks")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Progress")
        table.add_column("Created")

        status_colors = {
            TaskStatus.PENDING: "yellow",
            TaskStatus.RUNNING: "blue",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.CANCELLED: "dim",
        }

        for task in tasks:
            status_color = status_colors.get(task.status, "white")
            created = task.created_at.strftime("%Y-%m-%d %H:%M")
            progress = f"{task.progress.percent:.0f}%"

            table.add_row(
                task.id[:12],
                task.task_type.value,
                f"[{status_color}]{task.status.value}[/{status_color}]",
                progress,
                created,
            )

        console.print(table)

        # Show summary
        stats = queue.get_stats()
        console.print(f"\n[dim]Total: {stats['total']} tasks[/dim]")

    elif subcommand == "status":
        if not args.task_id:
            console.print("[red]Error: Please provide a task ID[/red]")
            return 1

        task = queue.get(args.task_id)
        if task is None:
            console.print(f"[red]Task not found: {args.task_id}[/red]")
            return 1

        # Show detailed status
        console.print(Panel(f"[bold]Task: {task.id}[/bold]", title="Task Details"))

        info_table = Table(show_header=False, box=None)
        info_table.add_column("Field", style="bold")
        info_table.add_column("Value")

        status_colors = {
            TaskStatus.PENDING: "yellow",
            TaskStatus.RUNNING: "blue",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.CANCELLED: "dim",
        }
        status_color = status_colors.get(task.status, "white")

        info_table.add_row("Type", task.task_type.value)
        info_table.add_row("Status", f"[{status_color}]{task.status.value}[/{status_color}]")
        info_table.add_row("Progress", f"{task.progress.percent:.1f}% - {task.progress.message}")
        info_table.add_row("Created", task.created_at.strftime("%Y-%m-%d %H:%M:%S"))

        if task.started_at:
            info_table.add_row("Started", task.started_at.strftime("%Y-%m-%d %H:%M:%S"))
        if task.completed_at:
            info_table.add_row("Completed", task.completed_at.strftime("%Y-%m-%d %H:%M:%S"))
        if task.duration_seconds:
            info_table.add_row("Duration", f"{task.duration_seconds:.2f}s")
        if task.error_message:
            info_table.add_row("Error", f"[red]{task.error_message}[/red]")
        if task.result:
            info_table.add_row(
                "Result",
                f"Processed: {task.result.items_processed}, Failed: {task.result.items_failed}",
            )

        console.print(info_table)

    elif subcommand == "cancel":
        if not args.task_id:
            console.print("[red]Error: Please provide a task ID[/red]")
            return 1

        task = queue.get(args.task_id)
        if task is None:
            console.print(f"[red]Task not found: {args.task_id}[/red]")
            return 1

        if task.status != TaskStatus.PENDING:
            console.print(f"[yellow]Cannot cancel task with status '{task.status.value}'[/yellow]")
            return 1

        success = queue.cancel(args.task_id)
        if success:
            console.print(f"[green]Task cancelled: {args.task_id}[/green]")
        else:
            console.print("[red]Failed to cancel task[/red]")
            return 1

    else:
        console.print(f"[red]Unknown tasks subcommand: {subcommand}[/red]")
        return 1

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
            "api.main:app",  # Full API with all endpoints for Tauri desktop app
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
        return 0
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        return 1


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    """Start the MCP (Model Context Protocol) server.

    Args:
        args: Parsed arguments with transport, host, and port options.

    Returns:
        Exit code.
    """
    import asyncio

    from mcp_server.server import MCPServer, StdioTransport, run_http_server

    transport = args.transport
    verbose = getattr(args, "verbose", False)

    if transport == "stdio":
        console.print(
            "[bold green]Starting JARVIS MCP Server (stdio mode)[/bold green]",
            file=sys.stderr,
        )
        console.print(
            "[dim]Communication via stdin/stdout - ready for Claude Code[/dim]",
            file=sys.stderr,
        )

        # Configure logging to stderr
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )

        server = MCPServer()
        stdio_transport = StdioTransport(server)
        try:
            stdio_transport.run()
            return 0
        except Exception as e:
            console.print(f"[red]Error running MCP server: {e}[/red]", file=sys.stderr)
            return 1
    else:
        # HTTP transport
        host = args.host
        port = args.port

        console.print(
            Panel(
                f"[bold green]Starting JARVIS MCP Server (HTTP mode)[/bold green]\n"
                f"Host: {host}\n"
                f"Port: {port}\n"
                f"Endpoint: http://{host}:{port}/mcp",
                title="MCP Server",
            )
        )

        try:
            asyncio.run(run_http_server(host, port))
            return 0
        except KeyboardInterrupt:
            console.print("\n[dim]MCP server stopped.[/dim]")
            return 0
        except Exception as e:
            console.print(f"[red]Error starting MCP server: {e}[/red]")
            return 1


def cmd_db(args: argparse.Namespace) -> int:
    """Handle database operations.

    Args:
        args: Parsed arguments with subcommand.

    Returns:
        Exit code.
    """
    subcommand = args.db_command
    if subcommand is None:
        console.print("[red]Error: Please specify a db subcommand[/red]")
        console.print(
            "Available: init, add-contact, list-contacts, extract, "
            "cluster, label-cluster, build-index, stats"
        )
        return 1

    if subcommand == "init":
        return _cmd_db_init(args)
    elif subcommand == "add-contact":
        return _cmd_db_add_contact(args)
    elif subcommand == "list-contacts":
        return _cmd_db_list_contacts(args)
    elif subcommand == "extract":
        return _cmd_db_extract(args)
    elif subcommand == "cluster":
        return _cmd_db_cluster(args)
    elif subcommand == "label-cluster":
        return _cmd_db_label_cluster(args)
    elif subcommand == "build-index":
        return _cmd_db_build_index(args)
    elif subcommand == "stats":
        return _cmd_db_stats(args)
    else:
        console.print(f"[red]Unknown db subcommand: {subcommand}[/red]")
        return 1


def _cmd_db_init(args: argparse.Namespace) -> int:
    """Initialize the JARVIS database."""

    console.print("[bold]Initializing JARVIS database...[/bold]")

    db = get_db()

    if db.exists() and not args.force:
        console.print(f"[yellow]Database already exists at {db.db_path}[/yellow]")
        console.print("Use --force to reinitialize")
        return 0

    created = db.init_schema()

    if created:
        console.print(f"[green]Database created at {db.db_path}[/green]")
    else:
        console.print(f"[green]Database already up to date at {db.db_path}[/green]")

    return 0


def _cmd_db_add_contact(args: argparse.Namespace) -> int:
    """Add or update a contact."""

    db = get_db()

    # Ensure database exists
    if not db.exists():
        db.init_schema()

    contact = db.add_contact(
        display_name=args.name,
        chat_id=args.chat_id,
        phone_or_email=args.phone,
        relationship=args.relationship,
        style_notes=args.style,
    )

    console.print("[green]Contact added/updated:[/green]")
    console.print(f"  Name: {contact.display_name}")
    if contact.relationship:
        console.print(f"  Relationship: {contact.relationship}")
    if contact.style_notes:
        console.print(f"  Style: {contact.style_notes}")
    if contact.chat_id:
        console.print(f"  Chat ID: {contact.chat_id}")

    return 0


def _cmd_db_list_contacts(args: argparse.Namespace) -> int:
    """List all contacts."""

    db = get_db()

    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    contacts = db.list_contacts(limit=args.limit)

    if not contacts:
        console.print("[dim]No contacts found. Add some with 'jarvis db add-contact'[/dim]")
        return 0

    table = Table(title=f"Contacts ({len(contacts)})")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Relationship")
    table.add_column("Style")
    table.add_column("Chat ID", style="dim")

    for c in contacts:
        table.add_row(
            str(c.id),
            c.display_name,
            c.relationship or "",
            c.style_notes or "",
            c.chat_id or "",
        )

    console.print(table)
    return 0


def _cmd_db_extract(args: argparse.Namespace) -> int:
    """Extract (trigger, response) pairs from iMessage history."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    db = get_db()
    if not db.exists():
        db.init_schema()
        console.print(f"[dim]Created database at {db.db_path}[/dim]")

    from integrations.imessage import ChatDBReader

    # Check if using v2 pipeline
    use_v2 = getattr(args, "v2", False)

    if use_v2:
        # V2 exchange-based extraction with validity gates
        from jarvis.extract import ExchangeBuilderConfig, extract_all_pairs_v2

        config = ExchangeBuilderConfig(
            time_gap_boundary_minutes=getattr(args, "time_gap", 30.0),
            context_window_size=getattr(args, "context_size", 20),
            max_response_delay_hours=args.max_delay,
        )

        # Optionally load embedder for Gate B
        embedder = None
        skip_nli = getattr(args, "skip_nli", False)
        nli_model = None

        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            console.print(f"[dim]Loaded embedder for Gate B (backend: {embedder.backend})[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not load embedder: {e}[/yellow]")
            console.print("[yellow]Gate B will be skipped[/yellow]")

        # Optionally load NLI model for Gate C
        if not skip_nli:
            try:
                from jarvis.validity_gate import load_nli_model

                nli_model = load_nli_model()
                if nli_model:
                    console.print("[dim]Loaded NLI model for Gate C[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load NLI model: {e}[/yellow]")

        console.print(
            "[bold]Extracting pairs using v2 pipeline (exchange-based with gates)...[/bold]\n"
        )

        with ChatDBReader() as reader:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing conversations...", total=None)

                def progress_cb(current: int, total: int, chat_id: str) -> None:
                    progress.update(
                        task,
                        description=f"Processing {current}/{total}: {chat_id[:30]}...",
                    )

                stats = extract_all_pairs_v2(
                    reader, db, config, embedder, nli_model, progress_cb, skip_nli
                )

        # Display v2 results
        console.print("\n[bold green]Extraction complete (v2 pipeline)![/bold green]")
        console.print(f"  Messages scanned: {stats['total_messages_scanned']}")
        console.print(f"  Exchanges built: {stats['exchanges_built']}")
        console.print(f"  Conversations processed: {stats['conversations_processed']}")
        console.print(f"  Pairs added: {stats['pairs_added']}")
        console.print(f"  Duplicates skipped: {stats['pairs_skipped_duplicate']}")

        # Gate statistics
        console.print("\n[bold]Validity Gate Results:[/bold]")
        console.print(f"  Gate A rejected: {stats['gate_a_rejected']}")
        console.print(f"  Gate B rejected: {stats['gate_b_rejected']}")
        console.print(f"  Gate C rejected: {stats['gate_c_rejected']}")
        console.print(f"  Final valid: {stats['final_valid']}")
        console.print(f"  Final invalid: {stats['final_invalid']}")
        console.print(f"  Final uncertain: {stats['final_uncertain']}")

        # Gate A rejection reasons
        gate_a_reasons = stats.get("gate_a_reasons", {})
        if gate_a_reasons:
            console.print("\n[dim]Gate A rejection reasons:[/dim]")
            for reason, count in sorted(gate_a_reasons.items(), key=lambda x: -x[1]):
                console.print(f"  {reason}: {count}")

    else:
        # V1 turn-based extraction (legacy)
        from jarvis.extract import ExtractionConfig, extract_all_pairs

        config = ExtractionConfig(
            min_trigger_length=args.min_length,
            min_response_length=args.min_length,
            max_response_delay_hours=args.max_delay,
        )

        console.print("[bold]Extracting (trigger, response) pairs from iMessage...[/bold]\n")

        with ChatDBReader() as reader:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing conversations...", total=None)

                def progress_cb(current: int, total: int, chat_id: str) -> None:
                    progress.update(
                        task,
                        description=f"Processing {current}/{total}: {chat_id[:30]}...",
                    )

                stats = extract_all_pairs(reader, db, config, progress_cb)

        # Display v1 results
        console.print("\n[bold green]Extraction complete![/bold green]")
        console.print(f"  Messages scanned: {stats.get('total_messages_scanned', 'N/A')}")
        console.print(f"  Turns identified: {stats.get('turns_identified', 'N/A')}")
        console.print(f"  Conversations processed: {stats['conversations_processed']}")
        console.print(f"  Candidate pairs: {stats.get('candidate_pairs', 'N/A')}")
        console.print(f"  Pairs extracted: {stats['pairs_extracted']}")
        console.print(f"  Pairs added to database: {stats['pairs_added']}")
        console.print(f"  Duplicates skipped: {stats['pairs_skipped_duplicate']}")

        # Show dropped reasons if available
        dropped = stats.get("dropped_by_reason", {})
        if dropped and any(v > 0 for v in dropped.values()):
            console.print("\n[dim]Dropped pairs by reason:[/dim]")
            for reason, count in dropped.items():
                if count > 0:
                    console.print(f"  {reason}: {count}")

    if stats.get("errors"):
        console.print(f"\n[yellow]Errors: {len(stats['errors'])}[/yellow]")

    return 0


def _cmd_db_cluster(args: argparse.Namespace) -> int:
    """Cluster response patterns using HDBSCAN."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from jarvis.cluster import ClusterConfig, cluster_and_store

    db = get_db()
    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    pair_count = db.count_pairs()
    if pair_count == 0:
        console.print("[yellow]No pairs found. Run 'jarvis db extract' first.[/yellow]")
        return 1

    console.print(f"[bold]Clustering {pair_count} response patterns...[/bold]\n")

    config = ClusterConfig(
        min_cluster_size=args.min_size,
        min_samples=args.min_samples,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)

        def progress_cb(stage: str, pct: float, msg: str) -> None:
            progress.update(task, description=msg)

        try:
            stats = cluster_and_store(db, config, progress_cb)
        except ImportError as e:
            console.print("\n[yellow]Optional dependency not installed.[/yellow]")
            console.print(f"\n{e}")
            console.print(
                "\n[dim]Note: Clustering is optional. The router uses top-K selection "
                "which works without clusters.[/dim]"
            )
            return 1

    # Display results
    console.print("\n[bold green]Clustering complete![/bold green]")
    console.print(f"  Pairs processed: {stats['pairs_processed']}")
    console.print(f"  Clusters found: {stats['clusters_found']}")
    console.print(f"  Noise (unclustered): {stats['noise_pairs']}")

    if stats["clusters_created"]:
        console.print("\n[bold]Clusters:[/bold]")
        for cluster in stats["clusters_created"]:
            console.print(f"  {cluster['name']}: {cluster['size']} responses")

    console.print("\n[dim]Use 'jarvis db label-cluster <id> <name>' to rename clusters[/dim]")

    return 0


def _cmd_db_label_cluster(args: argparse.Namespace) -> int:
    """Label a cluster with a name."""

    db = get_db()
    if not db.exists():
        console.print("[yellow]Database not initialized.[/yellow]")
        return 1

    cluster = db.get_cluster(args.cluster_id)
    if not cluster:
        console.print(f"[red]Cluster {args.cluster_id} not found[/red]")
        console.print("\nAvailable clusters:")
        for c in db.list_clusters():
            console.print(f"  {c.id}: {c.name}")
        return 1

    old_name = cluster.name
    db.update_cluster_label(args.cluster_id, args.name, args.description)

    console.print(f"[green]Renamed cluster {args.cluster_id}:[/green]")
    console.print(f"  {old_name} → {args.name}")

    return 0


def _cmd_db_build_index(args: argparse.Namespace) -> int:
    """Build FAISS index of triggers."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from jarvis.index import build_index_from_db

    db = get_db()
    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    pair_count = db.count_pairs()
    if pair_count == 0:
        console.print("[yellow]No pairs found. Run 'jarvis db extract' first.[/yellow]")
        return 1

    console.print(f"[bold]Building FAISS index for {pair_count} triggers...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)

        def progress_cb(stage: str, pct: float, msg: str) -> None:
            progress.update(task, description=msg)

        result = build_index_from_db(db, None, progress_cb)

    if not result["success"]:
        console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
        return 1

    console.print("\n[bold green]Index built successfully![/bold green]")
    console.print(f"  Triggers indexed: {result['pairs_indexed']}")
    console.print(f"  Embedding dimension: {result['dimension']}")
    console.print(f"  Index size: {result['index_size_bytes'] / 1024:.1f} KB")
    console.print(f"  Index path: {result['index_path']}")

    return 0


def _cmd_db_stats(args: argparse.Namespace) -> int:
    """Show database statistics."""
    from jarvis.index import get_index_stats

    db = get_db()

    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    stats = db.get_stats()

    console.print(Panel("[bold]JARVIS Database Statistics[/bold]", title="Stats"))

    # Overview table
    overview = Table(title="Overview")
    overview.add_column("Metric", style="bold")
    overview.add_column("Count")

    overview.add_row("Contacts", str(stats["contacts"]))
    overview.add_row("Pairs (total)", str(stats["pairs"]))
    overview.add_row("Pairs (quality >= 0.5)", str(stats.get("pairs_quality_gte_50", "N/A")))
    overview.add_row("Clusters", str(stats["clusters"]))
    overview.add_row("Embeddings", str(stats["embeddings"]))

    console.print(overview)

    # Pairs per contact
    if stats["pairs_per_contact"]:
        console.print("\n[bold]Top Contacts by Pairs:[/bold]")
        for item in stats["pairs_per_contact"][:5]:
            if item["count"] > 0:
                console.print(f"  {item['name']}: {item['count']} pairs")

    # Gate breakdown (if --gate-breakdown flag)
    if getattr(args, "gate_breakdown", False):
        gate_stats = db.get_gate_stats()
        if gate_stats.get("total_gated", 0) > 0:
            console.print("\n[bold]Validity Gate Breakdown:[/bold]")
            console.print(f"  Total gated pairs: {gate_stats['total_gated']}")
            console.print(f"  Valid: {gate_stats.get('status_valid', 0)}")
            console.print(f"  Invalid: {gate_stats.get('status_invalid', 0)}")
            console.print(f"  Uncertain: {gate_stats.get('status_uncertain', 0)}")

            if gate_stats.get("gate_a_rejected"):
                console.print(f"\n  Gate A rejected: {gate_stats['gate_a_rejected']}")
                if gate_stats.get("gate_a_reasons"):
                    console.print("  [dim]Gate A rejection reasons:[/dim]")
                    for reason, count in sorted(
                        gate_stats["gate_a_reasons"].items(), key=lambda x: -x[1]
                    ):
                        console.print(f"    {reason}: {count}")

            if gate_stats.get("gate_b_bands"):
                console.print("\n  [dim]Gate B bands:[/dim]")
                for band, count in gate_stats["gate_b_bands"].items():
                    console.print(f"    {band}: {count}")

            if gate_stats.get("gate_c_verdicts"):
                console.print("\n  [dim]Gate C verdicts:[/dim]")
                for verdict, count in gate_stats["gate_c_verdicts"].items():
                    console.print(f"    {verdict}: {count}")
        else:
            console.print("\n[dim]No pairs with gate data. Use --v2 extraction.[/dim]")

    # Index stats (pass db for versioned index support)
    index_stats = get_index_stats(db)
    if index_stats and index_stats.get("exists"):
        console.print("\n[bold]FAISS Index:[/bold]")
        console.print(f"  Version: {index_stats.get('version_id', 'N/A')}")
        console.print(f"  Model: {index_stats.get('model_name', 'N/A')}")
        console.print(f"  Vectors: {index_stats['num_vectors']}")
        console.print(f"  Dimension: {index_stats['dimension']}")
        console.print(f"  Size: {index_stats['size_bytes'] / 1024:.1f} KB")
        if index_stats.get("created_at"):
            console.print(f"  Created: {index_stats['created_at']}")
    else:
        console.print("\n[dim]FAISS index not built. Run 'jarvis db build-index'[/dim]")

    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    """Display detailed usage examples.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from jarvis.cli_examples import get_examples_text

    console.print(get_examples_text())
    return 0


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom help formatter with improved argument formatting."""

    def __init__(self, prog: str) -> None:
        """Initialize formatter with wider help text."""
        super().__init__(prog, max_help_position=30, width=100)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description=(
            "JARVIS - Local-first AI assistant for macOS\n\n"
            "An intelligent iMessage management system powered by MLX-based language models.\n"
            "Runs entirely on Apple Silicon with no cloud data transmission."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Quick Start:
  jarvis chat                      Start interactive chat session
  jarvis search-messages "dinner"  Search for messages about dinner
  jarvis reply John                Generate reply suggestions for John
  jarvis summarize Mom             Get summary of conversation with Mom
  jarvis health                    Check system health status

Export Commands:
  jarvis export --chat-id <id>     Export conversation to JSON
  jarvis export --chat-id <id> -f csv  Export to CSV format

Advanced Commands:
  jarvis reply Sarah -i "say yes"  Reply with specific instruction
  jarvis summarize Dad -n 100      Include last 100 messages in summary
  jarvis serve --port 3000         Start API server on port 3000
  jarvis benchmark memory          Run memory profiling benchmark

For detailed documentation, run: jarvis --examples
Or see: docs/CLI_GUIDE.md

Permissions:
  iMessage access requires Full Disk Access permission.
  Grant in: System Settings > Privacy & Security > Full Disk Access
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging for troubleshooting",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="show version information and exit",
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="show detailed usage examples for all commands",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Use 'jarvis <command> --help' for more information on a specific command.",
        metavar="<command>",
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="start interactive chat mode",
        description=(
            "Start an interactive chat session with JARVIS.\n\n"
            "The assistant uses intent-aware routing to understand your requests:\n"
            "  - Ask to reply to someone: 'reply to John'\n"
            "  - Request summaries: 'summarize my chat with Sarah'\n"
            "  - Search messages: 'find messages about dinner'\n"
            "  - General questions: anything else!\n\n"
            "Type 'quit', 'exit', or 'q' to leave chat mode.\n"
            "Press Ctrl+C to interrupt and exit."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis chat                      Start interactive chat
  jarvis -v chat                   Chat with debug logging enabled

Interactive Commands:
  You: reply to John               Generate reply suggestions
  You: summarize chat with Mom     Get conversation summary
  You: find messages about dinner  Search messages
  You: quit                        Exit chat mode
        """,
    )
    chat_parser.set_defaults(func=cmd_chat)

    # Reply command
    reply_parser = subparsers.add_parser(
        "reply",
        help="generate reply suggestions for a conversation",
        description=(
            "Generate intelligent reply suggestions for a conversation.\n\n"
            "Analyzes the conversation context and the last message received\n"
            "to suggest appropriate responses. Generates 3 different suggestions\n"
            "with varying tones for you to choose from.\n\n"
            "Requires Full Disk Access for iMessage access."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis reply John                Generate reply suggestions for John
  jarvis reply Sarah -i "say yes"  Reply agreeing to something
  jarvis reply Mom -i "decline politely"
                                   Reply declining a request
  jarvis reply Boss --instruction "confirm attendance"
                                   Reply confirming something

Tips:
  - The person name is matched against your iMessage conversations
  - Use partial names if full names don't match
  - Instructions help guide the tone and content of suggestions
        """,
    )
    reply_parser.add_argument(
        "person",
        metavar="<person>",
        help="name of the person to reply to (matched against conversations)",
    )
    reply_parser.add_argument(
        "-i",
        "--instruction",
        metavar="<text>",
        help="optional instruction to guide the reply tone/content",
    )
    reply_parser.set_defaults(func=cmd_reply)

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="summarize a conversation",
        description=(
            "Generate a summary of a conversation with a specific person.\n\n"
            "Analyzes recent messages to identify key topics, decisions,\n"
            "and important information from the conversation.\n\n"
            "Requires Full Disk Access for iMessage access."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis summarize John            Summarize last 50 messages with John
  jarvis summarize Sarah -n 100    Include last 100 messages
  jarvis summarize Mom --messages 20
                                   Quick summary of recent messages

Output includes:
  - Date range of messages analyzed
  - Total message count
  - Key topics and decisions
  - Action items (if any)
        """,
    )
    summarize_parser.add_argument(
        "person",
        metavar="<person>",
        help="name of the person/conversation to summarize",
    )
    summarize_parser.add_argument(
        "-n",
        "--messages",
        type=int,
        default=50,
        metavar="<count>",
        help="number of messages to include (default: 50)",
    )
    summarize_parser.set_defaults(func=cmd_summarize)

    # Search messages command
    search_parser = subparsers.add_parser(
        "search-messages",
        help="search iMessage conversations",
        description=(
            "Search through your iMessage conversations with powerful filtering.\n\n"
            "Supports full-text search across all conversations with optional\n"
            "filters for date range, sender, and attachments.\n\n"
            "Date formats: YYYY-MM-DD or 'YYYY-MM-DD HH:MM'\n\n"
            "Requires Full Disk Access for iMessage access."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis search-messages "dinner"
      Search for messages containing "dinner"

  jarvis search-messages "meeting" --limit 50
      Get up to 50 results

  jarvis search-messages "project" --sender "John"
      Search messages from John about "project"

  jarvis search-messages "photo" --has-attachment
      Find messages with attachments mentioning "photo"

  jarvis search-messages "birthday" --start-date 2024-06-01
      Search messages after June 1, 2024

  jarvis search-messages "I'll be there" --sender me
      Search your own messages

  jarvis search-messages "meeting" --start-date 2024-01-01 --end-date 2024-01-31
      Search within a specific date range

Date Formats:
  YYYY-MM-DD              e.g., 2024-01-15
  YYYY-MM-DD HH:MM        e.g., "2024-01-15 14:30" (quote for spaces)
        """,
    )
    search_parser.add_argument(
        "query",
        metavar="<query>",
        help="search query string",
    )
    search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=20,
        metavar="<n>",
        help="maximum number of results (default: 20)",
    )
    search_parser.add_argument(
        "--start-date",
        dest="start_date",
        metavar="<date>",
        help="filter messages after this date (YYYY-MM-DD)",
    )
    search_parser.add_argument(
        "--end-date",
        dest="end_date",
        metavar="<date>",
        help="filter messages before this date (YYYY-MM-DD)",
    )
    search_parser.add_argument(
        "--sender",
        metavar="<name>",
        help="filter by sender (use 'me' for your own messages)",
    )
    search_parser.add_argument(
        "--has-attachment",
        dest="has_attachment",
        action="store_true",
        default=None,
        help="show only messages with attachments",
    )
    search_parser.add_argument(
        "--no-attachment",
        dest="has_attachment",
        action="store_false",
        help="show only messages without attachments",
    )
    search_parser.set_defaults(func=cmd_search_messages)

    # Semantic search command
    semantic_search_parser = subparsers.add_parser(
        "search-semantic",
        help="semantic search using AI embeddings",
        description=(
            "Search through your iMessage conversations using semantic similarity.\n\n"
            "Unlike keyword search, semantic search finds messages by meaning,\n"
            "allowing you to find conceptually similar messages even if they don't\n"
            "contain the exact search terms.\n\n"
            "Uses the all-MiniLM-L6-v2 sentence embedding model.\n\n"
            "Requires Full Disk Access for iMessage access."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis search-semantic "dinner plans"
      Find messages about eating out, restaurants, meal planning

  jarvis search-semantic "meeting tomorrow" --limit 30
      Get up to 30 results about scheduling

  jarvis search-semantic "running late" --threshold 0.5
      Higher threshold = more relevant results only

  jarvis search-semantic "project deadline" --sender "John"
      Search messages from John about project deadlines

  jarvis search-semantic "vacation ideas" --start-date 2024-01-01
      Search messages after January 1, 2024

  jarvis search-semantic "thank you" --chat-id "chat123"
      Search within a specific conversation

Threshold Guide:
  0.3 (default)  - Broad search, more results
  0.4-0.5        - Balanced relevance
  0.6+           - Strict, highly relevant only
        """,
    )
    semantic_search_parser.add_argument(
        "query",
        metavar="<query>",
        help="natural language search query",
    )
    semantic_search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=20,
        metavar="<n>",
        help="maximum number of results (default: 20)",
    )
    semantic_search_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        metavar="<score>",
        help="minimum similarity score 0.0-1.0 (default: 0.3)",
    )
    semantic_search_parser.add_argument(
        "--index-limit",
        dest="index_limit",
        type=int,
        default=1000,
        metavar="<n>",
        help="maximum messages to search through (default: 1000)",
    )
    semantic_search_parser.add_argument(
        "--start-date",
        dest="start_date",
        metavar="<date>",
        help="filter messages after this date (YYYY-MM-DD)",
    )
    semantic_search_parser.add_argument(
        "--end-date",
        dest="end_date",
        metavar="<date>",
        help="filter messages before this date (YYYY-MM-DD)",
    )
    semantic_search_parser.add_argument(
        "--sender",
        metavar="<name>",
        help="filter by sender (use 'me' for your own messages)",
    )
    semantic_search_parser.add_argument(
        "--chat-id",
        dest="chat_id",
        metavar="<id>",
        help="filter to a specific conversation",
    )
    semantic_search_parser.set_defaults(func=cmd_search_semantic)

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="show system health status",
        description=(
            "Display comprehensive system health information.\n\n"
            "Shows:\n"
            "  - Memory status (available, used, operating mode)\n"
            "  - Feature availability (chat, iMessage access)\n"
            "  - Model status (loaded, memory usage)\n"
            "  - Permission status for iMessage\n\n"
            "Use this to diagnose issues or verify system readiness."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis health                    Show all health information
  jarvis -v health                 Include debug information

Operating Modes:
  FULL      - All features available (8GB+ RAM)
  LITE      - Reduced context windows (4-8GB RAM)
  MINIMAL   - Basic functionality only (<4GB RAM)

Feature States:
  HEALTHY   - Feature is working normally
  DEGRADED  - Feature works with reduced capability
  FAILED    - Feature is unavailable
        """,
    )
    health_parser.set_defaults(func=cmd_health)

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="run performance benchmarks",
        description=(
            "Run various performance benchmarks to evaluate system capabilities.\n\n"
            "Available benchmark types:\n"
            "  memory   - Profile model memory usage\n"
            "             Gate: Pass <5.5GB, Fail >6.5GB\n\n"
            "  latency  - Measure response times (cold/warm/hot start)\n"
            "             Gate: Pass warm <3s, cold <15s\n\n"
            "  hhem     - Evaluate hallucination scores using HHEM model\n"
            "             Gate: Pass mean >=0.5\n\n"
            "Note: Memory and latency benchmarks require Apple Silicon (MLX)."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis benchmark memory          Run memory profiling benchmark
  jarvis benchmark latency         Measure response latencies
  jarvis benchmark hhem            Evaluate hallucination rates
  jarvis benchmark memory -o results.json
                                   Save results to JSON file

Output:
  Results are displayed in the terminal and optionally saved to JSON.
  JSON output includes detailed metrics for automated analysis.
        """,
    )
    bench_parser.add_argument(
        "type",
        choices=["memory", "latency", "hhem"],
        metavar="<type>",
        help="benchmark type: memory, latency, or hhem",
    )
    bench_parser.add_argument(
        "-o",
        "--output",
        metavar="<file>",
        help="output file for results (JSON format)",
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    # Version command (also accessible via --version)
    version_parser = subparsers.add_parser(
        "version",
        help="show version information",
        description="Display JARVIS version information.",
        formatter_class=HelpFormatter,
    )
    version_parser.set_defaults(func=cmd_version)

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export a conversation to a file",
    )
    export_parser.add_argument(
        "--chat-id",
        dest="chat_id",
        required=True,
        help="Conversation ID to export",
    )
    export_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: auto-generated)",
    )
    export_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=1000,
        help="Maximum messages to export (default: 1000)",
    )
    export_parser.add_argument(
        "--include-attachments",
        dest="include_attachments",
        action="store_true",
        default=False,
        help="Include attachment info in export (CSV only)",
    )
    export_parser.set_defaults(func=cmd_export)

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch operations on multiple conversations",
        description=(
            "Run batch operations for bulk processing.\n\n"
            "Supports:\n"
            "  - Export multiple conversations\n"
            "  - Summarize multiple conversations\n\n"
            "Operations run as background tasks with progress tracking."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis batch export --all
      Export all conversations (up to 50)

  jarvis batch export --all --limit 100 --format csv
      Export up to 100 conversations as CSV

  jarvis batch export --chats chat1,chat2,chat3
      Export specific conversations

  jarvis batch summarize --recent 10
      Summarize 10 most recent conversations

  jarvis batch summarize --chats chat1,chat2 --messages 100
      Summarize specific conversations with 100 messages each
        """,
    )
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command")

    # batch export subcommand
    batch_export_parser = batch_subparsers.add_parser(
        "export",
        help="Export multiple conversations",
    )
    batch_export_parser.add_argument(
        "--all",
        action="store_true",
        help="Export all conversations",
    )
    batch_export_parser.add_argument(
        "--chats",
        metavar="<ids>",
        help="Comma-separated list of chat IDs to export",
    )
    batch_export_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        metavar="<n>",
        help="Maximum conversations to export when using --all (default: 50)",
    )
    batch_export_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="Export format (default: json)",
    )
    batch_export_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        metavar="<dir>",
        help="Output directory for export files",
    )
    batch_export_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for completion (return task ID immediately)",
    )

    # batch summarize subcommand
    batch_summarize_parser = batch_subparsers.add_parser(
        "summarize",
        help="Summarize multiple conversations",
    )
    batch_summarize_parser.add_argument(
        "--recent",
        type=int,
        metavar="<n>",
        help="Summarize N most recent conversations",
    )
    batch_summarize_parser.add_argument(
        "--chats",
        metavar="<ids>",
        help="Comma-separated list of chat IDs to summarize",
    )
    batch_summarize_parser.add_argument(
        "-m",
        "--messages",
        type=int,
        metavar="<n>",
        help="Number of messages to include in each summary (default: 50)",
    )
    batch_summarize_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for completion (return task ID immediately)",
    )

    batch_parser.set_defaults(func=cmd_batch)

    # Tasks command
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="Manage background tasks",
        description=(
            "Manage and monitor background tasks.\n\n"
            "Commands:\n"
            "  list     - List all tasks\n"
            "  status   - Get detailed task status\n"
            "  cancel   - Cancel a pending task"
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis tasks list
      List all tasks

  jarvis tasks list --status running
      List only running tasks

  jarvis tasks status abc123
      Get detailed status for task abc123

  jarvis tasks cancel abc123
      Cancel pending task abc123
        """,
    )
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command")

    # tasks list subcommand
    tasks_list_parser = tasks_subparsers.add_parser(
        "list",
        help="List all tasks",
    )
    tasks_list_parser.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed", "cancelled"],
        help="Filter by status",
    )
    tasks_list_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=20,
        metavar="<n>",
        help="Maximum tasks to show (default: 20)",
    )

    # tasks status subcommand
    tasks_status_parser = tasks_subparsers.add_parser(
        "status",
        help="Get detailed task status",
    )
    tasks_status_parser.add_argument(
        "task_id",
        metavar="<id>",
        help="Task ID to check",
    )

    # tasks cancel subcommand
    tasks_cancel_parser = tasks_subparsers.add_parser(
        "cancel",
        help="Cancel a pending task",
    )
    tasks_cancel_parser.add_argument(
        "task_id",
        metavar="<id>",
        help="Task ID to cancel",
    )

    tasks_parser.set_defaults(func=cmd_tasks)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="start the API server",
        description=(
            "Start the FastAPI REST server for external integration.\n\n"
            "The API server provides endpoints for:\n"
            "  - Chat functionality\n"
            "  - Message search\n"
            "  - Health monitoring\n\n"
            "Used by the Tauri desktop application and can be used\n"
            "for custom integrations.\n\n"
            "API documentation is available at /docs when running."
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis serve                     Start on localhost:8000
  jarvis serve --host 0.0.0.0      Listen on all interfaces
  jarvis serve -p 3000             Use port 3000
  jarvis serve --reload            Enable auto-reload for development
  jarvis serve --host 0.0.0.0 --port 8080
                                   Production setup

API Endpoints (when running):
  GET  /health            Health check
  POST /chat              Send chat messages
  GET  /messages/search   Search messages
  GET  /docs              Interactive API documentation
        """,
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="<addr>",
        help="host address to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        metavar="<port>",
        help="port number to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="enable auto-reload for development",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # MCP Serve command
    mcp_serve_parser = subparsers.add_parser(
        "mcp-serve",
        help="start the MCP (Model Context Protocol) server",
        description=(
            "Start the MCP server for Claude Code integration.\n\n"
            "The MCP server exposes JARVIS functionality as tools that can be\n"
            "used by Claude Code or other MCP-compatible clients.\n\n"
            "Transport modes:\n"
            "  stdio  - Communicate via stdin/stdout (default, for Claude Code)\n"
            "  http   - Communicate via HTTP endpoint (for network access)\n\n"
            "Available tools:\n"
            "  - search_messages: Search iMessage conversations\n"
            "  - get_summary: Get AI-generated conversation summary\n"
            "  - generate_reply: Generate reply suggestions\n"
            "  - get_contact_info: Retrieve contact information\n"
            "  - list_conversations: List recent conversations\n"
            "  - get_conversation_messages: Get messages from a conversation"
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis mcp-serve                   Start MCP server (stdio mode for Claude Code)
  jarvis mcp-serve --transport http  Start MCP server on HTTP
  jarvis mcp-serve --transport http --port 9000
                                     Use custom port for HTTP mode

Claude Code Integration:
  Add to your Claude Code settings (~/.claude/claude_desktop_config.json):

  {
    "mcpServers": {
      "jarvis": {
        "command": "jarvis",
        "args": ["mcp-serve"]
      }
    }
  }

For detailed documentation, see: docs/MCP_INTEGRATION.md
        """,
    )
    mcp_serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        metavar="<mode>",
        help="transport mode: stdio (default) or http",
    )
    mcp_serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="<addr>",
        help="host address for HTTP transport (default: 127.0.0.1)",
    )
    mcp_serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8765,
        metavar="<port>",
        help="port number for HTTP transport (default: 8765)",
    )
    mcp_serve_parser.set_defaults(func=cmd_mcp_serve)

    # Database command
    db_parser = subparsers.add_parser(
        "db",
        help="manage JARVIS database (contacts, pairs, clusters, index)",
        description=(
            "Manage the JARVIS database for personalized responses.\n\n"
            "The JARVIS database stores:\n"
            "  - Contacts with relationship labels and style notes\n"
            "  - (trigger, response) pairs extracted from your iMessage history\n"
            "  - Intent clusters discovered from your response patterns\n"
            "  - FAISS index for fast semantic search\n\n"
            "Workflow:\n"
            "  1. jarvis db init           - Create the database\n"
            "  2. jarvis db add-contact    - Add contacts with relationship info\n"
            "  3. jarvis db extract        - Extract pairs from iMessage\n"
            "  4. jarvis db cluster        - Cluster response patterns\n"
            "  5. jarvis db build-index    - Build FAISS search index"
        ),
        formatter_class=HelpFormatter,
        epilog="""
Examples:
  jarvis db init                         Initialize database
  jarvis db add-contact --name "Sarah" --relationship sister
                                         Add contact with relationship
  jarvis db list-contacts                View all contacts
  jarvis db extract                      Extract pairs from iMessage
  jarvis db cluster                      Cluster responses
  jarvis db label-cluster 1 GREETING     Rename a cluster
  jarvis db build-index                  Build FAISS index
  jarvis db stats                        Show database statistics
        """,
    )
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    # db init
    db_init_parser = db_subparsers.add_parser(
        "init",
        help="initialize the JARVIS database",
    )
    db_init_parser.add_argument(
        "--force",
        action="store_true",
        help="force reinitialization if database exists",
    )

    # db add-contact
    db_add_contact_parser = db_subparsers.add_parser(
        "add-contact",
        help="add or update a contact",
    )
    db_add_contact_parser.add_argument(
        "--name",
        required=True,
        metavar="<name>",
        help="contact display name",
    )
    db_add_contact_parser.add_argument(
        "--relationship",
        metavar="<type>",
        help="relationship type (e.g., sister, coworker, boss, friend)",
    )
    db_add_contact_parser.add_argument(
        "--style",
        metavar="<notes>",
        help="communication style notes (e.g., 'casual, uses emojis')",
    )
    db_add_contact_parser.add_argument(
        "--phone",
        metavar="<number>",
        help="phone number or email address",
    )
    db_add_contact_parser.add_argument(
        "--chat-id",
        dest="chat_id",
        metavar="<id>",
        help="iMessage chat ID to link",
    )

    # db list-contacts
    db_list_contacts_parser = db_subparsers.add_parser(
        "list-contacts",
        help="list all contacts",
    )
    db_list_contacts_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=100,
        metavar="<n>",
        help="maximum contacts to show (default: 100)",
    )

    # db extract
    db_extract_parser = db_subparsers.add_parser(
        "extract",
        help="extract (trigger, response) pairs from iMessage",
    )
    db_extract_parser.add_argument(
        "--min-length",
        dest="min_length",
        type=int,
        default=2,
        metavar="<n>",
        help="minimum message length in characters (default: 2)",
    )
    db_extract_parser.add_argument(
        "--max-delay",
        dest="max_delay",
        type=float,
        default=1.0,
        metavar="<hours>",
        help="maximum hours between trigger and response (default: 1.0)",
    )
    db_extract_parser.add_argument(
        "--v2",
        action="store_true",
        help="use v2 exchange-based extraction with validity gates",
    )
    db_extract_parser.add_argument(
        "--time-gap",
        dest="time_gap",
        type=float,
        default=30.0,
        metavar="<minutes>",
        help="time gap boundary for new conversation (v2 only, default: 30)",
    )
    db_extract_parser.add_argument(
        "--context-size",
        dest="context_size",
        type=int,
        default=20,
        metavar="<n>",
        help="context window size in messages (v2 only, default: 20)",
    )
    db_extract_parser.add_argument(
        "--skip-nli",
        dest="skip_nli",
        action="store_true",
        help="skip Gate C NLI validation (v2 only, faster but less accurate)",
    )

    # db cluster
    db_cluster_parser = db_subparsers.add_parser(
        "cluster",
        help="[optional] cluster response patterns using HDBSCAN (requires: pip install hdbscan)",
    )
    db_cluster_parser.add_argument(
        "--min-size",
        dest="min_size",
        type=int,
        default=10,
        metavar="<n>",
        help="minimum cluster size (default: 10)",
    )
    db_cluster_parser.add_argument(
        "--min-samples",
        dest="min_samples",
        type=int,
        default=5,
        metavar="<n>",
        help="minimum samples for core points (default: 5)",
    )

    # db label-cluster
    db_label_cluster_parser = db_subparsers.add_parser(
        "label-cluster",
        help="label a cluster with a name",
    )
    db_label_cluster_parser.add_argument(
        "cluster_id",
        type=int,
        metavar="<id>",
        help="cluster ID to label",
    )
    db_label_cluster_parser.add_argument(
        "name",
        metavar="<name>",
        help="new name for the cluster (e.g., GREETING, ACCEPT_INVITATION)",
    )
    db_label_cluster_parser.add_argument(
        "--description",
        metavar="<text>",
        help="optional description for the cluster",
    )

    # db build-index
    db_subparsers.add_parser(
        "build-index",
        help="build FAISS index of triggers",
    )

    # db stats
    db_stats_parser = db_subparsers.add_parser(
        "stats",
        help="show database statistics",
    )
    db_stats_parser.add_argument(
        "--gate-breakdown",
        dest="gate_breakdown",
        action="store_true",
        help="show validity gate statistics (for v2 extracted pairs)",
    )

    db_parser.set_defaults(func=cmd_db)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code.
    """
    parser = create_parser()

    # Enable shell completion if argcomplete is available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)

    # Handle --version flag
    if args.version:
        return cmd_version(args)

    # Handle --examples flag
    if args.examples:
        return cmd_examples(args)

    # Setup logging
    setup_logging(args.verbose)

    # Initialize system
    success, warnings = initialize_system()
    if not success:
        console.print("[red]Failed to initialize JARVIS system.[/red]")
        console.print("\n[dim]Troubleshooting tips:[/dim]")
        console.print("  1. Run 'python -m jarvis.setup' to validate environment")
        console.print("  2. Check system requirements (Apple Silicon, Python 3.11+)")
        console.print("  3. Run 'jarvis -v <command>' for debug logging")
        console.print("  4. See docs/CLI_GUIDE.md for detailed troubleshooting")
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


if __name__ == "__main__":
    run()
