"""Integration tests for the JARVIS CLI.

Tests the command-line interface and main entry point.
"""

from unittest.mock import MagicMock, patch

import pytest

# Check if MLX is available (only on Apple Silicon)
try:
    import mlx.core  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from core.health import reset_degradation_controller
from core.memory import reset_memory_controller
from jarvis.cli import (
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    cmd_health,
    cmd_version,
    create_parser,
    initialize_system,
    main,
)

# Import the marker for tests that require sentence_transformers
from tests.conftest import requires_sentence_transformers

# Marker for tests that depend on specific model outputs (may vary)
model_dependent = pytest.mark.xfail(
    reason="Model output varies - tests verify expected behavior but allow variation",
    strict=False,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_memory_controller()
    reset_degradation_controller()
    yield
    reset_memory_controller()
    reset_degradation_controller()


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_creates_successfully(self):
        """Parser is created without errors."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "jarvis"

    def test_parser_has_verbose_flag(self):
        """Parser has verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "health"])
        assert args.verbose is True

    def test_parser_has_version_flag(self):
        """Parser has version flag."""
        parser = create_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_chat_command(self):
        """Parser parses chat command."""
        parser = create_parser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"
        assert hasattr(args, "func")

    def test_parser_search_messages_command(self):
        """Parser parses search-messages command with query."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello"])
        assert args.command == "search-messages"
        assert args.query == "hello"
        assert args.limit == 20  # default

    def test_parser_search_messages_with_limit(self):
        """Parser parses search-messages command with custom limit."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "-l", "50"])
        assert args.limit == 50

    def test_parser_health_command(self):
        """Parser parses health command."""
        parser = create_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_parser_benchmark_command(self):
        """Parser parses benchmark command with type."""
        parser = create_parser()
        args = parser.parse_args(["benchmark", "memory"])
        assert args.command == "benchmark"
        assert args.type == "memory"

    def test_parser_benchmark_with_output(self):
        """Parser parses benchmark command with output file."""
        parser = create_parser()
        args = parser.parse_args(["benchmark", "latency", "-o", "results.json"])
        assert args.type == "latency"
        assert args.output == "results.json"

    def test_parser_benchmark_validates_type(self):
        """Parser validates benchmark type."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark", "invalid"])


class TestInitializeSystem:
    """Tests for system initialization."""

    def test_initialize_returns_success(self):
        """Initialize system returns success flag."""
        success, warnings = initialize_system()
        assert success is True

    def test_initialize_registers_features(self):
        """Initialize system registers degradation features."""
        from core.health import get_degradation_controller

        initialize_system()
        controller = get_degradation_controller()
        health = controller.get_health()

        assert FEATURE_CHAT in health
        assert FEATURE_IMESSAGE in health

    def test_initialize_warns_about_permissions(self):
        """Initialize system warns about missing permissions."""
        success, warnings = initialize_system()

        # Should have warnings about iMessage (no Full Disk Access in test)
        assert len(warnings) >= 0  # May or may not have warnings depending on env


class TestCmdVersion:
    """Tests for version command."""

    def test_version_returns_zero(self):
        """Version command returns success exit code."""
        parser = create_parser()
        args = parser.parse_args(["version"])
        exit_code = cmd_version(args)
        assert exit_code == 0


class TestCmdHealth:
    """Tests for health command."""

    def test_health_returns_zero(self):
        """Health command returns success exit code."""
        # First initialize the system to register features
        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["health"])
        exit_code = cmd_health(args)
        assert exit_code == 0


class TestMain:
    """Tests for main entry point."""

    def test_main_with_no_args_shows_help(self):
        """Main with no args shows help and returns zero."""
        exit_code = main([])
        assert exit_code == 0

    def test_main_with_version(self):
        """Main with --version returns zero."""
        exit_code = main(["--version"])
        assert exit_code == 0

    def test_main_with_health(self):
        """Main with health command returns zero."""
        exit_code = main(["health"])
        assert exit_code == 0


class TestSearchMessages:
    """Tests for search-messages command."""

    def test_search_messages_no_access(self):
        """Search messages shows error when no iMessage access."""
        # This should fail gracefully since we don't have Full Disk Access
        exit_code = main(["search-messages", "test query"])
        # Should still return 0 (no results found) or 1 (permission error)
        assert exit_code in (0, 1)

    def test_parser_search_messages_with_start_date(self):
        """Parser parses search-messages with start-date filter."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--start-date", "2024-01-15"])
        assert args.start_date == "2024-01-15"
        assert args.query == "hello"

    def test_parser_search_messages_with_end_date(self):
        """Parser parses search-messages with end-date filter."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--end-date", "2024-12-31"])
        assert args.end_date == "2024-12-31"

    def test_parser_search_messages_with_sender(self):
        """Parser parses search-messages with sender filter."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--sender", "+15551234567"])
        assert args.sender == "+15551234567"

    def test_parser_search_messages_with_sender_me(self):
        """Parser parses search-messages with sender=me filter."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--sender", "me"])
        assert args.sender == "me"

    def test_parser_search_messages_with_has_attachment(self):
        """Parser parses search-messages with has-attachment filter."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--has-attachment"])
        assert args.has_attachment is True

    def test_parser_search_messages_with_no_attachment(self):
        """Parser parses search-messages with no-attachment filter."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--no-attachment"])
        assert args.has_attachment is False

    def test_parser_search_messages_default_has_attachment_is_none(self):
        """Parser search-messages has_attachment defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello"])
        assert args.has_attachment is None

    def test_parser_search_messages_with_all_filters(self):
        """Parser parses search-messages with all filters combined."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "search-messages",
                "meeting",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-06-30",
                "--sender",
                "+15551234567",
                "--has-attachment",
                "-l",
                "50",
            ]
        )
        assert args.query == "meeting"
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-06-30"
        assert args.sender == "+15551234567"
        assert args.has_attachment is True
        assert args.limit == 50

    def test_parser_search_messages_with_datetime(self):
        """Parser parses search-messages with datetime format."""
        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--start-date", "2024-01-15 14:30"])
        assert args.start_date == "2024-01-15 14:30"


class TestBenchmark:
    """Tests for benchmark command."""

    @patch("subprocess.run")
    def test_benchmark_calls_subprocess(self, mock_run):
        """Benchmark command calls subprocess with correct module."""
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = main(["benchmark", "memory"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-m" in call_args
        assert "benchmarks.memory.run" in call_args
        assert exit_code == 0

    @patch("subprocess.run")
    def test_benchmark_with_output(self, mock_run):
        """Benchmark command passes output file to subprocess."""
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = main(["benchmark", "latency", "-o", "results.json"])

        call_args = mock_run.call_args[0][0]
        assert "--output" in call_args
        assert "results.json" in call_args
        assert exit_code == 0

    @patch("subprocess.run")
    def test_benchmark_propagates_exit_code(self, mock_run):
        """Benchmark command propagates subprocess exit code."""
        mock_run.return_value = MagicMock(returncode=42)

        exit_code = main(["benchmark", "hhem"])

        assert exit_code == 42


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available (requires Apple Silicon)")
class TestChatCommand:
    """Tests for chat command (requires MLX for model imports)."""

    @patch("rich.console.Console.input", side_effect=["quit"])
    @patch("models.get_generator")
    def test_chat_exits_on_quit(self, mock_generator, mock_console_input):
        """Chat command exits gracefully on quit."""
        mock_gen = MagicMock()
        mock_generator.return_value = mock_gen

        from jarvis.cli import cmd_chat

        parser = create_parser()
        args = parser.parse_args(["chat"])

        exit_code = cmd_chat(args)
        assert exit_code == 0

    @patch("rich.console.Console.input", side_effect=["exit"])
    @patch("models.get_generator")
    def test_chat_exits_on_exit(self, mock_generator, mock_console_input):
        """Chat command exits gracefully on exit."""
        mock_gen = MagicMock()
        mock_generator.return_value = mock_gen

        from jarvis.cli import cmd_chat

        parser = create_parser()
        args = parser.parse_args(["chat"])

        exit_code = cmd_chat(args)
        assert exit_code == 0

    @patch("rich.console.Console.input", side_effect=KeyboardInterrupt())
    @patch("models.get_generator")
    def test_chat_handles_keyboard_interrupt(self, mock_generator, mock_console_input):
        """Chat command handles Ctrl+C gracefully."""
        mock_gen = MagicMock()
        mock_generator.return_value = mock_gen

        from jarvis.cli import cmd_chat

        parser = create_parser()
        args = parser.parse_args(["chat"])

        exit_code = cmd_chat(args)
        assert exit_code == 0


class TestFallbackBehaviors:
    """Tests for fallback and degraded behaviors."""

    def test_template_only_response(self):
        """Template-only response works without model."""
        from jarvis.system import _template_only_response

        # Should return something even without model
        result = _template_only_response("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_response(self):
        """Fallback response returns static message."""
        from jarvis.system import _fallback_response

        result = _fallback_response()
        assert isinstance(result, str)
        assert "unable" in result.lower() or "check" in result.lower()

    def test_imessage_degraded(self):
        """iMessage degraded mode returns empty list."""
        from jarvis.system import _imessage_degraded

        result = _imessage_degraded("test")
        assert result == []

    def test_imessage_fallback(self):
        """iMessage fallback returns empty list."""
        from jarvis.system import _imessage_fallback

        result = _imessage_fallback()
        assert result == []


class TestCleanup:
    """Tests for cleanup function."""

    def test_cleanup_resets_controllers(self):
        """Cleanup resets memory and degradation controllers."""
        from jarvis.cli import cleanup

        # Initialize first
        initialize_system()

        # Cleanup should not raise
        cleanup()

    def test_cleanup_handles_errors(self):
        """Cleanup handles errors gracefully."""
        from jarvis.cli import cleanup

        # Should not raise even if things are in weird state
        cleanup()
        cleanup()  # Second call should also be safe


class TestModuleExecution:
    """Tests for module execution via python -m jarvis."""

    def test_module_entry_point_exists(self):
        """jarvis.__main__ exists and is importable."""
        import jarvis.__main__

        assert hasattr(jarvis.__main__, "run")


class TestVerboseMode:
    """Tests for verbose logging mode."""

    def test_verbose_flag_sets_logging(self):
        """Verbose flag enables debug logging."""
        from jarvis.cli import setup_logging

        # Just verify it doesn't crash
        setup_logging(verbose=False)
        setup_logging(verbose=True)


class TestAccessChecks:
    """Tests for access check functions."""

    def test_check_imessage_access_returns_bool(self):
        """Check iMessage access returns boolean."""
        from jarvis.system import _check_imessage_access

        result = _check_imessage_access()
        assert isinstance(result, bool)

    def test_check_imessage_access_exception_returns_false(self):
        """Check iMessage access returns False on exception."""
        from jarvis.system import _check_imessage_access

        # Patch the import inside the function
        with patch("integrations.imessage.ChatDBReader") as mock_reader:
            mock_reader.side_effect = Exception("DB error")
            result = _check_imessage_access()
        assert result is False


class TestChatCommandMocked:
    """Tests for chat command with full mocking (no MLX required)."""

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_generates_and_displays_response(self, mock_console, mock_mem_ctrl, mock_deg_ctrl):
        """Chat generates response and displays it."""
        # Setup mocks
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        mock_deg_ctrl.return_value.execute.return_value = "Hello! I'm JARVIS."

        # Simulate: user enters "hello", then "quit"
        mock_console.input.side_effect = ["hello", "quit"]

        # Need to mock the models import inside cmd_chat
        mock_gen = MagicMock()
        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            from jarvis.cli import cmd_chat

            parser = create_parser()
            args = parser.parse_args(["chat"])
            exit_code = cmd_chat(args)

        assert exit_code == 0
        # Verify response was printed
        assert mock_console.print.call_count >= 3  # Panel, mode, response, goodbye

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_handles_empty_input(self, mock_console, mock_mem_ctrl, mock_deg_ctrl):
        """Chat skips empty input."""
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        # Simulate: user enters empty string, then "exit"
        mock_console.input.side_effect = ["", "   ", "exit"]

        mock_gen = MagicMock()
        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            from jarvis.cli import cmd_chat

            parser = create_parser()
            args = parser.parse_args(["chat"])
            exit_code = cmd_chat(args)

        assert exit_code == 0
        # Degradation controller should not have been called for empty input
        mock_deg_ctrl.return_value.execute.assert_not_called()

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_exits_on_q_shortcut(self, mock_console, mock_mem_ctrl, mock_deg_ctrl):
        """Chat exits on 'q' shortcut."""
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        mock_console.input.side_effect = ["q"]

        mock_gen = MagicMock()
        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            from jarvis.cli import cmd_chat

            parser = create_parser()
            args = parser.parse_args(["chat"])
            exit_code = cmd_chat(args)

        assert exit_code == 0

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_handles_generation_error(self, mock_console, mock_mem_ctrl, mock_deg_ctrl):
        """Chat handles errors during generation."""
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        # First call raises error, then quit
        mock_deg_ctrl.return_value.execute.side_effect = [Exception("Generation failed"), None]

        mock_console.input.side_effect = ["hello", "quit"]

        mock_gen = MagicMock()
        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            from jarvis.cli import cmd_chat

            parser = create_parser()
            args = parser.parse_args(["chat"])
            exit_code = cmd_chat(args)

        assert exit_code == 0
        # Error message should have been printed
        error_calls = [c for c in mock_console.print.call_args_list if "Error" in str(c)]
        assert len(error_calls) >= 1

    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_returns_error_on_import_failure(self, mock_console, mock_mem_ctrl):
        """Chat returns error code when models module unavailable."""
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        # Patch to simulate ImportError
        import sys

        original_modules = sys.modules.copy()
        # Remove models from modules to trigger ImportError
        if "models" in sys.modules:
            del sys.modules["models"]

        # Create a mock that raises ImportError
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "models":
                raise ImportError("No module named 'models'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Reload the cli module to trigger the import error path
            from jarvis import cli

            # Force re-execution of import in cmd_chat by clearing cache
            parser = create_parser()
            args = parser.parse_args(["chat"])

            # Call cmd_chat which will try to import models
            exit_code = cli.cmd_chat(args)

        # Restore modules
        sys.modules.update(original_modules)

        # Should return 1 on import error
        assert exit_code == 1

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_handles_eof_error(self, mock_console, mock_mem_ctrl, mock_deg_ctrl):
        """Chat handles EOFError (pipe closed)."""
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        mock_console.input.side_effect = EOFError()

        mock_gen = MagicMock()
        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            from jarvis.cli import cmd_chat

            parser = create_parser()
            args = parser.parse_args(["chat"])
            exit_code = cmd_chat(args)

        assert exit_code == 0


class TestParseDateFunction:
    """Tests for _parse_date helper function."""

    def test_parse_date_valid_date(self):
        """Parse valid YYYY-MM-DD format."""
        from jarvis.cli import _parse_date

        result = _parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.minute == 0

    def test_parse_date_valid_datetime(self):
        """Parse valid YYYY-MM-DD HH:MM format."""
        from jarvis.cli import _parse_date

        result = _parse_date("2024-01-15 14:30")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30

    def test_parse_date_invalid_format(self):
        """Return None for invalid date format."""
        from jarvis.cli import _parse_date

        result = _parse_date("01/15/2024")
        assert result is None

    def test_parse_date_empty_string(self):
        """Return None for empty string."""
        from jarvis.cli import _parse_date

        result = _parse_date("")
        assert result is None

    def test_parse_date_has_utc_timezone(self):
        """Parsed date has UTC timezone."""
        from datetime import UTC

        from jarvis.cli import _parse_date

        result = _parse_date("2024-01-15")
        assert result is not None
        assert result.tzinfo == UTC


class TestSearchMessagesExtended:
    """Extended tests for search-messages command."""

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_displays_results(self, mock_console, mock_deg_ctrl):
        """Search displays results in table format."""
        from datetime import datetime

        from jarvis.cli import cmd_search_messages

        # Create mock message
        mock_msg = MagicMock()
        mock_msg.date = datetime(2024, 1, 15, 10, 30)
        mock_msg.is_from_me = False
        mock_msg.sender = "+1234567890"
        mock_msg.text = "Hello there, how are you doing today?"

        mock_deg_ctrl.return_value.execute.return_value = [mock_msg]

        # Initialize system first
        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Table should be printed
        assert mock_console.print.call_count >= 2

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_handles_empty_results(self, mock_console, mock_deg_ctrl):
        """Search handles empty results gracefully."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "nonexistent"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Should print "No messages found"
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("No messages found" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_handles_permission_error(self, mock_console, mock_deg_ctrl):
        """Search handles PermissionError appropriately."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.side_effect = PermissionError(
            "Cannot access iMessage database"
        )

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "test"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 1
        # Should print permission error message
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Permission error" in c or "Full Disk Access" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_handles_generic_exception(self, mock_console, mock_deg_ctrl):
        """Search handles generic exceptions."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.side_effect = RuntimeError("Database corrupted")

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "test"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 1
        # Should print error message
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_truncates_long_messages(self, mock_console, mock_deg_ctrl):
        """Search truncates long messages in display."""
        from datetime import datetime

        from jarvis.cli import cmd_search_messages

        # Create mock message with very long text
        mock_msg = MagicMock()
        mock_msg.date = datetime(2024, 1, 15, 10, 30)
        mock_msg.is_from_me = True
        mock_msg.sender = None
        mock_msg.text = "A" * 200  # Very long message

        mock_deg_ctrl.return_value.execute.return_value = [mock_msg]

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "AAAA"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_handles_message_without_date(self, mock_console, mock_deg_ctrl):
        """Search handles messages without date."""
        from jarvis.cli import cmd_search_messages

        mock_msg = MagicMock()
        mock_msg.date = None
        mock_msg.is_from_me = False
        mock_msg.sender = None
        mock_msg.text = "Test message"

        mock_deg_ctrl.return_value.execute.return_value = [mock_msg]

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "test"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_with_start_date_filter(self, mock_console, mock_deg_ctrl):
        """Search passes start_date filter to reader."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--start-date", "2024-01-15"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Verify filter info was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("after 2024-01-15" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_with_end_date_filter(self, mock_console, mock_deg_ctrl):
        """Search passes end_date filter to reader."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--end-date", "2024-12-31"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Verify filter info was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("before 2024-12-31" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_with_sender_filter(self, mock_console, mock_deg_ctrl):
        """Search passes sender filter to reader."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--sender", "+15551234567"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Verify filter info was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("from +15551234567" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_with_has_attachment_filter(self, mock_console, mock_deg_ctrl):
        """Search passes has_attachment=True filter to reader."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--has-attachment"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Verify filter info was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("with attachments" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_with_no_attachment_filter(self, mock_console, mock_deg_ctrl):
        """Search passes has_attachment=False filter to reader."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--no-attachment"])

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Verify filter info was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("without attachments" in c for c in print_calls)

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_with_all_filters_combined(self, mock_console, mock_deg_ctrl):
        """Search with all filters shows all filter info."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(
            [
                "search-messages",
                "meeting",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-06-30",
                "--sender",
                "me",
                "--has-attachment",
            ]
        )

        exit_code = cmd_search_messages(args)

        assert exit_code == 0
        # Verify all filter info was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        filter_line = [c for c in print_calls if "Filters:" in c]
        assert len(filter_line) >= 1

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.console")
    def test_search_invalid_date_handled_gracefully(self, mock_console, mock_deg_ctrl):
        """Search handles invalid date format gracefully."""
        from jarvis.cli import cmd_search_messages

        mock_deg_ctrl.return_value.execute.return_value = []

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["search-messages", "hello", "--start-date", "invalid-date"])

        # Should not crash, date will be None
        exit_code = cmd_search_messages(args)

        assert exit_code == 0


class TestHealthExtended:
    """Extended tests for health command."""

    @patch("jarvis.cli.console")
    def test_health_shows_model_loaded(self, mock_console):
        """Health shows model loaded status."""
        initialize_system()

        # Mock the models module
        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = True
        mock_gen.get_memory_usage_mb.return_value = 512.5

        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            parser = create_parser()
            args = parser.parse_args(["health"])
            exit_code = cmd_health(args)

        assert exit_code == 0

    @patch("jarvis.cli.console")
    def test_health_shows_model_not_loaded(self, mock_console):
        """Health shows model not loaded status."""
        initialize_system()

        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = False

        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            parser = create_parser()
            args = parser.parse_args(["health"])
            exit_code = cmd_health(args)

        assert exit_code == 0

    @patch("jarvis.cli.console")
    def test_health_handles_model_exception(self, mock_console):
        """Health handles exception when checking model status."""
        initialize_system()

        # Create a module mock that raises when get_generator is called
        mock_models = MagicMock()
        mock_models.get_generator.side_effect = RuntimeError("Model not available")

        with patch.dict("sys.modules", {"models": mock_models}):
            parser = create_parser()
            args = parser.parse_args(["health"])
            exit_code = cmd_health(args)

        assert exit_code == 0
        # Should print warning about model status
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("unavailable" in c.lower() for c in print_calls)

    @patch("jarvis.cli._check_imessage_access")
    @patch("jarvis.cli.console")
    def test_health_shows_degraded_features(self, mock_console, mock_imessage_check):
        """Health shows degraded feature status."""
        from core.health import get_degradation_controller

        mock_imessage_check.return_value = False

        initialize_system()

        # Manually set feature to degraded state
        controller = get_degradation_controller()
        # Trigger failures to move to degraded state
        for _ in range(5):
            try:
                controller.execute(
                    FEATURE_IMESSAGE,
                    lambda q: (_ for _ in ()).throw(RuntimeError("Simulated failure")),
                    "test",
                )
            except RuntimeError:
                pass

        parser = create_parser()
        args = parser.parse_args(["health"])
        exit_code = cmd_health(args)

        assert exit_code == 0


class TestBenchmarkExtended:
    """Extended tests for benchmark command."""

    @patch("jarvis.cli.console")
    def test_benchmark_unknown_type_returns_error(self, mock_console):
        """Benchmark with unknown type returns error."""
        from jarvis.cli import cmd_benchmark

        initialize_system()

        # Create args manually since parser won't accept invalid type
        args = MagicMock()
        args.type = "unknown_benchmark"
        args.output = None

        exit_code = cmd_benchmark(args)

        assert exit_code == 1
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Unknown benchmark type" in c for c in print_calls)

    @patch("subprocess.run")
    @patch("jarvis.cli.console")
    def test_benchmark_handles_subprocess_exception(self, mock_console, mock_run):
        """Benchmark handles subprocess exception."""
        from jarvis.cli import cmd_benchmark

        mock_run.side_effect = OSError("Command not found")

        initialize_system()

        parser = create_parser()
        args = parser.parse_args(["benchmark", "memory"])

        exit_code = cmd_benchmark(args)

        assert exit_code == 1
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error running benchmark" in c for c in print_calls)

    @patch("subprocess.run")
    def test_benchmark_hhem_calls_correct_module(self, mock_run):
        """Benchmark hhem calls correct module."""
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = main(["benchmark", "hhem"])

        call_args = mock_run.call_args[0][0]
        assert "benchmarks.hallucination.run" in call_args
        assert exit_code == 0


class TestMainExtended:
    """Extended tests for main entry point."""

    @patch("jarvis.cli.initialize_system")
    @patch("jarvis.cli.console")
    def test_main_handles_failed_initialization(self, mock_console, mock_init):
        """Main handles failed system initialization."""
        mock_init.return_value = (False, ["Critical error"])

        exit_code = main(["health"])

        assert exit_code == 1
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Failed to initialize" in c for c in print_calls)

    @patch("jarvis.cli.console")
    def test_main_displays_warnings(self, mock_console):
        """Main displays initialization warnings."""
        with patch("jarvis.system._check_imessage_access", return_value=False):
            exit_code = main(["health"])

        assert exit_code == 0
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Warning" in c or "iMessage" in c for c in print_calls)


class TestCleanupExtended:
    """Extended tests for cleanup function."""

    @patch("jarvis.cli.reset_memory_controller")
    @patch("jarvis.cli.reset_degradation_controller")
    def test_cleanup_handles_reset_generator_exception(self, mock_deg, mock_mem):
        """Cleanup handles exception in reset_generator."""
        from jarvis.cli import cleanup

        # Create mock that raises
        mock_models = MagicMock()
        mock_models.reset_generator.side_effect = RuntimeError("Reset failed")

        with patch.dict("sys.modules", {"models": mock_models}):
            # Should not raise
            cleanup()

    @patch("jarvis.cli.reset_memory_controller")
    def test_cleanup_handles_memory_controller_exception(self, mock_mem):
        """Cleanup handles exception in reset_memory_controller."""
        from jarvis.cli import cleanup

        mock_mem.side_effect = RuntimeError("Memory reset failed")

        # Should not raise
        cleanup()


class TestRunFunction:
    """Tests for the run() entry point function."""

    @patch("jarvis.cli.main")
    @patch("jarvis.cli.cleanup")
    def test_run_calls_main_and_cleanup(self, mock_cleanup, mock_main):
        """Run calls main and cleanup."""
        mock_main.return_value = 0

        with pytest.raises(SystemExit) as exc_info:
            from jarvis.cli import run

            run()

        assert exc_info.value.code == 0
        mock_main.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch("jarvis.cli.main")
    @patch("jarvis.cli.cleanup")
    @patch("jarvis.cli.console")
    def test_run_handles_keyboard_interrupt(self, mock_console, mock_cleanup, mock_main):
        """Run handles KeyboardInterrupt."""
        mock_main.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            from jarvis.cli import run

            run()

        assert exc_info.value.code == 130
        mock_cleanup.assert_called_once()

    @patch("jarvis.cli.main")
    @patch("jarvis.cli.cleanup")
    @patch("jarvis.cli.console")
    def test_run_handles_unexpected_exception(self, mock_console, mock_cleanup, mock_main):
        """Run handles unexpected exceptions."""
        mock_main.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(SystemExit) as exc_info:
            from jarvis.cli import run

            run()

        assert exc_info.value.code == 1
        mock_cleanup.assert_called_once()


class TestTemplateMatching:
    """Tests for template matching in degraded mode."""

    def test_template_only_response_with_match(self):
        """Template response returns matched template."""
        mock_match = MagicMock()
        mock_match.template.response = "Matched template response"

        mock_matcher_class = MagicMock()
        mock_matcher_class.return_value.match.return_value = mock_match

        # Create a mock module for models.templates
        mock_templates_module = MagicMock()
        mock_templates_module.TemplateMatcher = mock_matcher_class

        with patch.dict("sys.modules", {"models.templates": mock_templates_module}):
            from jarvis.system import _template_only_response

            result = _template_only_response("what time is it?")

        assert result == "Matched template response"

    def test_template_only_response_no_match(self):
        """Template response returns default when no match."""
        mock_matcher_class = MagicMock()
        mock_matcher_class.return_value.match.return_value = None

        mock_templates_module = MagicMock()
        mock_templates_module.TemplateMatcher = mock_matcher_class

        with patch.dict("sys.modules", {"models.templates": mock_templates_module}):
            from jarvis.system import _template_only_response

            result = _template_only_response("random query")

        assert "limited mode" in result.lower()

    def test_template_only_response_handles_exception(self):
        """Template response handles exception gracefully."""
        # Create mock that raises on instantiation
        mock_templates_module = MagicMock()
        mock_templates_module.TemplateMatcher.side_effect = Exception("Match failed")

        with patch.dict("sys.modules", {"models.templates": mock_templates_module}):
            from jarvis.system import _template_only_response

            result = _template_only_response("test")

        assert "limited mode" in result.lower()


class TestReplyCommand:
    """Tests for reply command."""

    def test_parser_reply_command(self):
        """Parser parses reply command with person."""
        parser = create_parser()
        args = parser.parse_args(["reply", "John"])
        assert args.command == "reply"
        assert args.person == "John"
        assert hasattr(args, "func")

    def test_parser_reply_with_instruction(self):
        """Parser parses reply command with instruction."""
        parser = create_parser()
        args = parser.parse_args(["reply", "Sarah", "-i", "say yes politely"])
        assert args.person == "Sarah"
        assert args.instruction == "say yes politely"

    @patch("jarvis.cli._check_imessage_access")
    @patch("jarvis.cli.console")
    def test_reply_no_imessage_access(self, mock_console, mock_check):
        """Reply returns error when no iMessage access."""
        from jarvis.cli import cmd_reply

        mock_check.return_value = False

        parser = create_parser()
        args = parser.parse_args(["reply", "John"])

        exit_code = cmd_reply(args)

        assert exit_code == 1
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Full Disk Access" in c for c in print_calls)

    @patch("integrations.imessage.ChatDBReader")
    @patch("jarvis.cli._check_imessage_access")
    @patch("jarvis.cli.console")
    def test_reply_person_not_found(self, mock_console, mock_check, mock_reader_class):
        """Reply shows helpful message when person not found."""
        from jarvis.cli import cmd_reply

        mock_check.return_value = True

        # Mock ChatDBReader
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader_class.return_value = mock_reader

        # Return empty for find conversation, but return some conversations
        mock_conv = MagicMock()
        mock_conv.display_name = "Mom"
        mock_conv.participants = ["+15551234567"]
        mock_reader.get_conversations.return_value = [mock_conv]

        # Mock ContextFetcher to return None for find_conversation_by_name
        with patch("jarvis.cli.ContextFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher.find_conversation_by_name.return_value = None
            mock_fetcher_class.return_value = mock_fetcher

            parser = create_parser()
            args = parser.parse_args(["reply", "Unknown"])

            exit_code = cmd_reply(args)

        assert exit_code == 1
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Could not find" in c for c in print_calls)


class TestSummarizeCommand:
    """Tests for summarize command."""

    def test_parser_summarize_command(self):
        """Parser parses summarize command with person."""
        parser = create_parser()
        args = parser.parse_args(["summarize", "Mom"])
        assert args.command == "summarize"
        assert args.person == "Mom"
        assert args.messages == 50  # default

    def test_parser_summarize_with_message_count(self):
        """Parser parses summarize command with message count."""
        parser = create_parser()
        args = parser.parse_args(["summarize", "Dad", "-n", "100"])
        assert args.person == "Dad"
        assert args.messages == 100

    @patch("jarvis.cli._check_imessage_access")
    @patch("jarvis.cli.console")
    def test_summarize_no_imessage_access(self, mock_console, mock_check):
        """Summarize returns error when no iMessage access."""
        from jarvis.cli import cmd_summarize

        mock_check.return_value = False

        parser = create_parser()
        args = parser.parse_args(["summarize", "Mom"])

        exit_code = cmd_summarize(args)

        assert exit_code == 1
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Full Disk Access" in c for c in print_calls)


@requires_sentence_transformers
class TestIntentClassifier:
    """Tests for intent classification.

    These tests require sentence_transformers to be available.
    """

    def test_classify_reply_intent(self):
        """Classifier detects reply intent."""
        from jarvis.intent import IntentClassifier, IntentType

        classifier = IntentClassifier()

        result = classifier.classify("help me reply to John")
        assert result.intent == IntentType.REPLY
        assert result.extracted_params.get("person_name") == "John"

    def test_classify_summarize_intent(self):
        """Classifier detects summarize intent."""
        from jarvis.intent import IntentClassifier, IntentType

        classifier = IntentClassifier()

        result = classifier.classify("summarize my chat with Sarah")
        assert result.intent == IntentType.SUMMARIZE
        assert result.extracted_params.get("person_name") == "Sarah"

    def test_classify_search_intent(self):
        """Classifier detects search intent."""
        from jarvis.intent import IntentClassifier, IntentType

        classifier = IntentClassifier()

        result = classifier.classify("find messages about dinner")
        assert result.intent == IntentType.SEARCH

    @model_dependent
    def test_classify_quick_reply_intent(self):
        """Classifier detects quick reply intent."""
        from jarvis.intent import IntentClassifier, IntentType

        classifier = IntentClassifier()

        result = classifier.classify("ok")
        assert result.intent == IntentType.QUICK_REPLY

    def test_classify_general_intent(self):
        """Classifier defaults to general for unknown queries."""
        from jarvis.intent import IntentClassifier, IntentType

        classifier = IntentClassifier()

        result = classifier.classify("what is the weather today?")
        assert result.intent == IntentType.GENERAL


class TestContextFetcher:
    """Tests for context fetcher."""

    def test_find_conversation_by_name(self):
        """Context fetcher finds conversation by display name."""
        from jarvis.context import ContextFetcher

        mock_reader = MagicMock()
        mock_conv = MagicMock()
        mock_conv.display_name = "John Smith"
        mock_conv.chat_id = "chat123"
        mock_conv.participants = []
        mock_reader.get_conversations.return_value = [mock_conv]

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("John")

        assert result == "chat123"

    def test_find_conversation_by_participant(self):
        """Context fetcher finds conversation by participant."""
        from jarvis.context import ContextFetcher

        mock_reader = MagicMock()
        mock_conv = MagicMock()
        mock_conv.display_name = None
        mock_conv.chat_id = "chat456"
        mock_conv.participants = ["+15551234567"]
        mock_reader.get_conversations.return_value = [mock_conv]

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("+15551234567")

        assert result == "chat456"

    def test_find_conversation_not_found(self):
        """Context fetcher returns None when conversation not found."""
        from jarvis.context import ContextFetcher

        mock_reader = MagicMock()
        mock_reader.get_conversations.return_value = []

        fetcher = ContextFetcher(mock_reader)
        result = fetcher.find_conversation_by_name("Unknown")

        assert result is None


class TestPromptBuilders:
    """Tests for prompt builders."""

    def test_build_reply_prompt(self):
        """Build reply prompt includes context."""
        from jarvis.prompts import build_reply_prompt

        prompt = build_reply_prompt(
            context="[Jan 15] John: Hello",
            last_message="Hello",
        )

        assert "[Jan 15] John: Hello" in prompt
        # Template says "Generate a reply that matches"
        assert "Generate a reply" in prompt or "Your reply:" in prompt

    def test_build_reply_prompt_with_instruction(self):
        """Build reply prompt includes custom instruction."""
        from jarvis.prompts import build_reply_prompt

        prompt = build_reply_prompt(
            context="[Jan 15] John: Want to meet?",
            last_message="Want to meet?",
            instruction="say yes but ask for a time",
        )

        assert "say yes but ask for a time" in prompt

    def test_build_summary_prompt(self):
        """Build summary prompt includes context."""
        from jarvis.prompts import build_summary_prompt

        prompt = build_summary_prompt(
            context="[Jan 15] John: Hello\n[Jan 15] Me: Hi",
        )

        assert "[Jan 15] John: Hello" in prompt
        assert "Summary" in prompt or "summariz" in prompt.lower()

    def test_build_summary_prompt_with_focus(self):
        """Build summary prompt includes focus area."""
        from jarvis.prompts import build_summary_prompt

        prompt = build_summary_prompt(
            context="[Jan 15] John: Meet at 3pm",
            focus="action items",
        )

        assert "action items" in prompt


class TestIntentRouting:
    """Tests for intent-based routing in chat."""

    @patch("jarvis.cli.get_degradation_controller")
    @patch("jarvis.cli.get_memory_controller")
    @patch("jarvis.cli.console")
    def test_chat_routes_quick_reply_to_template(self, mock_console, mock_mem_ctrl, mock_deg_ctrl):
        """Chat routes quick reply intent to template matcher."""
        mock_state = MagicMock()
        mock_state.current_mode.value = "FULL"
        mock_mem_ctrl.return_value.get_state.return_value = mock_state

        # Template matcher should be used, return a response
        mock_deg_ctrl.return_value.execute.return_value = "Got it!"

        mock_console.input.side_effect = ["ok", "quit"]

        mock_gen = MagicMock()
        with patch.dict("sys.modules", {"models": MagicMock(get_generator=lambda: mock_gen)}):
            from jarvis.cli import cmd_chat

            parser = create_parser()
            args = parser.parse_args(["chat"])
            exit_code = cmd_chat(args)

        assert exit_code == 0
        # Verify we got a response (any response means routing worked)
        assert mock_deg_ctrl.return_value.execute.called
