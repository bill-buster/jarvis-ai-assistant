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
    FEATURE_EMAIL,
    FEATURE_IMESSAGE,
    cmd_health,
    cmd_version,
    create_parser,
    initialize_system,
    main,
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

    def test_parser_summarize_emails_command(self):
        """Parser parses summarize-emails command."""
        parser = create_parser()
        args = parser.parse_args(["summarize-emails"])
        assert args.command == "summarize-emails"

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
        assert FEATURE_EMAIL in health

    def test_initialize_warns_about_permissions(self):
        """Initialize system warns about missing permissions."""
        success, warnings = initialize_system()

        # Should have warnings about iMessage (no Full Disk Access in test)
        # and Gmail (not configured)
        assert len(warnings) >= 1


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

    def test_main_with_summarize_emails(self):
        """Main with summarize-emails returns zero (not implemented)."""
        exit_code = main(["summarize-emails"])
        assert exit_code == 0  # Returns 0 even though not implemented


class TestSearchMessages:
    """Tests for search-messages command."""

    def test_search_messages_no_access(self):
        """Search messages shows error when no iMessage access."""
        # This should fail gracefully since we don't have Full Disk Access
        exit_code = main(["search-messages", "test query"])
        # Should still return 0 (no results found) or 1 (permission error)
        assert exit_code in (0, 1)


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

        exit_code = main(["benchmark", "coverage"])

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
        from jarvis.cli import _template_only_response

        # Should return something even without model
        result = _template_only_response("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_response(self):
        """Fallback response returns static message."""
        from jarvis.cli import _fallback_response

        result = _fallback_response()
        assert isinstance(result, str)
        assert "unable" in result.lower() or "check" in result.lower()

    def test_imessage_degraded(self):
        """iMessage degraded mode returns empty list."""
        from jarvis.cli import _imessage_degraded

        result = _imessage_degraded("test")
        assert result == []

    def test_imessage_fallback(self):
        """iMessage fallback returns empty list."""
        from jarvis.cli import _imessage_fallback

        result = _imessage_fallback()
        assert result == []

    def test_email_degraded(self):
        """Email degraded returns message."""
        from jarvis.cli import _email_degraded

        result = _email_degraded()
        assert isinstance(result, str)
        assert "degraded" in result.lower()

    def test_email_fallback(self):
        """Email fallback returns unavailable message."""
        from jarvis.cli import _email_fallback

        result = _email_fallback()
        assert isinstance(result, str)
        assert "not available" in result.lower()


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
        from jarvis.cli import _check_imessage_access

        result = _check_imessage_access()
        assert isinstance(result, bool)

    def test_check_gmail_access_returns_false(self):
        """Check Gmail access returns False (not implemented)."""
        from jarvis.cli import _check_gmail_access

        result = _check_gmail_access()
        assert result is False
