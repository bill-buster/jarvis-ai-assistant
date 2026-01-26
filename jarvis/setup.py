"""JARVIS Setup and Onboarding Wizard.

Validates the environment and guides users through first-time setup.
Performs permission checks, database validation, config initialization,
model checks, and produces a health report.

Usage:
    python -m jarvis.setup          # Run full setup
    python -m jarvis.setup --check  # Just check, don't modify
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from contracts.health import Permission, PermissionStatus, SchemaInfo
from contracts.memory import MemoryMode
from jarvis.config import (
    JarvisConfig,
    load_config,
    save_config,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_TEMPLATE_THRESHOLD = 0.7
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
JARVIS_CONFIG_DIR = Path.home() / ".jarvis"
JARVIS_CONFIG_FILE = JARVIS_CONFIG_DIR / "config.json"

# Estimated memory requirements for the default model (in MB)
DEFAULT_MODEL_MEMORY_MB = 800


class CheckStatus(Enum):
    """Status of a setup check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single setup check."""

    name: str
    status: CheckStatus
    message: str
    details: str | None = None
    fix_instructions: str | None = None


@dataclass
class SetupResult:
    """Result of the full setup process."""

    success: bool
    checks: list[CheckResult] = field(default_factory=list)
    config_created: bool = False
    config_path: Path | None = None


class PermissionMonitorImpl:
    """Implementation of PermissionMonitor for macOS TCC checks.

    Checks Full Disk Access by attempting to access protected paths.
    """

    def __init__(self) -> None:
        """Initialize the permission monitor."""
        self._fix_instructions = {
            Permission.FULL_DISK_ACCESS: (
                "Grant Full Disk Access to your terminal app:\n"
                "1. Open System Settings > Privacy & Security > Full Disk Access\n"
                "2. Click the '+' button and add Terminal.app (or your IDE/terminal)\n"
                "3. Restart your terminal application"
            ),
            Permission.CONTACTS: (
                "Grant Contacts access in System Settings > Privacy & Security > Contacts"
            ),
            Permission.CALENDAR: (
                "Grant Calendar access in System Settings > Privacy & Security > Calendars"
            ),
            Permission.AUTOMATION: ("Grant Automation permissions when prompted by macOS"),
        }

    def check_permission(self, permission: Permission) -> PermissionStatus:
        """Check if a specific permission is granted.

        Args:
            permission: The permission to check.

        Returns:
            PermissionStatus with check results.
        """
        granted = False
        timestamp = datetime.now().isoformat()

        if permission == Permission.FULL_DISK_ACCESS:
            granted = self._check_full_disk_access()
        else:
            # For other permissions, we'd need to attempt access
            # For now, mark as granted (placeholder for future implementation)
            granted = True

        return PermissionStatus(
            permission=permission,
            granted=granted,
            last_checked=timestamp,
            fix_instructions=self._fix_instructions.get(permission, ""),
        )

    def check_all(self) -> list[PermissionStatus]:
        """Check all required permissions.

        Returns:
            List of PermissionStatus for each permission.
        """
        return [
            self.check_permission(Permission.FULL_DISK_ACCESS),
        ]

    def wait_for_permission(self, permission: Permission, timeout_seconds: int) -> bool:
        """Block until permission granted or timeout.

        Args:
            permission: The permission to wait for.
            timeout_seconds: Maximum time to wait.

        Returns:
            True if permission granted, False if timeout.
        """
        import time

        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.check_permission(permission)
            if status.granted:
                return True
            time.sleep(1)
        return False

    def _check_full_disk_access(self) -> bool:
        """Check if Full Disk Access is granted.

        Attempts to read from protected locations to verify FDA.
        This is a low-level permission check for setup/onboarding.
        For runtime iMessage access checks, use the ChatDBReader integration
        via jarvis.cli._check_imessage_access() which tests the full stack.

        Returns:
            True if FDA is granted.
        """
        # Check if we can access the iMessage database
        if CHAT_DB_PATH.exists():
            try:
                # Try to actually read from it
                with CHAT_DB_PATH.open("rb") as f:
                    f.read(1)
                return True
            except (PermissionError, OSError):
                return False
        # If the file doesn't exist, we can't determine FDA status from it
        # Try another protected location
        tcc_db = Path.home() / "Library" / "Application Support" / "com.apple.TCC" / "TCC.db"
        if tcc_db.exists():
            try:
                with tcc_db.open("rb") as f:
                    f.read(1)
                return True
            except (PermissionError, OSError):
                return False

        # If neither exists, assume we have access (non-macOS or clean system)
        return True


class SchemaDetectorImpl:
    """Implementation of SchemaDetector for chat.db schema detection."""

    # Known schema versions and their identifying characteristics
    _KNOWN_SCHEMAS = {
        "v14": {"required_tables": {"message", "chat", "handle", "chat_message_join"}},
        "v15": {"required_tables": {"message", "chat", "handle", "chat_message_join"}},
    }

    def detect(self, db_path: str) -> SchemaInfo:
        """Detect schema version and compatibility.

        Args:
            db_path: Path to the chat.db database.

        Returns:
            SchemaInfo with version and compatibility details.
        """
        import sqlite3

        path = Path(db_path)
        if not path.exists():
            return SchemaInfo(
                version="unknown",
                tables=[],
                compatible=False,
                migration_needed=False,
                known_schema=False,
            )

        try:
            # Open read-only
            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Detect version based on message table columns
            cursor.execute("PRAGMA table_info(message)")
            message_columns = {row[1] for row in cursor.fetchall()}

            cursor.execute("PRAGMA table_info(chat)")
            chat_columns = {row[1] for row in cursor.fetchall()}

            conn.close()

            # Determine version based on column presence
            version = "v14"
            if "thread_originator_guid" in message_columns:
                if "service_name" in chat_columns:
                    version = "v15"
                # else: Has thread_originator_guid but not service_name - use v14
            else:
                # No thread_originator_guid - could be v14 or unknown older version
                # Check for expected v14 columns to validate
                expected_v14_columns = {"text", "date", "is_from_me", "handle_id"}
                if not expected_v14_columns.issubset(message_columns):
                    logging.warning(
                        "Could not reliably detect schema version (missing expected columns), "
                        "defaulting to v14. Some queries may fail."
                    )

            # Check compatibility
            required = {"message", "chat", "handle", "chat_message_join", "chat_handle_join"}
            compatible = required.issubset(set(tables))
            known_schema = version in self._KNOWN_SCHEMAS

            return SchemaInfo(
                version=version,
                tables=tables,
                compatible=compatible,
                migration_needed=False,
                known_schema=known_schema,
            )

        except sqlite3.OperationalError as e:
            logger.warning(f"Cannot read database schema: {e}")
            return SchemaInfo(
                version="unknown",
                tables=[],
                compatible=False,
                migration_needed=False,
                known_schema=False,
            )

    def get_query(self, query_name: str, schema_version: str) -> str:
        """Get appropriate SQL query for the detected schema.

        Args:
            query_name: Name of the query.
            schema_version: Schema version string.

        Returns:
            SQL query string.
        """
        # Delegate to the existing queries module
        from integrations.imessage.queries import get_query

        return get_query(query_name, schema_version)


class SetupWizard:
    """JARVIS setup and onboarding wizard.

    Validates the environment and guides users through first-time setup.
    """

    def __init__(
        self,
        console: Console | None = None,
        permission_monitor: PermissionMonitorImpl | None = None,
        schema_detector: SchemaDetectorImpl | None = None,
    ) -> None:
        """Initialize the setup wizard.

        Args:
            console: Rich console for output. Creates default if not provided.
            permission_monitor: Permission monitor. Creates default if not provided.
            schema_detector: Schema detector. Creates default if not provided.
        """
        self.console = console or Console()
        self.permission_monitor = permission_monitor or PermissionMonitorImpl()
        self.schema_detector = schema_detector or SchemaDetectorImpl()
        self._checks: list[CheckResult] = []

    def run(self, check_only: bool = False) -> SetupResult:
        """Run the setup wizard.

        Args:
            check_only: If True, only check status without making changes.

        Returns:
            SetupResult with all check results.
        """
        self._checks = []
        config_created = False
        config_path = None

        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold blue]JARVIS Setup Wizard[/bold blue]\nLocal-first AI assistant for macOS",
                border_style="blue",
            )
        )
        self.console.print()

        # Run all checks
        self._check_platform()
        self._check_permissions()
        self._check_database()
        memory_mode = self._check_memory()
        self._check_model()

        # Initialize config if not check-only
        if not check_only:
            config_created, config_path = self._init_config(memory_mode)

        # Print health report
        self._print_health_report()

        # Determine overall success
        failures = [c for c in self._checks if c.status == CheckStatus.FAIL]
        success = len(failures) == 0

        return SetupResult(
            success=success,
            checks=self._checks,
            config_created=config_created,
            config_path=config_path,
        )

    def _check_platform(self) -> None:
        """Check if running on macOS."""
        system = platform.system()

        if system == "Darwin":
            version = platform.mac_ver()[0]
            self._checks.append(
                CheckResult(
                    name="Platform",
                    status=CheckStatus.PASS,
                    message=f"macOS {version}",
                    details="JARVIS is optimized for Apple Silicon Macs",
                )
            )
        else:
            self._checks.append(
                CheckResult(
                    name="Platform",
                    status=CheckStatus.WARN,
                    message=f"Running on {system}",
                    details="JARVIS is designed for macOS. Some features may not work.",
                )
            )

    def _check_permissions(self) -> None:
        """Check required macOS permissions."""
        # Only check FDA - other permissions checked on demand
        fda_status = self.permission_monitor.check_permission(Permission.FULL_DISK_ACCESS)

        if fda_status.granted:
            self._checks.append(
                CheckResult(
                    name="Full Disk Access",
                    status=CheckStatus.PASS,
                    message="Permission granted",
                    details="Can access iMessage database",
                )
            )
        else:
            self._checks.append(
                CheckResult(
                    name="Full Disk Access",
                    status=CheckStatus.FAIL,
                    message="Permission denied",
                    details="Cannot access protected files",
                    fix_instructions=fda_status.fix_instructions,
                )
            )

    def _check_database(self) -> None:
        """Check iMessage database accessibility and schema."""
        db_path = CHAT_DB_PATH

        if not db_path.exists():
            self._checks.append(
                CheckResult(
                    name="iMessage Database",
                    status=CheckStatus.WARN,
                    message="Database not found",
                    details=f"Expected at {db_path}",
                    fix_instructions="Ensure iMessage is set up on this Mac",
                )
            )
            return

        # Check schema
        schema_info = self.schema_detector.detect(str(db_path))

        if schema_info.compatible and schema_info.known_schema:
            self._checks.append(
                CheckResult(
                    name="iMessage Database",
                    status=CheckStatus.PASS,
                    message=f"Schema {schema_info.version} detected",
                    details=f"Found {len(schema_info.tables)} tables, fully compatible",
                )
            )
        elif schema_info.compatible:
            self._checks.append(
                CheckResult(
                    name="iMessage Database",
                    status=CheckStatus.WARN,
                    message=f"Unknown schema version: {schema_info.version}",
                    details="Database structure appears compatible",
                )
            )
        else:
            self._checks.append(
                CheckResult(
                    name="iMessage Database",
                    status=CheckStatus.FAIL,
                    message="Incompatible schema",
                    details="Required tables not found",
                    fix_instructions="Database may be corrupted or from unsupported macOS version",
                )
            )

    def _check_memory(self) -> MemoryMode:
        """Check system memory and determine recommended mode.

        Returns:
            Recommended MemoryMode based on available memory.
        """
        try:
            from core.memory.controller import get_memory_controller

            controller = get_memory_controller()
            state = controller.get_state()
            mode = controller.get_mode()

            mode_desc = {
                MemoryMode.FULL: "All features enabled, concurrent model loading",
                MemoryMode.LITE: "Sequential loading, reduced context window",
                MemoryMode.MINIMAL: "Template-only mode, minimal memory usage",
            }

            available_gb = state.available_mb / 1024
            self._checks.append(
                CheckResult(
                    name="System Memory",
                    status=CheckStatus.PASS,
                    message=f"{available_gb:.1f}GB available - {mode.value.upper()} mode",
                    details=mode_desc.get(mode, ""),
                )
            )
            return mode

        except ImportError:
            # Memory controller not available, estimate from psutil
            import psutil

            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)

            if available_gb >= 8:
                mode = MemoryMode.FULL
            elif available_gb >= 4:
                mode = MemoryMode.LITE
            else:
                mode = MemoryMode.MINIMAL

            mode_msg = f"{available_gb:.1f}GB available - {mode.value.upper()} mode recommended"
            self._checks.append(
                CheckResult(
                    name="System Memory",
                    status=CheckStatus.PASS,
                    message=mode_msg,
                )
            )
            return mode

    def _check_model(self) -> None:
        """Check if the default model is available."""
        model_path = DEFAULT_MODEL_PATH

        # Check if model is downloaded via huggingface cache
        try:
            from huggingface_hub import try_to_load_from_cache

            # Try to find the model config file in cache
            result = try_to_load_from_cache(model_path, "config.json")
            model_downloaded = result is not None

        except ImportError:
            # huggingface_hub not available, check local path
            local_path = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = model_path.replace("/", "--")
            model_downloaded = any(local_path.glob(f"models--{model_dir}*"))

        except Exception:
            # Any error means we couldn't verify
            model_downloaded = False

        if model_downloaded:
            self._checks.append(
                CheckResult(
                    name="Default Model",
                    status=CheckStatus.PASS,
                    message=f"{model_path}",
                    details=f"Estimated memory: {DEFAULT_MODEL_MEMORY_MB}MB",
                )
            )
        else:
            self._checks.append(
                CheckResult(
                    name="Default Model",
                    status=CheckStatus.WARN,
                    message=f"{model_path}",
                    details="Model not found in cache",
                    fix_instructions=(
                        f"Download with: huggingface-cli download {model_path}\n"
                        "The model will be downloaded automatically on first use."
                    ),
                )
            )

    def _init_config(self, memory_mode: MemoryMode) -> tuple[bool, Path | None]:
        """Initialize JARVIS configuration.

        Uses the unified config system from jarvis.config which handles
        migration from older config versions automatically.

        Args:
            memory_mode: Recommended memory mode.

        Returns:
            Tuple of (config_created, config_path).
        """
        # Check if config already exists - load_config handles migration automatically
        if JARVIS_CONFIG_FILE.exists():
            try:
                config = load_config(JARVIS_CONFIG_FILE)
                # Re-save to apply any migrations
                if save_config(config, JARVIS_CONFIG_FILE):
                    self._checks.append(
                        CheckResult(
                            name="Configuration",
                            status=CheckStatus.PASS,
                            message="Existing config preserved",
                            details=str(JARVIS_CONFIG_FILE),
                        )
                    )
                    return False, JARVIS_CONFIG_FILE
            except Exception as e:
                logger.warning(f"Error reading existing config: {e}")

        # Create new config with defaults
        try:
            config = JarvisConfig(
                model_path=DEFAULT_MODEL_PATH,
                template_similarity_threshold=DEFAULT_TEMPLATE_THRESHOLD,
            )

            if save_config(config, JARVIS_CONFIG_FILE):
                self._checks.append(
                    CheckResult(
                        name="Configuration",
                        status=CheckStatus.PASS,
                        message="Config created",
                        details=str(JARVIS_CONFIG_FILE),
                    )
                )
                return True, JARVIS_CONFIG_FILE
            else:
                raise OSError("save_config returned False")

        except OSError as e:
            self._checks.append(
                CheckResult(
                    name="Configuration",
                    status=CheckStatus.FAIL,
                    message="Failed to create config",
                    details=str(e),
                    fix_instructions=f"Ensure write permission for {JARVIS_CONFIG_DIR}",
                )
            )
            return False, None

    def _print_health_report(self) -> None:
        """Print the health report summary."""
        self.console.print()

        # Create summary table
        table = Table(title="Setup Check Results", show_header=True, header_style="bold")
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        status_icons = {
            CheckStatus.PASS: "[green]PASS[/green]",
            CheckStatus.WARN: "[yellow]WARN[/yellow]",
            CheckStatus.FAIL: "[red]FAIL[/red]",
            CheckStatus.SKIP: "[dim]SKIP[/dim]",
        }

        for check in self._checks:
            status = status_icons.get(check.status, check.status.value)
            details = check.message
            if check.details:
                details += f"\n[dim]{check.details}[/dim]"
            table.add_row(check.name, status, details)

        self.console.print(table)
        self.console.print()

        # Print fix instructions for failures
        failures = [c for c in self._checks if c.status == CheckStatus.FAIL]
        warnings = [c for c in self._checks if c.status == CheckStatus.WARN]

        if failures:
            self.console.print("[red]Issues requiring attention:[/red]")
            for check in failures:
                if check.fix_instructions:
                    self.console.print(f"\n[bold]{check.name}:[/bold]")
                    self.console.print(f"  {check.fix_instructions}")
            self.console.print()

        if warnings:
            self.console.print("[yellow]Warnings:[/yellow]")
            for check in warnings:
                if check.fix_instructions:
                    self.console.print(f"\n[bold]{check.name}:[/bold]")
                    self.console.print(f"  {check.fix_instructions}")
            self.console.print()

        # Print overall status
        if not failures:
            self.console.print(
                Panel.fit(
                    "[green]Setup complete![/green]\nJARVIS is ready to use.",
                    border_style="green",
                )
            )
        else:
            self.console.print(
                Panel.fit(
                    "[red]Setup incomplete[/red]\n"
                    "Please resolve the issues above before using JARVIS.",
                    border_style="red",
                )
            )


def open_system_preferences_fda() -> bool:
    """Open System Preferences to Full Disk Access pane.

    Returns:
        True if opened successfully.
    """
    if platform.system() != "Darwin":
        return False

    try:
        # macOS Ventura+ uses System Settings
        subprocess.run(
            [
                "open",
                "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        # Fallback for older macOS
        try:
            subprocess.run(
                ["open", "/System/Library/PreferencePanes/Security.prefPane"],
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False


def run_setup(check_only: bool = False) -> SetupResult:
    """Run the JARVIS setup wizard.

    Args:
        check_only: If True, only check status without making changes.

    Returns:
        SetupResult with all check results.
    """
    wizard = SetupWizard()
    return wizard.run(check_only=check_only)


def main() -> int:
    """Main entry point for the setup wizard.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="JARVIS Setup Wizard - validate environment and configure JARVIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m jarvis.setup          # Run full setup
  python -m jarvis.setup --check  # Just check, don't modify
        """,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check status without making changes",
    )
    parser.add_argument(
        "--open-preferences",
        action="store_true",
        help="Open System Preferences to Full Disk Access",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.open_preferences:
        if open_system_preferences_fda():
            print("Opened System Preferences to Full Disk Access")
            return 0
        else:
            print("Failed to open System Preferences")
            return 1

    result = run_setup(check_only=args.check)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
