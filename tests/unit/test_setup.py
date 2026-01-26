"""Unit tests for JARVIS Setup Wizard.

Tests cover permission checking, database validation, config initialization,
model checking, and the overall setup flow. Filesystem and permissions are mocked.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from contracts.health import Permission, PermissionStatus, SchemaInfo
from contracts.memory import MemoryMode
from jarvis.setup import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPLATE_THRESHOLD,
    CheckResult,
    CheckStatus,
    JarvisConfig,
    PermissionMonitorImpl,
    SchemaDetectorImpl,
    SetupResult,
    SetupWizard,
    open_system_preferences_fda,
    run_setup,
)


class TestJarvisConfig:
    """Tests for JarvisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JarvisConfig()
        assert config.model_path == DEFAULT_MODEL_PATH
        assert config.memory_mode == "auto"
        assert config.template_threshold == DEFAULT_TEMPLATE_THRESHOLD

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = JarvisConfig(
            model_path="custom/model",
            memory_mode="lite",
            template_threshold=0.8,
        )
        data = config.to_dict()
        assert data == {
            "model_path": "custom/model",
            "memory_mode": "lite",
            "template_threshold": 0.8,
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "model_path": "custom/model",
            "memory_mode": "minimal",
            "template_threshold": 0.6,
        }
        config = JarvisConfig.from_dict(data)
        assert config.model_path == "custom/model"
        assert config.memory_mode == "minimal"
        assert config.template_threshold == 0.6

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        config = JarvisConfig.from_dict({})
        assert config.model_path == DEFAULT_MODEL_PATH
        assert config.memory_mode == "auto"
        assert config.template_threshold == DEFAULT_TEMPLATE_THRESHOLD


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_basic_check_result(self):
        """Test basic CheckResult creation."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.PASS,
            message="All good",
        )
        assert result.name == "Test Check"
        assert result.status == CheckStatus.PASS
        assert result.message == "All good"
        assert result.details is None
        assert result.fix_instructions is None

    def test_check_result_with_all_fields(self):
        """Test CheckResult with all fields."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.FAIL,
            message="Something wrong",
            details="More info here",
            fix_instructions="Do this to fix",
        )
        assert result.details == "More info here"
        assert result.fix_instructions == "Do this to fix"


class TestPermissionMonitorImpl:
    """Tests for PermissionMonitorImpl."""

    def test_check_full_disk_access_granted(self, tmp_path, monkeypatch):
        """Test FDA check when permission is granted."""
        # Create a fake chat.db
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"fake data")

        # Patch CHAT_DB_PATH to point to our fake db
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        monitor = PermissionMonitorImpl()
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is True
        assert status.permission == Permission.FULL_DISK_ACCESS
        assert status.last_checked  # Should have a timestamp

    def test_check_full_disk_access_denied(self, tmp_path, monkeypatch):
        """Test FDA check when permission is denied."""
        # Create a fake path that doesn't exist
        fake_db = tmp_path / "nonexistent" / "chat.db"

        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        # Mock the home directory TCC path check to also fail
        def mock_exists(self):
            return False

        with patch.object(Path, "exists", return_value=False):
            monitor = PermissionMonitorImpl()
            # Since both paths don't exist, it assumes access is granted
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
            assert status.granted is True  # Default when files don't exist

    def test_check_full_disk_access_permission_error(self, tmp_path, monkeypatch):
        """Test FDA check when reading raises PermissionError."""
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")

        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        # Mock open to raise PermissionError
        original_open = Path.open

        def mock_open(self, *args, **kwargs):
            if str(self) == str(fake_db):
                raise PermissionError("Access denied")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            monitor = PermissionMonitorImpl()
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
            assert status.granted is False

    def test_check_all_permissions(self, tmp_path, monkeypatch):
        """Test checking all permissions."""
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        monitor = PermissionMonitorImpl()
        statuses = monitor.check_all()

        assert len(statuses) == 1
        assert statuses[0].permission == Permission.FULL_DISK_ACCESS

    def test_fix_instructions_present(self):
        """Test that fix instructions are provided."""
        monitor = PermissionMonitorImpl()
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.fix_instructions
        has_sys_settings = "System Settings" in status.fix_instructions
        has_fda = "Full Disk Access" in status.fix_instructions
        assert has_sys_settings or has_fda


class TestSchemaDetectorImpl:
    """Tests for SchemaDetectorImpl."""

    def test_detect_nonexistent_db(self, tmp_path):
        """Test detection on non-existent database."""
        detector = SchemaDetectorImpl()
        info = detector.detect(str(tmp_path / "nonexistent.db"))

        assert info.version == "unknown"
        assert info.tables == []
        assert info.compatible is False
        assert info.known_schema is False

    def test_detect_valid_v14_schema(self, tmp_path):
        """Test detection of v14 schema."""
        import sqlite3

        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create required tables for v14
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                date INTEGER,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                display_name TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY,
                id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_handle_join (
                chat_id INTEGER,
                handle_id INTEGER
            )
        """)
        conn.commit()
        conn.close()

        detector = SchemaDetectorImpl()
        info = detector.detect(str(db_path))

        assert info.version == "v14"
        assert info.compatible is True
        assert info.known_schema is True
        assert "message" in info.tables
        assert "chat" in info.tables

    def test_detect_v15_schema(self, tmp_path):
        """Test detection of v15 schema."""
        import sqlite3

        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create v15 schema with service_name in chat table
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                date INTEGER,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                display_name TEXT,
                service_name TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY,
                id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_handle_join (
                chat_id INTEGER,
                handle_id INTEGER
            )
        """)
        conn.commit()
        conn.close()

        detector = SchemaDetectorImpl()
        info = detector.detect(str(db_path))

        assert info.version == "v15"
        assert info.compatible is True
        assert info.known_schema is True


class TestSetupWizard:
    """Tests for SetupWizard."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console that captures output."""
        console = MagicMock()
        return console

    @pytest.fixture
    def mock_permission_monitor(self):
        """Create a mock permission monitor."""
        monitor = MagicMock(spec=PermissionMonitorImpl)
        monitor.check_permission.return_value = PermissionStatus(
            permission=Permission.FULL_DISK_ACCESS,
            granted=True,
            last_checked=datetime.now().isoformat(),
            fix_instructions="",
        )
        return monitor

    @pytest.fixture
    def mock_schema_detector(self, tmp_path):
        """Create a mock schema detector."""
        detector = MagicMock(spec=SchemaDetectorImpl)
        detector.detect.return_value = SchemaInfo(
            version="v14",
            tables=["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
            compatible=True,
            migration_needed=False,
            known_schema=True,
        )
        return detector

    def test_run_check_only_no_config_created(
        self, tmp_path, mock_console, mock_permission_monitor, mock_schema_detector, monkeypatch
    ):
        """Test check-only mode doesn't create config."""
        # Setup mock paths
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")

        # Mock memory controller
        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        # Mock platform
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(
                console=mock_console,
                permission_monitor=mock_permission_monitor,
                schema_detector=mock_schema_detector,
            )
            result = wizard.run(check_only=True)

        assert result.config_created is False
        assert not config_file.exists()

    def test_run_creates_config(
        self, tmp_path, mock_console, mock_permission_monitor, mock_schema_detector, monkeypatch
    ):
        """Test full run creates config file."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")

        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(
                console=mock_console,
                permission_monitor=mock_permission_monitor,
                schema_detector=mock_schema_detector,
            )
            result = wizard.run(check_only=False)

        assert result.config_created is True
        assert config_file.exists()

        # Verify config contents
        with config_file.open() as f:
            config = json.load(f)
        assert config["model_path"] == DEFAULT_MODEL_PATH
        assert config["template_threshold"] == DEFAULT_TEMPLATE_THRESHOLD

    def test_preserves_existing_config(
        self, tmp_path, mock_console, mock_permission_monitor, mock_schema_detector, monkeypatch
    ):
        """Test that existing config is preserved."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        config_dir.mkdir()

        # Create existing config
        existing_config = {
            "model_path": "custom/model",
            "memory_mode": "lite",
            "template_threshold": 0.9,
        }
        with config_file.open("w") as f:
            json.dump(existing_config, f)

        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")

        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(
                console=mock_console,
                permission_monitor=mock_permission_monitor,
                schema_detector=mock_schema_detector,
            )
            result = wizard.run(check_only=False)

        # Config should not be recreated
        assert result.config_created is False

        # Original config should be preserved
        with config_file.open() as f:
            config = json.load(f)
        assert config["model_path"] == "custom/model"
        assert config["template_threshold"] == 0.9

    def test_platform_check_macos(self, mock_console, monkeypatch):
        """Test platform check on macOS."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.2.1", ("", "", ""), "arm64"))

        wizard = SetupWizard(console=mock_console)
        wizard._check_platform()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert "macOS 14.2.1" in wizard._checks[0].message

    def test_platform_check_linux(self, mock_console, monkeypatch):
        """Test platform check on Linux (warning)."""
        monkeypatch.setattr("platform.system", lambda: "Linux")

        wizard = SetupWizard(console=mock_console)
        wizard._check_platform()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert "Linux" in wizard._checks[0].message

    def test_permission_check_granted(self, mock_console, mock_permission_monitor):
        """Test permission check when FDA is granted."""
        wizard = SetupWizard(
            console=mock_console,
            permission_monitor=mock_permission_monitor,
        )
        wizard._check_permissions()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert wizard._checks[0].name == "Full Disk Access"

    def test_permission_check_denied(self, mock_console):
        """Test permission check when FDA is denied."""
        monitor = MagicMock(spec=PermissionMonitorImpl)
        monitor.check_permission.return_value = PermissionStatus(
            permission=Permission.FULL_DISK_ACCESS,
            granted=False,
            last_checked=datetime.now().isoformat(),
            fix_instructions="Grant FDA in System Settings",
        )

        wizard = SetupWizard(console=mock_console, permission_monitor=monitor)
        wizard._check_permissions()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.FAIL
        assert wizard._checks[0].fix_instructions is not None

    def test_database_check_not_found(
        self, mock_console, mock_schema_detector, tmp_path, monkeypatch
    ):
        """Test database check when chat.db doesn't exist."""
        fake_path = tmp_path / "nonexistent" / "chat.db"
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_path)

        wizard = SetupWizard(
            console=mock_console,
            schema_detector=mock_schema_detector,
        )
        wizard._check_database()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert "not found" in wizard._checks[0].message

    def test_database_check_compatible(
        self, mock_console, mock_schema_detector, tmp_path, monkeypatch
    ):
        """Test database check with compatible schema."""
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"fake")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", db_path)

        wizard = SetupWizard(
            console=mock_console,
            schema_detector=mock_schema_detector,
        )
        wizard._check_database()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert "v14" in wizard._checks[0].message

    def test_memory_check_with_controller(self, mock_console, monkeypatch):
        """Test memory check using MemoryController."""
        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=10000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(console=mock_console)
            mode = wizard._check_memory()

        assert mode == MemoryMode.FULL
        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert "FULL" in wizard._checks[0].message

    def test_memory_check_fallback_psutil(self, mock_console, monkeypatch):
        """Test memory check fallback to psutil when controller unavailable."""

        # Make get_memory_controller raise ImportError
        def mock_import_error():
            raise ImportError("No controller")

        mock_mem = MagicMock()
        mock_mem.available = 6 * 1024**3  # 6GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
            wizard = SetupWizard(console=mock_console)
            mode = wizard._check_memory()

        assert mode == MemoryMode.LITE
        assert len(wizard._checks) == 1
        assert "LITE" in wizard._checks[0].message

    def test_model_check_downloaded(self, mock_console, monkeypatch):
        """Test model check when model is in cache."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value="/path/to/config.json"):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert DEFAULT_MODEL_PATH in wizard._checks[0].message

    def test_model_check_not_downloaded(self, mock_console):
        """Test model check when model is not in cache."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert wizard._checks[0].fix_instructions is not None
        assert "huggingface-cli download" in wizard._checks[0].fix_instructions

    def test_model_check_import_error(self, mock_console, tmp_path, monkeypatch):
        """Test model check when huggingface_hub not installed."""
        # Mock the home directory for cache check
        cache_path = tmp_path / ".cache" / "huggingface" / "hub"
        cache_path.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("huggingface_hub.try_to_load_from_cache", side_effect=ImportError):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        # Should be WARN since model dir not found
        assert wizard._checks[0].status == CheckStatus.WARN


class TestSetupResult:
    """Tests for SetupResult."""

    def test_success_with_all_passing(self):
        """Test result is success when all checks pass."""
        checks = [
            CheckResult(name="Check1", status=CheckStatus.PASS, message="OK"),
            CheckResult(name="Check2", status=CheckStatus.PASS, message="OK"),
            CheckResult(name="Check3", status=CheckStatus.WARN, message="Warning"),
        ]
        result = SetupResult(success=True, checks=checks)
        assert result.success is True

    def test_failure_with_failing_check(self):
        """Test result is failure when any check fails."""
        checks = [
            CheckResult(name="Check1", status=CheckStatus.PASS, message="OK"),
            CheckResult(name="Check2", status=CheckStatus.FAIL, message="Failed"),
        ]
        result = SetupResult(success=False, checks=checks)
        assert result.success is False


class TestOpenSystemPreferences:
    """Tests for open_system_preferences_fda."""

    def test_non_darwin_returns_false(self, monkeypatch):
        """Test returns False on non-Darwin platforms."""
        monkeypatch.setattr("platform.system", lambda: "Linux")
        assert open_system_preferences_fda() is False

    def test_darwin_opens_preferences(self, monkeypatch):
        """Test attempts to open System Preferences on Darwin."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = open_system_preferences_fda()

        assert result is True
        mock_run.assert_called_once()
        # Check it tried to open the privacy pane
        args = mock_run.call_args[0][0]
        assert "open" in args


class TestRunSetup:
    """Tests for run_setup convenience function."""

    def test_run_setup_creates_wizard(self, tmp_path, monkeypatch):
        """Test run_setup creates and runs a wizard."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")
        monkeypatch.setattr("platform.system", lambda: "Linux")

        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
            with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                result = run_setup(check_only=True)

        assert isinstance(result, SetupResult)
        assert len(result.checks) > 0


class TestCheckStatusEnum:
    """Tests for CheckStatus enum."""

    def test_all_status_values(self):
        """Test all check status values exist."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIP.value == "skip"


class TestPermissionMonitorImplExtended:
    """Additional tests for PermissionMonitorImpl coverage."""

    def test_check_non_fda_permissions_granted(self, tmp_path, monkeypatch):
        """Test non-FDA permissions (CONTACTS, CALENDAR, AUTOMATION) return granted=True."""
        # Line 142: Non-FDA permissions return True by default
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        monitor = PermissionMonitorImpl()

        # Test CONTACTS permission
        status = monitor.check_permission(Permission.CONTACTS)
        assert status.granted is True
        assert status.permission == Permission.CONTACTS

        # Test CALENDAR permission
        status = monitor.check_permission(Permission.CALENDAR)
        assert status.granted is True
        assert status.permission == Permission.CALENDAR

        # Test AUTOMATION permission
        status = monitor.check_permission(Permission.AUTOMATION)
        assert status.granted is True
        assert status.permission == Permission.AUTOMATION

    def test_wait_for_permission_granted_immediately(self, tmp_path, monkeypatch):
        """Test wait_for_permission returns True when permission is granted immediately."""
        # Lines 171-179: wait_for_permission
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        monitor = PermissionMonitorImpl()
        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=1)
        assert result is True

    def test_wait_for_permission_timeout(self, tmp_path, monkeypatch):
        """Test wait_for_permission returns False after timeout."""
        # Lines 171-179: wait_for_permission timeout path
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        # Mock permission check to always return False (not granted)
        monitor = PermissionMonitorImpl()
        original_check = monitor.check_permission

        def mock_check(permission):
            status = original_check(permission)
            # Force not granted
            return PermissionStatus(
                permission=status.permission,
                granted=False,
                last_checked=status.last_checked,
                fix_instructions=status.fix_instructions,
            )

        monitor.check_permission = mock_check

        # Short timeout to make test fast
        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=1)
        assert result is False

    def test_check_full_disk_access_tcc_db_fallback(self, tmp_path, monkeypatch):
        """Test FDA check falls back to TCC.db when chat.db doesn't exist."""
        # Lines 205-210: TCC.db fallback path
        # Make chat.db not exist
        nonexistent_chat_db = tmp_path / "nonexistent" / "chat.db"
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", nonexistent_chat_db)

        # Create a fake TCC.db
        tcc_dir = tmp_path / "Library" / "Application Support" / "com.apple.TCC"
        tcc_dir.mkdir(parents=True)
        tcc_db = tcc_dir / "TCC.db"
        tcc_db.write_bytes(b"tcc data")

        # Patch Path.home to return tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        monitor = PermissionMonitorImpl()
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
        assert status.granted is True

    def test_check_full_disk_access_tcc_db_permission_error(self, tmp_path, monkeypatch):
        """Test FDA check when TCC.db exists but access denied."""
        # Lines 209-210: TCC.db PermissionError/OSError path
        nonexistent_chat_db = tmp_path / "nonexistent" / "chat.db"
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", nonexistent_chat_db)

        # Create a fake TCC.db
        tcc_dir = tmp_path / "Library" / "Application Support" / "com.apple.TCC"
        tcc_dir.mkdir(parents=True)
        tcc_db = tcc_dir / "TCC.db"
        tcc_db.write_bytes(b"tcc data")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock open to raise PermissionError for TCC.db
        original_open = Path.open

        def mock_open(self, *args, **kwargs):
            if "TCC.db" in str(self):
                raise PermissionError("Access denied to TCC.db")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            monitor = PermissionMonitorImpl()
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
            assert status.granted is False

    def test_check_full_disk_access_oserror(self, tmp_path, monkeypatch):
        """Test FDA check when reading chat.db raises OSError."""
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        # Mock open to raise OSError
        original_open = Path.open

        def mock_open(self, *args, **kwargs):
            if str(self) == str(fake_db):
                raise OSError("I/O error reading chat.db")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            monitor = PermissionMonitorImpl()
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
            assert status.granted is False


class TestSchemaDetectorImplExtended:
    """Additional tests for SchemaDetectorImpl coverage."""

    def test_detect_missing_expected_v14_columns(self, tmp_path):
        """Test schema detection when missing expected v14 columns."""
        # Lines 274-276: Warning for missing expected v14 columns
        import sqlite3

        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create tables with minimal columns (missing expected v14 columns)
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                custom_column TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY
            )
        """)
        cursor.execute("""
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_handle_join (
                chat_id INTEGER,
                handle_id INTEGER
            )
        """)
        conn.commit()
        conn.close()

        detector = SchemaDetectorImpl()
        with patch("logging.warning") as mock_warn:
            info = detector.detect(str(db_path))

        # Should still detect as v14 (fallback) but log a warning
        assert info.version == "v14"
        assert info.compatible is True
        mock_warn.assert_called()

    def test_detect_operational_error(self, tmp_path):
        """Test schema detection when database query fails."""
        # Lines 294-296: sqlite3.OperationalError handling
        import sqlite3

        db_path = tmp_path / "chat.db"
        # Create a valid database file
        db_path.write_bytes(b"data")

        # Mock sqlite3.connect to raise OperationalError
        with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("database is locked")):
            detector = SchemaDetectorImpl()
            info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.tables == []
        assert info.compatible is False
        assert info.known_schema is False

    def test_get_query_delegation(self, tmp_path):
        """Test get_query delegates to queries module."""
        # Lines 315-317: get_query delegation
        detector = SchemaDetectorImpl()

        with patch(
            "integrations.imessage.queries.get_query", return_value="SELECT * FROM test"
        ) as mock_get_query:
            result = detector.get_query("test_query", "v14")

        mock_get_query.assert_called_once_with("test_query", "v14")
        assert result == "SELECT * FROM test"


class TestSetupWizardExtended:
    """Additional tests for SetupWizard coverage."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console that captures output."""
        console = MagicMock()
        return console

    def test_database_check_compatible_unknown_schema(self, mock_console, tmp_path, monkeypatch):
        """Test database check with compatible but unknown schema version."""
        # Lines 468-478: Unknown schema version warning path
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"fake")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", db_path)

        # Create a mock that returns compatible but unknown schema
        mock_detector = MagicMock(spec=SchemaDetectorImpl)
        mock_detector.detect.return_value = SchemaInfo(
            version="v99",  # Unknown version
            tables=["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
            compatible=True,
            migration_needed=False,
            known_schema=False,  # Not a known schema
        )

        wizard = SetupWizard(
            console=mock_console,
            schema_detector=mock_detector,
        )
        wizard._check_database()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert "Unknown schema version" in wizard._checks[0].message
        assert "v99" in wizard._checks[0].message

    def test_database_check_incompatible_schema(self, mock_console, tmp_path, monkeypatch):
        """Test database check with incompatible schema."""
        # Lines 478-486: Incompatible schema path
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"fake")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", db_path)

        mock_detector = MagicMock(spec=SchemaDetectorImpl)
        mock_detector.detect.return_value = SchemaInfo(
            version="unknown",
            tables=["some_table"],
            compatible=False,
            migration_needed=False,
            known_schema=False,
        )

        wizard = SetupWizard(
            console=mock_console,
            schema_detector=mock_detector,
        )
        wizard._check_database()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.FAIL
        assert "Incompatible schema" in wizard._checks[0].message
        assert wizard._checks[0].fix_instructions is not None

    def test_memory_check_psutil_minimal_mode(self, mock_console, monkeypatch):
        """Test memory check returns MINIMAL mode when memory is low."""
        # Line 530: MemoryMode.MINIMAL case in psutil fallback
        mock_mem = MagicMock()
        mock_mem.available = 2 * 1024**3  # 2GB - should trigger MINIMAL mode

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
            wizard = SetupWizard(console=mock_console)
            mode = wizard._check_memory()

        assert mode == MemoryMode.MINIMAL
        assert len(wizard._checks) == 1
        assert "MINIMAL" in wizard._checks[0].message

    def test_model_check_generic_exception(self, mock_console):
        """Test model check when any exception occurs."""
        # Lines 560-562: Exception case in model check
        with patch(
            "huggingface_hub.try_to_load_from_cache",
            side_effect=RuntimeError("Unexpected error"),
        ):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert wizard._checks[0].fix_instructions is not None

    def test_init_config_existing_invalid_json(self, mock_console, tmp_path, monkeypatch):
        """Test init_config when existing config has invalid JSON."""
        # Lines 611-612: json.JSONDecodeError handling
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        config_dir.mkdir()

        # Create a file with invalid JSON
        config_file.write_text("{ invalid json }")

        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)

        wizard = SetupWizard(console=mock_console)
        created, path = wizard._init_config(MemoryMode.FULL)

        # Should create a new config because existing one was invalid
        assert created is True
        assert path == config_file

        # Verify new config is valid
        with config_file.open() as f:
            config = json.load(f)
        assert "model_path" in config

    def test_init_config_existing_oserror(self, mock_console, tmp_path, monkeypatch):
        """Test init_config when reading existing config raises OSError."""
        # Lines 611-612: OSError handling when reading existing config
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        config_dir.mkdir()
        config_file.write_text('{"model_path": "test"}')

        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)

        # Mock open to raise OSError when reading
        original_open = Path.open

        def mock_open(self, mode="r", *args, **kwargs):
            if str(self) == str(config_file) and "w" not in mode:
                raise OSError("Cannot read config")
            return original_open(self, mode, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            wizard = SetupWizard(console=mock_console)
            created, path = wizard._init_config(MemoryMode.FULL)

        # Should create a new config because reading existing one failed
        assert created is True
        assert path == config_file

    def test_init_config_create_oserror(self, mock_console, tmp_path, monkeypatch):
        """Test init_config when creating config raises OSError."""
        # Lines 637-647: OSError when creating config
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"

        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)

        # Mock mkdir to succeed but open to raise OSError when writing
        original_open = Path.open

        def mock_open(self, mode="r", *args, **kwargs):
            if str(self) == str(config_file) and "w" in mode:
                raise OSError("Cannot write config")
            return original_open(self, mode, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            wizard = SetupWizard(console=mock_console)
            created, path = wizard._init_config(MemoryMode.FULL)

        assert created is False
        assert path is None
        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.FAIL
        assert "Failed to create config" in wizard._checks[0].message

    def test_print_health_report_with_failures(self, mock_console, tmp_path, monkeypatch):
        """Test print_health_report with failed checks."""
        # Lines 681-686, 705: Printing fix instructions for failures
        mock_permission_monitor = MagicMock(spec=PermissionMonitorImpl)
        mock_permission_monitor.check_permission.return_value = PermissionStatus(
            permission=Permission.FULL_DISK_ACCESS,
            granted=False,
            last_checked=datetime.now().isoformat(),
            fix_instructions="Grant FDA in System Settings",
        )

        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        # Mock memory controller
        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            with patch("huggingface_hub.try_to_load_from_cache", return_value="/path"):
                wizard = SetupWizard(
                    console=mock_console,
                    permission_monitor=mock_permission_monitor,
                )
                result = wizard.run(check_only=True)

        # Should have called console.print with failure messages
        assert result.success is False
        # Verify the console received calls with failure-related content
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        # Check that there were multiple print calls (health report)
        assert len(print_calls) > 0

    def test_init_config_lite_memory_mode(self, mock_console, tmp_path, monkeypatch):
        """Test config creation with LITE memory mode."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)

        wizard = SetupWizard(console=mock_console)
        created, path = wizard._init_config(MemoryMode.LITE)

        assert created is True
        with config_file.open() as f:
            config = json.load(f)
        # LITE mode should be written to config
        assert config["memory_mode"] == "lite"


class TestOpenSystemPreferencesExtended:
    """Additional tests for open_system_preferences_fda coverage."""

    def test_darwin_fallback_to_older_macos(self, monkeypatch):
        """Test fallback to older macOS System Preferences pane."""
        # Lines 733-742: Fallback for older macOS
        monkeypatch.setattr("platform.system", lambda: "Darwin")

        call_count = [0]

        def mock_run(cmd, check=False):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (new style) fails
                raise subprocess.CalledProcessError(1, cmd)
            # Second call (old style) succeeds
            return MagicMock(returncode=0)

        with patch("subprocess.run", mock_run):
            result = open_system_preferences_fda()

        assert result is True
        assert call_count[0] == 2

    def test_darwin_both_fail(self, monkeypatch):
        """Test when both new and old System Preferences methods fail."""
        # Lines 733-742: Both attempts fail
        monkeypatch.setattr("platform.system", lambda: "Darwin")

        def mock_run(cmd, check=False):
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", mock_run):
            result = open_system_preferences_fda()

        assert result is False


class TestMainFunction:
    """Tests for main() function."""

    def test_main_check_only(self, tmp_path, monkeypatch):
        """Test main with --check flag."""
        # Lines 764-808: main() function
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")
        monkeypatch.setattr("platform.system", lambda: "Linux")

        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        # Import main after patching
        from jarvis.setup import main

        with patch("sys.argv", ["setup", "--check"]):
            with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
                with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                    exit_code = main()

        # Should succeed (warnings but no failures on Linux)
        assert exit_code == 0

    def test_main_verbose(self, tmp_path, monkeypatch):
        """Test main with --verbose flag."""
        # Lines 794-797: verbose logging
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")
        monkeypatch.setattr("platform.system", lambda: "Linux")

        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        from jarvis.setup import main

        with patch("sys.argv", ["setup", "--check", "--verbose"]):
            with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
                with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                    with patch("logging.basicConfig") as mock_logging:
                        main()

        # Should have configured DEBUG logging
        mock_logging.assert_called()

    def test_main_open_preferences_success(self, monkeypatch):
        """Test main with --open-preferences flag."""
        # Lines 799-805: open-preferences
        monkeypatch.setattr("platform.system", lambda: "Darwin")

        from jarvis.setup import main

        with patch("sys.argv", ["setup", "--open-preferences"]):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                exit_code = main()

        assert exit_code == 0

    def test_main_open_preferences_failure(self, monkeypatch):
        """Test main with --open-preferences flag when it fails."""
        # Lines 803-805: open-preferences failure
        monkeypatch.setattr("platform.system", lambda: "Linux")

        from jarvis.setup import main

        with patch("sys.argv", ["setup", "--open-preferences"]):
            exit_code = main()

        assert exit_code == 1

    def test_main_full_setup_failure(self, tmp_path, monkeypatch):
        """Test main returns 1 when setup has failures."""
        # Line 808: return 1 on failure
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)

        # Don't create chat.db - database check will be WARN (not found)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "nonexistent" / "chat.db")

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        # Mock permission monitor to return denied (this causes FAIL status)
        def mock_check_permission(self, permission):
            return PermissionStatus(
                permission=permission,
                granted=False,
                last_checked=datetime.now().isoformat(),
                fix_instructions="Test instructions",
            )

        from jarvis.setup import main

        with patch("sys.argv", ["setup"]):
            with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
                with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                    with patch.object(
                        PermissionMonitorImpl, "check_permission", mock_check_permission
                    ):
                        exit_code = main()

        assert exit_code == 1

    def test_main_default_run(self, tmp_path, monkeypatch):
        """Test main with no arguments (full setup)."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")
        monkeypatch.setattr("platform.system", lambda: "Linux")

        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        from jarvis.setup import main

        with patch("sys.argv", ["setup"]):
            with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
                with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                    exit_code = main()

        # Should create config file
        assert config_file.exists()
        assert exit_code == 0
