"""Unit tests for Workstream 7: Permission Monitor.

Tests TCC permission checking with mocked file access.
"""

import threading
import time
from pathlib import Path
from unittest.mock import patch

from contracts.health import Permission, PermissionStatus
from core.health.permissions import (
    FIX_INSTRUCTIONS,
    TCCPermissionMonitor,
    get_permission_monitor,
    reset_permission_monitor,
)


class TestPermissionStatusDataclass:
    """Tests for PermissionStatus dataclass."""

    def test_permission_status_creation(self):
        """Test creating a PermissionStatus."""
        status = PermissionStatus(
            permission=Permission.FULL_DISK_ACCESS,
            granted=True,
            last_checked="2024-01-01T00:00:00Z",
            fix_instructions="",
        )
        assert status.permission == Permission.FULL_DISK_ACCESS
        assert status.granted is True
        assert status.last_checked == "2024-01-01T00:00:00Z"
        assert status.fix_instructions == ""

    def test_permission_status_with_fix_instructions(self):
        """Test PermissionStatus with fix instructions when not granted."""
        status = PermissionStatus(
            permission=Permission.CONTACTS,
            granted=False,
            last_checked="2024-01-01T00:00:00Z",
            fix_instructions="Grant Contacts access...",
        )
        assert status.granted is False
        assert "Contacts" in status.fix_instructions


class TestPermissionEnum:
    """Tests for Permission enum."""

    def test_all_permissions_defined(self):
        """Verify all expected permissions exist."""
        assert Permission.FULL_DISK_ACCESS.value == "full_disk_access"
        assert Permission.CONTACTS.value == "contacts"
        assert Permission.CALENDAR.value == "calendar"
        assert Permission.AUTOMATION.value == "automation"

    def test_fix_instructions_for_all_permissions(self):
        """Verify fix instructions exist for all permission types."""
        for permission in Permission:
            assert permission in FIX_INSTRUCTIONS
            assert len(FIX_INSTRUCTIONS[permission]) > 0


class TestTCCPermissionMonitorInitialization:
    """Tests for TCCPermissionMonitor initialization."""

    def test_default_initialization(self):
        """Test default paths are set correctly."""
        monitor = TCCPermissionMonitor()
        assert monitor._chat_db_path == Path.home() / "Library" / "Messages" / "chat.db"

    def test_custom_paths(self):
        """Test custom paths can be provided."""
        custom_path = Path("/custom/path/chat.db")
        monitor = TCCPermissionMonitor(chat_db_path=custom_path)
        assert monitor._chat_db_path == custom_path

    def test_cache_starts_empty(self):
        """Test that cache is empty on initialization."""
        monitor = TCCPermissionMonitor()
        assert len(monitor._cache) == 0


class TestUnknownPermissionType:
    """Tests for unknown permission type handling."""

    def test_unknown_permission_returns_false(self, tmp_path):
        """Test that unknown permission types return False (lines 150-151)."""
        from unittest.mock import MagicMock

        monitor = TCCPermissionMonitor(chat_db_path=tmp_path / "chat.db")

        # Create a mock permission that doesn't match any known type
        mock_permission = MagicMock()
        mock_permission.value = "unknown_permission"

        # Directly call the implementation method
        result = monitor._check_permission_impl(mock_permission)

        assert result is False


class TestFullDiskAccessCheck:
    """Tests for Full Disk Access permission check."""

    def test_full_disk_access_granted_when_file_readable(self, tmp_path):
        """Test FULL_DISK_ACCESS returns granted when file is readable."""
        # Create a mock chat.db file
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"SQLite format 3\x00")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is True
        assert status.permission == Permission.FULL_DISK_ACCESS
        assert status.fix_instructions == ""

    def test_full_disk_access_granted_when_file_missing(self, tmp_path):
        """Test FULL_DISK_ACCESS returns granted when file doesn't exist."""
        # Non-existent path - not a permission issue
        missing_path = tmp_path / "nonexistent" / "chat.db"

        monitor = TCCPermissionMonitor(chat_db_path=missing_path)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        # File doesn't exist, so we can't test permission - assume granted
        assert status.granted is True

    def test_full_disk_access_denied_on_permission_error(self, tmp_path):
        """Test FULL_DISK_ACCESS returns denied on PermissionError."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)

        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is False
        assert "Full Disk Access" in status.fix_instructions

    def test_full_disk_access_denied_on_tcc_os_error(self, tmp_path):
        """Test FULL_DISK_ACCESS returns denied on TCC OSError."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)

        # Mock open to raise OSError with TCC message
        with patch(
            "builtins.open",
            side_effect=OSError("Operation not permitted"),
        ):
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is False

    def test_full_disk_access_denied_on_other_os_error(self, tmp_path):
        """Test FULL_DISK_ACCESS returns denied on non-TCC OSError (lines 181-182)."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)

        # Mock open to raise OSError with a different message (not TCC)
        with patch(
            "builtins.open",
            side_effect=OSError("Disk I/O error"),
        ):
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        # Non-TCC OSErrors still return False (conservative approach)
        assert status.granted is False


class TestContactsAccessCheck:
    """Tests for Contacts permission check."""

    def test_contacts_access_granted_when_dir_accessible(self, tmp_path):
        """Test CONTACTS returns granted when directory is accessible."""
        contacts_dir = tmp_path / "AddressBook" / "Sources"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "test.abcddb").touch()

        monitor = TCCPermissionMonitor(contacts_path=contacts_dir)
        status = monitor.check_permission(Permission.CONTACTS)

        assert status.granted is True

    def test_contacts_access_granted_when_dir_missing(self, tmp_path):
        """Test CONTACTS returns granted when directory doesn't exist."""
        missing_path = tmp_path / "nonexistent" / "AddressBook"

        monitor = TCCPermissionMonitor(contacts_path=missing_path)
        status = monitor.check_permission(Permission.CONTACTS)

        # Directory doesn't exist - not a permission issue
        assert status.granted is True

    def test_contacts_access_denied_on_permission_error(self, tmp_path):
        """Test CONTACTS returns denied on PermissionError."""
        contacts_dir = tmp_path / "AddressBook"
        contacts_dir.mkdir()

        monitor = TCCPermissionMonitor(contacts_path=contacts_dir)

        with patch.object(Path, "iterdir", side_effect=PermissionError("Access denied")):
            status = monitor.check_permission(Permission.CONTACTS)

        assert status.granted is False
        assert "Contacts" in status.fix_instructions

    def test_contacts_access_denied_on_tcc_os_error(self, tmp_path):
        """Test CONTACTS returns denied on TCC OSError (lines 200-202)."""
        contacts_dir = tmp_path / "AddressBook"
        contacts_dir.mkdir()

        monitor = TCCPermissionMonitor(contacts_path=contacts_dir)

        # Mock iterdir to raise OSError with TCC message
        with patch.object(Path, "iterdir", side_effect=OSError("Operation not permitted")):
            status = monitor.check_permission(Permission.CONTACTS)

        assert status.granted is False
        assert "Contacts" in status.fix_instructions

    def test_contacts_access_granted_on_other_os_error(self, tmp_path):
        """Test CONTACTS returns granted on non-TCC OSError (line 203)."""
        contacts_dir = tmp_path / "AddressBook"
        contacts_dir.mkdir()

        monitor = TCCPermissionMonitor(contacts_path=contacts_dir)

        # Mock iterdir to raise OSError with a different message (not TCC)
        with patch.object(Path, "iterdir", side_effect=OSError("Network error")):
            status = monitor.check_permission(Permission.CONTACTS)

        # Non-TCC OSErrors return True (not a permission issue)
        assert status.granted is True


class TestCalendarAccessCheck:
    """Tests for Calendar permission check."""

    def test_calendar_access_granted_when_dir_accessible(self, tmp_path):
        """Test CALENDAR returns granted when directory is accessible."""
        calendar_dir = tmp_path / "Calendars"
        calendar_dir.mkdir()
        (calendar_dir / "test.calendar").touch()

        monitor = TCCPermissionMonitor(calendar_path=calendar_dir)
        status = monitor.check_permission(Permission.CALENDAR)

        assert status.granted is True

    def test_calendar_access_granted_when_dir_missing(self, tmp_path):
        """Test CALENDAR returns granted when directory doesn't exist."""
        missing_path = tmp_path / "nonexistent" / "Calendars"

        monitor = TCCPermissionMonitor(calendar_path=missing_path)
        status = monitor.check_permission(Permission.CALENDAR)

        # Directory doesn't exist - not a permission issue
        assert status.granted is True

    def test_calendar_access_denied_on_permission_error(self, tmp_path):
        """Test CALENDAR returns denied on PermissionError (lines 219-220)."""
        calendar_dir = tmp_path / "Calendars"
        calendar_dir.mkdir()

        monitor = TCCPermissionMonitor(calendar_path=calendar_dir)

        with patch.object(Path, "iterdir", side_effect=PermissionError("Access denied")):
            status = monitor.check_permission(Permission.CALENDAR)

        assert status.granted is False
        assert "Calendar" in status.fix_instructions

    def test_calendar_access_denied_on_tcc_os_error(self, tmp_path):
        """Test CALENDAR returns denied on TCC OSError (lines 221-223)."""
        calendar_dir = tmp_path / "Calendars"
        calendar_dir.mkdir()

        monitor = TCCPermissionMonitor(calendar_path=calendar_dir)

        # Mock iterdir to raise OSError with TCC message
        with patch.object(Path, "iterdir", side_effect=OSError("Operation not permitted")):
            status = monitor.check_permission(Permission.CALENDAR)

        assert status.granted is False
        assert "Calendar" in status.fix_instructions

    def test_calendar_access_granted_on_other_os_error(self, tmp_path):
        """Test CALENDAR returns granted on non-TCC OSError (line 224)."""
        calendar_dir = tmp_path / "Calendars"
        calendar_dir.mkdir()

        monitor = TCCPermissionMonitor(calendar_path=calendar_dir)

        # Mock iterdir to raise OSError with a different message (not TCC)
        with patch.object(Path, "iterdir", side_effect=OSError("Network error")):
            status = monitor.check_permission(Permission.CALENDAR)

        # Non-TCC OSErrors return True (not a permission issue)
        assert status.granted is True


class TestAutomationAccessCheck:
    """Tests for Automation permission check."""

    def test_automation_access_always_granted(self):
        """Test AUTOMATION returns granted (can't probe without side effects)."""
        monitor = TCCPermissionMonitor()
        status = monitor.check_permission(Permission.AUTOMATION)

        # Automation can't be probed - always returns True
        assert status.granted is True


class TestCheckAll:
    """Tests for check_all method."""

    def test_check_all_returns_all_permissions(self, tmp_path):
        """Test check_all returns status for all permissions."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(
            chat_db_path=chat_db,
            contacts_path=tmp_path / "contacts",
            calendar_path=tmp_path / "calendars",
        )

        results = monitor.check_all()

        assert len(results) == len(Permission)
        permission_types = {s.permission for s in results}
        assert permission_types == set(Permission)

    def test_check_all_includes_timestamps(self, tmp_path):
        """Test check_all results have timestamps."""
        monitor = TCCPermissionMonitor(
            chat_db_path=tmp_path / "chat.db",
        )

        results = monitor.check_all()

        for status in results:
            assert status.last_checked is not None
            assert len(status.last_checked) > 0


class TestPermissionCaching:
    """Tests for permission caching."""

    def test_cached_result_returned_within_ttl(self, tmp_path):
        """Test that cached results are returned within TTL."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        monitor._cache_ttl_seconds = 60.0  # Long TTL

        # First call
        status1 = monitor.check_permission(Permission.FULL_DISK_ACCESS)
        timestamp1 = status1.last_checked

        # Second call should return cached result
        status2 = monitor.check_permission(Permission.FULL_DISK_ACCESS)
        timestamp2 = status2.last_checked

        assert timestamp1 == timestamp2

    def test_cache_expires_after_ttl(self, tmp_path):
        """Test that cache expires after TTL."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        monitor._cache_ttl_seconds = 0.01  # Very short TTL

        # First call
        status1 = monitor.check_permission(Permission.FULL_DISK_ACCESS)
        timestamp1 = status1.last_checked

        # Wait for cache to expire
        time.sleep(0.02)

        # Second call should get fresh result
        status2 = monitor.check_permission(Permission.FULL_DISK_ACCESS)
        timestamp2 = status2.last_checked

        assert timestamp1 != timestamp2

    def test_clear_cache(self, tmp_path):
        """Test clear_cache method."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        monitor._cache_ttl_seconds = 60.0  # Long TTL

        # Populate cache
        monitor.check_permission(Permission.FULL_DISK_ACCESS)
        assert len(monitor._cache) == 1

        # Clear cache
        monitor.clear_cache()
        assert len(monitor._cache) == 0


class TestWaitForPermission:
    """Tests for wait_for_permission method."""

    def test_wait_returns_true_when_permission_granted(self, tmp_path):
        """Test wait_for_permission returns True when already granted."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)

        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=1)

        assert result is True

    def test_wait_returns_false_on_timeout(self, tmp_path):
        """Test wait_for_permission returns False when timeout exceeded."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)

        # Mock to always return denied
        with patch.object(
            monitor,
            "_check_full_disk_access",
            return_value=False,
        ):
            start = time.time()
            result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=1)
            elapsed = time.time() - start

        assert result is False
        assert elapsed >= 1.0  # Should have waited for timeout

    def test_wait_polls_periodically(self, tmp_path):
        """Test wait_for_permission polls multiple times."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)

        call_count = {"value": 0}

        def mock_check():
            call_count["value"] += 1
            return call_count["value"] >= 3  # Grant on 3rd call

        with patch.object(monitor, "_check_full_disk_access", side_effect=mock_check):
            result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=10)

        assert result is True
        assert call_count["value"] == 3


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_permission_monitor_returns_singleton(self):
        """Test get_permission_monitor returns same instance."""
        reset_permission_monitor()

        m1 = get_permission_monitor()
        m2 = get_permission_monitor()

        assert m1 is m2

        reset_permission_monitor()

    def test_reset_creates_new_instance(self):
        """Test reset_permission_monitor creates new instance."""
        reset_permission_monitor()

        m1 = get_permission_monitor()
        reset_permission_monitor()
        m2 = get_permission_monitor()

        assert m1 is not m2

        reset_permission_monitor()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_permission_checks(self, tmp_path):
        """Test concurrent permission checks are thread-safe."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        results = []
        lock = threading.Lock()

        def check_permissions():
            for _ in range(10):
                status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
                with lock:
                    results.append(status)

        threads = [threading.Thread(target=check_permissions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        assert all(s.permission == Permission.FULL_DISK_ACCESS for s in results)

    def test_concurrent_check_all(self, tmp_path):
        """Test concurrent check_all calls are thread-safe."""
        monitor = TCCPermissionMonitor(
            chat_db_path=tmp_path / "chat.db",
            contacts_path=tmp_path / "contacts",
            calendar_path=tmp_path / "calendars",
        )

        results = []
        lock = threading.Lock()

        def check_all():
            for _ in range(5):
                all_status = monitor.check_all()
                with lock:
                    results.extend(all_status)

        threads = [threading.Thread(target=check_all) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 4 threads * 5 iterations * 4 permissions = 80 results
        assert len(results) == 80


class TestProtocolCompliance:
    """Verify TCCPermissionMonitor implements PermissionMonitor protocol."""

    def test_has_check_permission(self):
        """Monitor has check_permission method."""
        monitor = TCCPermissionMonitor()
        assert hasattr(monitor, "check_permission")
        assert callable(monitor.check_permission)

    def test_has_check_all(self):
        """Monitor has check_all method."""
        monitor = TCCPermissionMonitor()
        assert hasattr(monitor, "check_all")
        assert callable(monitor.check_all)

    def test_has_wait_for_permission(self):
        """Monitor has wait_for_permission method."""
        monitor = TCCPermissionMonitor()
        assert hasattr(monitor, "wait_for_permission")
        assert callable(monitor.wait_for_permission)

    def test_check_permission_returns_permission_status(self, tmp_path):
        """check_permission returns PermissionStatus."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        result = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert isinstance(result, PermissionStatus)

    def test_check_all_returns_list_of_permission_status(self, tmp_path):
        """check_all returns list of PermissionStatus."""
        monitor = TCCPermissionMonitor(chat_db_path=tmp_path / "chat.db")
        result = monitor.check_all()

        assert isinstance(result, list)
        assert all(isinstance(s, PermissionStatus) for s in result)

    def test_wait_for_permission_returns_bool(self, tmp_path):
        """wait_for_permission returns bool."""
        chat_db = tmp_path / "chat.db"
        chat_db.write_bytes(b"test")

        monitor = TCCPermissionMonitor(chat_db_path=chat_db)
        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=1)

        assert isinstance(result, bool)
