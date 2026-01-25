"""macOS TCC permission monitoring implementation.

Implements the PermissionMonitor protocol from contracts/health.py.
Provides checking for Full Disk Access, Contacts, Calendar, and Automation.

Workstream 7 implementation.
"""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from contracts.health import Permission, PermissionStatus

logger = logging.getLogger(__name__)

# Default paths for permission checks
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
CONTACTS_DB_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "Sources"
CALENDAR_DB_PATH = Path.home() / "Library" / "Calendars"

# Fix instructions for each permission type
FIX_INSTRUCTIONS = {
    Permission.FULL_DISK_ACCESS: (
        "Grant Full Disk Access to your terminal or IDE:\n"
        "1. Open System Settings → Privacy & Security → Full Disk Access\n"
        "2. Click the lock to make changes\n"
        "3. Click '+' and add your terminal app (Terminal, iTerm2, VS Code, etc.)\n"
        "4. Restart the application after granting access"
    ),
    Permission.CONTACTS: (
        "Grant Contacts access to your terminal or IDE:\n"
        "1. Open System Settings → Privacy & Security → Contacts\n"
        "2. Click the lock to make changes\n"
        "3. Enable access for your terminal app\n"
        "4. Restart the application after granting access"
    ),
    Permission.CALENDAR: (
        "Grant Calendar access to your terminal or IDE:\n"
        "1. Open System Settings → Privacy & Security → Calendar\n"
        "2. Click the lock to make changes\n"
        "3. Enable access for your terminal app\n"
        "4. Restart the application after granting access"
    ),
    Permission.AUTOMATION: (
        "Grant Automation access for controlling other apps:\n"
        "1. Open System Settings → Privacy & Security → Automation\n"
        "2. Find your terminal app in the list\n"
        "3. Enable access for the apps you want to control\n"
        "4. If not listed, the app will request access when first needed"
    ),
}


def _get_iso_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


class TCCPermissionMonitor:
    """Thread-safe TCC permission monitor.

    Checks macOS Transparency, Consent, and Control (TCC) permissions
    by testing access to protected resources.

    Implements the PermissionMonitor protocol.
    """

    def __init__(
        self,
        chat_db_path: Path | None = None,
        contacts_path: Path | None = None,
        calendar_path: Path | None = None,
    ) -> None:
        """Initialize the permission monitor.

        Args:
            chat_db_path: Custom path to chat.db for testing Full Disk Access
            contacts_path: Custom path for Contacts permission check
            calendar_path: Custom path for Calendar permission check
        """
        self._chat_db_path = chat_db_path or CHAT_DB_PATH
        self._contacts_path = contacts_path or CONTACTS_DB_PATH
        self._calendar_path = calendar_path or CALENDAR_DB_PATH
        self._lock = threading.Lock()
        self._cache: dict[Permission, PermissionStatus] = {}
        self._cache_ttl_seconds = 5.0  # Cache validity period
        self._cache_timestamps: dict[Permission, float] = {}
        logger.info("TCCPermissionMonitor initialized")

    def check_permission(self, permission: Permission) -> PermissionStatus:
        """Check if a specific permission is granted.

        Uses cached result if available and not expired.

        Args:
            permission: The permission to check

        Returns:
            PermissionStatus with granted status and fix instructions
        """
        with self._lock:
            # Check cache validity
            if permission in self._cache:
                cached_time = self._cache_timestamps.get(permission, 0)
                if time.time() - cached_time < self._cache_ttl_seconds:
                    return self._cache[permission]

            # Perform fresh check
            granted = self._check_permission_impl(permission)
            status = PermissionStatus(
                permission=permission,
                granted=granted,
                last_checked=_get_iso_timestamp(),
                fix_instructions="" if granted else FIX_INSTRUCTIONS[permission],
            )

            # Update cache
            self._cache[permission] = status
            self._cache_timestamps[permission] = time.time()

            if not granted:
                logger.warning(
                    "Permission %s not granted",
                    permission.value,
                )

            return status

    def _check_permission_impl(self, permission: Permission) -> bool:
        """Perform the actual permission check.

        Args:
            permission: The permission to check

        Returns:
            True if permission is granted, False otherwise
        """
        if permission == Permission.FULL_DISK_ACCESS:
            return self._check_full_disk_access()
        elif permission == Permission.CONTACTS:
            return self._check_contacts_access()
        elif permission == Permission.CALENDAR:
            return self._check_calendar_access()
        elif permission == Permission.AUTOMATION:
            return self._check_automation_access()
        else:
            logger.warning("Unknown permission type: %s", permission)
            return False

    def _check_full_disk_access(self) -> bool:
        """Check Full Disk Access by testing chat.db readability.

        Returns:
            True if chat.db is readable
        """
        try:
            # First check if file exists
            if not self._chat_db_path.exists():
                logger.debug("chat.db not found at %s", self._chat_db_path)
                # File doesn't exist - not a permission issue
                # Return True since we can't test without the file
                return True

            # Try to open and read the file
            with open(self._chat_db_path, "rb") as f:
                # Read first few bytes to verify actual access
                f.read(16)
            return True
        except PermissionError:
            logger.debug("Permission denied reading chat.db")
            return False
        except OSError as e:
            # Handle "Operation not permitted" which is TCC denial
            if "Operation not permitted" in str(e):
                logger.debug("TCC denied access to chat.db")
                return False
            # Other OS errors might not be permission related
            logger.debug("OS error checking chat.db: %s", e)
            return False

    def _check_contacts_access(self) -> bool:
        """Check Contacts access by testing AddressBook directory.

        Returns:
            True if Contacts directory is accessible
        """
        try:
            if not self._contacts_path.exists():
                # Directory doesn't exist - not a permission issue
                return True

            # Try to list directory contents
            list(self._contacts_path.iterdir())
            return True
        except PermissionError:
            return False
        except OSError as e:
            if "Operation not permitted" in str(e):
                return False
            return True

    def _check_calendar_access(self) -> bool:
        """Check Calendar access by testing Calendars directory.

        Returns:
            True if Calendar directory is accessible
        """
        try:
            if not self._calendar_path.exists():
                # Directory doesn't exist - not a permission issue
                return True

            # Try to list directory contents
            list(self._calendar_path.iterdir())
            return True
        except PermissionError:
            return False
        except OSError as e:
            if "Operation not permitted" in str(e):
                return False
            return True

    def _check_automation_access(self) -> bool:
        """Check Automation access.

        Automation permissions can't be probed directly - they're requested
        when an AppleScript/osascript command targets another application.
        We return True optimistically since we can't test without side effects.

        Returns:
            True (always - can only be tested through actual automation attempts)
        """
        # Automation permissions require actually trying to control an app
        # We can't probe this without side effects, so we assume granted
        # The actual check happens when automation is attempted
        return True

    def check_all(self) -> list[PermissionStatus]:
        """Check all required permissions.

        Returns:
            List of PermissionStatus for all Permission types
        """
        results = []
        for permission in Permission:
            status = self.check_permission(permission)
            results.append(status)
        return results

    def wait_for_permission(
        self,
        permission: Permission,
        timeout_seconds: int,
    ) -> bool:
        """Block until permission is granted or timeout.

        Polls the permission status periodically until granted or timeout.

        Args:
            permission: The permission to wait for
            timeout_seconds: Maximum seconds to wait

        Returns:
            True if permission was granted within timeout, False otherwise
        """
        poll_interval = 1.0  # Check every second
        elapsed = 0.0

        logger.info(
            "Waiting for %s permission (timeout: %ds)",
            permission.value,
            timeout_seconds,
        )

        while elapsed < timeout_seconds:
            # Clear cache to force fresh check
            with self._lock:
                self._cache.pop(permission, None)
                self._cache_timestamps.pop(permission, None)

            status = self.check_permission(permission)
            if status.granted:
                logger.info("Permission %s granted", permission.value)
                return True

            time.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(
            "Timeout waiting for %s permission after %ds",
            permission.value,
            timeout_seconds,
        )
        return False

    def clear_cache(self) -> None:
        """Clear the permission status cache.

        Forces fresh checks on next permission query.
        """
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()


# Module-level singleton
_monitor: TCCPermissionMonitor | None = None
_monitor_lock = threading.Lock()


def get_permission_monitor() -> TCCPermissionMonitor:
    """Get the singleton permission monitor instance.

    Returns:
        The shared TCCPermissionMonitor instance
    """
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = TCCPermissionMonitor()
    return _monitor


def reset_permission_monitor() -> None:
    """Reset the singleton permission monitor.

    Useful for testing or reinitializing the system.
    """
    global _monitor
    with _monitor_lock:
        _monitor = None
        logger.info("Permission monitor singleton reset")
