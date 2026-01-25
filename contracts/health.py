"""System health monitoring interfaces.

Workstreams 6 and 7 implement against these contracts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class FeatureState(Enum):
    """Health state of a feature."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class Permission(Enum):
    """macOS permissions required by JARVIS."""

    FULL_DISK_ACCESS = "full_disk_access"
    CONTACTS = "contacts"
    CALENDAR = "calendar"
    AUTOMATION = "automation"


@dataclass
class PermissionStatus:
    """Status of a single permission."""

    permission: Permission
    granted: bool
    last_checked: str
    fix_instructions: str  # User-friendly instructions if not granted


@dataclass
class SchemaInfo:
    """Information about iMessage chat.db schema."""

    version: str
    tables: list[str]
    compatible: bool
    migration_needed: bool
    known_schema: bool  # False if we don't recognize this version


@dataclass
class DegradationPolicy:
    """Policy for degrading a feature gracefully."""

    feature_name: str
    health_check: Callable[[], bool]
    degraded_behavior: Callable[..., Any]
    fallback_behavior: Callable[..., Any]
    recovery_check: Callable[[], bool]
    max_failures: int = 3


class DegradationController(Protocol):
    """Interface for graceful degradation (Workstream 6)."""

    def register_feature(self, policy: DegradationPolicy) -> None:
        """Register a feature with its degradation policy."""
        ...

    def execute(self, feature_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute feature with automatic fallback on failure."""
        ...

    def get_health(self) -> dict[str, FeatureState]:
        """Return health status of all features."""
        ...

    def reset_feature(self, feature_name: str) -> None:
        """Reset failure count and try healthy mode again."""
        ...


class PermissionMonitor(Protocol):
    """Interface for TCC permission monitoring (Workstream 7)."""

    def check_permission(self, permission: Permission) -> PermissionStatus:
        """Check if a specific permission is granted."""
        ...

    def check_all(self) -> list[PermissionStatus]:
        """Check all required permissions."""
        ...

    def wait_for_permission(self, permission: Permission, timeout_seconds: int) -> bool:
        """Block until permission granted or timeout."""
        ...


class SchemaDetector(Protocol):
    """Interface for chat.db schema detection (Workstream 7)."""

    def detect(self, db_path: str) -> SchemaInfo:
        """Detect schema version and compatibility."""
        ...

    def get_query(self, query_name: str, schema_version: str) -> str:
        """Get appropriate SQL query for the detected schema."""
        ...
