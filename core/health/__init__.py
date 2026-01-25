"""System health monitoring (Workstreams 6-7).

Provides graceful degradation, circuit breaker, permission monitoring,
and schema detection functionality.
"""

from .circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitOpenError,
    CircuitState,
)
from .degradation import (
    GracefulDegradationController,
    get_degradation_controller,
    reset_degradation_controller,
)
from .permissions import (
    TCCPermissionMonitor,
    get_permission_monitor,
    reset_permission_monitor,
)
from .schema import (
    ChatDBSchemaDetector,
    get_schema_detector,
    reset_schema_detector,
)

__all__ = [
    # Circuit breaker (WS6)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitOpenError",
    "CircuitState",
    # Degradation controller (WS6)
    "GracefulDegradationController",
    "get_degradation_controller",
    "reset_degradation_controller",
    # Permission monitor (WS7)
    "TCCPermissionMonitor",
    "get_permission_monitor",
    "reset_permission_monitor",
    # Schema detector (WS7)
    "ChatDBSchemaDetector",
    "get_schema_detector",
    "reset_schema_detector",
]
