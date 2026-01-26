"""Graceful degradation controller implementation.

Implements the DegradationController protocol from contracts/health.py.
Provides automatic fallback behavior when features fail.

Workstream 6 implementation.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from contracts.health import DegradationPolicy, FeatureState

from .circuit import CircuitBreaker, CircuitBreakerConfig, CircuitState

logger = logging.getLogger(__name__)


class FeatureRegistration:
    """Internal registration for a degradable feature."""

    def __init__(
        self,
        policy: DegradationPolicy,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Initialize feature registration.

        Args:
            policy: Degradation policy for this feature
            circuit_breaker: Circuit breaker managing this feature
        """
        self.policy = policy
        self.circuit_breaker = circuit_breaker
        self.primary_callable: Callable[..., Any] | None = None


class GracefulDegradationController:
    """Thread-safe graceful degradation controller.

    Manages feature registration and automatic fallback execution
    based on circuit breaker states and health checks.

    Implements the DegradationController protocol.
    """

    def __init__(self) -> None:
        """Initialize the degradation controller."""
        self._features: dict[str, FeatureRegistration] = {}
        self._lock = threading.Lock()
        logger.info("GracefulDegradationController initialized")

    def register_feature(self, policy: DegradationPolicy) -> None:
        """Register a feature with its degradation policy.

        Args:
            policy: Degradation policy defining health checks and fallbacks
        """
        with self._lock:
            if policy.feature_name in self._features:
                logger.warning(
                    "Feature '%s' already registered, replacing policy",
                    policy.feature_name,
                )

            config = CircuitBreakerConfig(
                failure_threshold=policy.max_failures,
                recovery_timeout_seconds=60.0,  # Default recovery timeout
                half_open_max_calls=1,
            )

            circuit_breaker = CircuitBreaker(
                name=f"circuit_{policy.feature_name}",
                config=config,
            )

            self._features[policy.feature_name] = FeatureRegistration(
                policy=policy,
                circuit_breaker=circuit_breaker,
            )

            logger.info(
                "Registered feature '%s' with max_failures=%d",
                policy.feature_name,
                policy.max_failures,
            )

    def execute(
        self,
        feature_name: str,
        primary: Callable[..., Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute feature with automatic fallback on failure.

        The execution flow depends on the circuit breaker state:
        1. CLOSED (healthy): Try primary, fall through to degraded/fallback on failure
        2. OPEN (failed): Skip primary, use fallback directly
        3. HALF_OPEN (testing): Try primary for recovery check

        Args:
            feature_name: Name of the registered feature
            primary: Primary function to execute (optional, can be provided at call time)
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable

        Returns:
            Result from primary, degraded, or fallback behavior

        Raises:
            KeyError: If feature is not registered
        """
        with self._lock:
            if feature_name not in self._features:
                msg = f"Feature '{feature_name}' is not registered"
                raise KeyError(msg)
            registration = self._features[feature_name]

        policy = registration.policy
        circuit = registration.circuit_breaker

        # Determine which callable to use as primary
        callable_func = primary or registration.primary_callable

        # Get current state
        state = circuit.state

        # OPEN state - go directly to fallback
        if state == CircuitState.OPEN:
            logger.debug(
                "Feature '%s' circuit is OPEN, using fallback",
                feature_name,
            )
            return self._execute_fallback(policy, *args, **kwargs)

        # CLOSED or HALF_OPEN - try primary
        if callable_func is not None:
            try:
                result = callable_func(*args, **kwargs)
                circuit.record_success()
                return result
            except TypeError as e:
                # TypeError usually indicates a programming error (wrong arguments),
                # not a transient failure. Re-raise to surface the bug.
                error_msg = str(e)
                if "argument" in error_msg or "positional" in error_msg or "keyword" in error_msg:
                    logger.error(
                        "Feature '%s' callable signature mismatch: %s",
                        feature_name,
                        error_msg,
                    )
                    raise
                # Other TypeErrors might be from the callable's logic, treat as failure
                circuit.record_failure()
                logger.warning(
                    "Feature '%s' primary execution failed with TypeError: %s",
                    feature_name,
                    error_msg,
                )
                new_state = circuit.state
                if new_state == CircuitState.OPEN:
                    return self._execute_fallback(policy, *args, **kwargs)
                return self._execute_degraded(policy, *args, **kwargs)
            except Exception as e:
                circuit.record_failure()
                logger.warning(
                    "Feature '%s' primary execution failed: %s",
                    feature_name,
                    str(e),
                )

                # Check new state after failure
                new_state = circuit.state
                if new_state == CircuitState.OPEN:
                    return self._execute_fallback(policy, *args, **kwargs)
                return self._execute_degraded(policy, *args, **kwargs)

        # No primary callable, check health and return appropriate behavior
        if self._check_health(policy):
            # Healthy but no primary - return degraded behavior
            return self._execute_degraded(policy, *args, **kwargs)
        return self._execute_fallback(policy, *args, **kwargs)

    def _check_health(self, policy: DegradationPolicy) -> bool:
        """Run health check for a feature.

        Args:
            policy: Policy containing health check callable

        Returns:
            True if healthy, False otherwise
        """
        try:
            return policy.health_check()
        except Exception as e:
            logger.warning(
                "Health check for '%s' raised exception: %s",
                policy.feature_name,
                str(e),
            )
            return False

    def _execute_degraded(
        self,
        policy: DegradationPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute degraded behavior.

        Args:
            policy: Policy containing degraded behavior callable
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from degraded behavior
        """
        logger.info("Executing degraded behavior for '%s'", policy.feature_name)
        try:
            return policy.degraded_behavior(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "Degraded behavior for '%s' failed: %s, falling back",
                policy.feature_name,
                str(e),
            )
            return self._execute_fallback(policy, *args, **kwargs)

    def _execute_fallback(
        self,
        policy: DegradationPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute fallback behavior.

        Args:
            policy: Policy containing fallback behavior callable
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from fallback behavior
        """
        logger.info("Executing fallback behavior for '%s'", policy.feature_name)
        return policy.fallback_behavior(*args, **kwargs)

    def get_health(self) -> dict[str, FeatureState]:
        """Return health status of all registered features.

        Returns:
            Dictionary mapping feature names to their health states
        """
        with self._lock:
            health_status: dict[str, FeatureState] = {}

            for name, registration in self._features.items():
                state = registration.circuit_breaker.state

                if state == CircuitState.CLOSED:
                    health_status[name] = FeatureState.HEALTHY
                elif state == CircuitState.HALF_OPEN:
                    health_status[name] = FeatureState.DEGRADED
                else:  # OPEN
                    health_status[name] = FeatureState.FAILED

            return health_status

    def reset_feature(self, feature_name: str) -> None:
        """Reset failure count and try healthy mode again.

        Args:
            feature_name: Name of the feature to reset

        Raises:
            KeyError: If feature is not registered
        """
        with self._lock:
            if feature_name not in self._features:
                msg = f"Feature '{feature_name}' is not registered"
                raise KeyError(msg)

            registration = self._features[feature_name]
            registration.circuit_breaker.reset()

            logger.info("Feature '%s' has been reset", feature_name)

    def unregister_feature(self, feature_name: str) -> None:
        """Unregister a feature.

        Args:
            feature_name: Name of the feature to unregister

        Raises:
            KeyError: If feature is not registered
        """
        with self._lock:
            if feature_name not in self._features:
                msg = f"Feature '{feature_name}' is not registered"
                raise KeyError(msg)

            del self._features[feature_name]
            logger.info("Feature '%s' unregistered", feature_name)

    def get_feature_stats(self, feature_name: str) -> dict[str, Any]:
        """Get detailed statistics for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with circuit breaker statistics

        Raises:
            KeyError: If feature is not registered
        """
        with self._lock:
            if feature_name not in self._features:
                msg = f"Feature '{feature_name}' is not registered"
                raise KeyError(msg)

            registration = self._features[feature_name]
            stats = registration.circuit_breaker.stats
            state = registration.circuit_breaker.state

            return {
                "feature_name": feature_name,
                "state": state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "total_executions": stats.total_executions,
                "total_failures": stats.total_failures,
                "total_successes": stats.total_successes,
                "last_failure_time": stats.last_failure_time,
                "last_success_time": stats.last_success_time,
            }


# Module-level singleton
_controller: GracefulDegradationController | None = None
_controller_lock = threading.Lock()


def get_degradation_controller() -> GracefulDegradationController:
    """Get the singleton degradation controller instance.

    Returns:
        The shared GracefulDegradationController instance
    """
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = GracefulDegradationController()
    return _controller


def reset_degradation_controller() -> None:
    """Reset the singleton degradation controller.

    Useful for testing or reinitializing the system.
    """
    global _controller
    with _controller_lock:
        _controller = None
        logger.info("Degradation controller singleton reset")
