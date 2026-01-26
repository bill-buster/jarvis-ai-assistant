"""Circuit breaker state machine for graceful degradation.

Implements the circuit breaker pattern with three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing, requests are blocked
- HALF_OPEN: Testing recovery, limited requests allowed

Workstream 6 implementation.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Healthy - requests pass through
    OPEN = "open"  # Failing - requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 3  # Failures before opening
    recovery_timeout_seconds: float = 60.0  # Time before trying recovery
    half_open_max_calls: int = 1  # Calls allowed in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changed_at: float = field(default_factory=time.monotonic)
    total_executions: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """Thread-safe circuit breaker implementation.

    State transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: When recovery_timeout has elapsed
    - HALF_OPEN -> CLOSED: On successful call
    - HALF_OPEN -> OPEN: On failed call
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            config: Configuration settings. Uses defaults if not provided.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.Lock()
        self._half_open_calls = 0

        logger.info(
            "Circuit breaker '%s' initialized: threshold=%d, timeout=%.1fs",
            self.name,
            self.config.failure_threshold,
            self.config.recovery_timeout_seconds,
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                failure_count=self._stats.failure_count,
                success_count=self._stats.success_count,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state_changed_at=self._stats.state_changed_at,
                total_executions=self._stats.total_executions,
                total_failures=self._stats.total_failures,
                total_successes=self._stats.total_successes,
            )

    def _check_state_transition(self) -> None:
        """Check if state should transition (called under lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._stats.state_changed_at
            if elapsed >= self.config.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state (called under lock)."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._stats.state_changed_at = time.monotonic()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        if new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.success_count = 0

        logger.info(
            "Circuit breaker '%s' transitioned: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )

    def can_execute(self) -> bool:
        """Check if a request can be executed.

        Returns:
            True if request should proceed, False if circuit is open.

        Note:
            This method only checks state; it does not reserve an execution slot.
            For atomic check-and-reserve, use _try_acquire_execution_slot() internally.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # HALF_OPEN - allow limited calls
            if self._half_open_calls < self.config.half_open_max_calls:
                return True

            return False

    def _try_acquire_execution_slot(self) -> tuple[bool, bool]:
        """Atomically check if execution is allowed and reserve a slot if in HALF_OPEN.

        This method prevents race conditions in the execute() method by combining
        the can_execute check with slot reservation in a single atomic operation.

        Returns:
            A tuple of (can_execute, slot_acquired):
            - can_execute: True if execution should proceed, False if blocked
            - slot_acquired: True if a HALF_OPEN slot was reserved (caller must
              pass _slot_already_acquired=True to record_success/failure)
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return (True, False)

            if self._state == CircuitState.OPEN:
                return (False, False)

            # HALF_OPEN - reserve slot atomically if available
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return (True, True)

            return (False, False)

    def record_success(self, *, _slot_already_acquired: bool = False) -> None:
        """Record a successful execution.

        Args:
            _slot_already_acquired: Internal flag, True if slot was reserved via
                _try_acquire_execution_slot(). External callers should not use this.
        """
        with self._lock:
            self._check_state_transition()
            self._stats.success_count += 1
            self._stats.total_successes += 1
            self._stats.total_executions += 1
            self._stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                if not _slot_already_acquired:
                    self._half_open_calls += 1
                # Recovery successful - close the circuit
                self._transition_to(CircuitState.CLOSED)
                logger.info(
                    "Circuit breaker '%s' recovered after successful call",
                    self.name,
                )

    def record_failure(self, *, _slot_already_acquired: bool = False) -> None:
        """Record a failed execution.

        Args:
            _slot_already_acquired: Internal flag, True if slot was reserved via
                _try_acquire_execution_slot(). External callers should not use this.
        """
        with self._lock:
            self._check_state_transition()
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.total_executions += 1
            self._stats.last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                if not _slot_already_acquired:
                    self._half_open_calls += 1
                # Recovery failed - reopen the circuit
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker '%s' recovery failed, reopening",
                    self.name,
                )
            elif self._state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "Circuit breaker '%s' opened after %d failures",
                        self.name,
                        self._stats.failure_count,
                    )

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._stats = CircuitBreakerStats()
            self._half_open_calls = 0

            logger.info(
                "Circuit breaker '%s' reset from %s to CLOSED",
                self.name,
                old_state.value,
            )

    def execute(
        self,
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Execute a function with circuit breaker protection.

        This method is thread-safe. It atomically checks if execution is allowed
        and reserves a slot (if in HALF_OPEN state) before executing the function.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func if successful

        Raises:
            CircuitOpenError: If circuit is open and call is blocked
            Exception: If func raises and circuit allows the call
        """
        can_exec, slot_acquired = self._try_acquire_execution_slot()
        if not can_exec:
            msg = f"Circuit breaker '{self.name}' is open"
            raise CircuitOpenError(msg)

        try:
            result = func(*args, **kwargs)
            self.record_success(_slot_already_acquired=slot_acquired)
            return result
        except Exception:
            self.record_failure(_slot_already_acquired=slot_acquired)
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and blocking requests."""

    pass
