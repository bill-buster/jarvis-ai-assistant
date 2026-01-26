"""Unit tests for retry utilities.

Tests the retry decorators and context managers.
"""

import time
from unittest.mock import MagicMock

import pytest

from jarvis.retry import RetryAttempt, RetryContext, retry_with_backoff


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_succeeds_first_try(self):
        """Returns immediately on success."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_exception(self):
        """Retries on specified exceptions."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Raises exception after max retries exhausted."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count == 3

    def test_does_not_catch_unspecified_exceptions(self):
        """Does not retry on unspecified exception types."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not caught")

        with pytest.raises(TypeError):
            raises_type_error()

        assert call_count == 1

    def test_exponential_backoff(self):
        """Delays increase exponentially."""
        times = []

        @retry_with_backoff(max_retries=4, base_delay=0.05, max_delay=1.0, exceptions=(ValueError,))
        def record_times():
            times.append(time.time())
            if len(times) < 4:
                raise ValueError("Retry")
            return "done"

        record_times()

        # Check that delays roughly follow exponential pattern
        # Delay 1: ~0.05s, Delay 2: ~0.1s, Delay 3: ~0.2s
        delay1 = times[1] - times[0]
        delay2 = times[2] - times[1]
        delay3 = times[3] - times[2]

        # Each delay should be roughly 2x the previous (with tolerance)
        assert delay2 > delay1 * 1.5  # Allow for timing variance
        assert delay3 > delay2 * 1.5

    def test_max_delay_respected(self):
        """Does not exceed max_delay."""
        times = []

        @retry_with_backoff(max_retries=5, base_delay=0.1, max_delay=0.15, exceptions=(ValueError,))
        def record_times():
            times.append(time.time())
            if len(times) < 5:
                raise ValueError("Retry")
            return "done"

        record_times()

        # All delays should be at most max_delay + tolerance
        for i in range(1, len(times)):
            delay = times[i] - times[i - 1]
            assert delay < 0.25  # max_delay + tolerance

    def test_on_retry_callback(self):
        """Calls on_retry callback on each retry."""
        callback = MagicMock()
        call_count = 0

        @retry_with_backoff(
            max_retries=3, base_delay=0.01, exceptions=(ValueError,), on_retry=callback
        )
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary")
            return "success"

        fails_twice()

        assert callback.call_count == 2
        # First retry: attempt=1, exception is a ValueError
        call_args = callback.call_args_list[0]
        assert call_args[0][0] == 1  # First argument is attempt number
        assert isinstance(call_args[0][1], ValueError)  # Second argument is exception

    def test_preserves_function_metadata(self):
        """Preserves function name and docstring."""

        @retry_with_backoff()
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_passes_args_and_kwargs(self):
        """Passes arguments to wrapped function."""

        @retry_with_backoff(max_retries=1)
        def add(a, b, multiplier=1):
            return (a + b) * multiplier

        assert add(2, 3) == 5
        assert add(2, 3, multiplier=2) == 10


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_succeeds_first_try(self):
        """Exits loop on first success."""
        attempts = 0
        retry = RetryContext(max_retries=3, exceptions=(ValueError,))

        for attempt in retry:
            with attempt:
                attempts += 1
                result = "success"

        assert attempts == 1
        assert result == "success"

    def test_retries_on_exception(self):
        """Retries on caught exceptions."""
        attempts = 0
        retry = RetryContext(max_retries=3, base_delay=0.01, exceptions=(ValueError,))

        for attempt in retry:
            with attempt:
                attempts += 1
                if attempts < 3:
                    raise ValueError("Temporary")
                result = "success"

        assert attempts == 3
        assert result == "success"

    def test_raises_after_exhaustion(self):
        """Raises last exception after retries exhausted."""
        attempts = 0
        retry = RetryContext(max_retries=3, base_delay=0.01, exceptions=(ValueError,))

        with pytest.raises(ValueError, match="Always fails"):
            for attempt in retry:
                with attempt:
                    attempts += 1
                    raise ValueError("Always fails")

        assert attempts == 3

    def test_does_not_catch_unspecified(self):
        """Does not suppress unspecified exceptions."""
        attempts = 0
        retry = RetryContext(max_retries=3, exceptions=(ValueError,))

        with pytest.raises(TypeError):
            for attempt in retry:
                with attempt:
                    attempts += 1
                    raise TypeError("Not caught")

        assert attempts == 1

    def test_attempt_number_tracking(self):
        """Tracks attempt number correctly."""
        attempt_numbers = []
        retry = RetryContext(max_retries=3, base_delay=0.01, exceptions=(ValueError,))

        for attempt in retry:
            with attempt:
                attempt_numbers.append(attempt.attempt_number)
                if len(attempt_numbers) < 3:
                    raise ValueError("Retry")

        assert attempt_numbers == [0, 1, 2]


class TestRetryAttempt:
    """Tests for RetryAttempt context manager."""

    def test_enter_returns_self(self):
        """__enter__ returns the attempt object."""
        context = RetryContext(max_retries=1)
        for attempt in context:
            entered = attempt.__enter__()
            assert entered is attempt
            break

    def test_exit_returns_true_for_caught_exceptions(self):
        """__exit__ returns True for caught exceptions to suppress."""
        context = RetryContext(max_retries=2, base_delay=0.01, exceptions=(ValueError,))

        # Manually test __exit__ behavior
        attempt = RetryAttempt(context, 0)
        attempt.__enter__()
        result = attempt.__exit__(ValueError, ValueError("test"), None)
        assert result is True  # Exception suppressed

    def test_exit_returns_false_for_uncaught_exceptions(self):
        """__exit__ returns False for uncaught exceptions."""
        context = RetryContext(max_retries=2, exceptions=(ValueError,))

        attempt = RetryAttempt(context, 0)
        attempt.__enter__()
        result = attempt.__exit__(TypeError, TypeError("test"), None)
        assert result is False  # Exception not suppressed

    def test_exit_returns_false_on_success(self):
        """__exit__ returns False when no exception occurred."""
        context = RetryContext(max_retries=2)

        attempt = RetryAttempt(context, 0)
        attempt.__enter__()
        result = attempt.__exit__(None, None, None)
        assert result is False


class TestRetryEdgeCases:
    """Edge case tests for retry utilities."""

    def test_zero_retries(self):
        """Handles max_retries=0 (no retries at all)."""
        # With 0 max_retries, should never execute

        @retry_with_backoff(max_retries=0, exceptions=(ValueError,))
        def never_runs():
            raise ValueError("Should not run")

        # This should raise RuntimeError because no exception was captured
        # but all retries failed (none attempted)
        with pytest.raises(RuntimeError, match="No exception captured"):
            never_runs()

    def test_single_retry(self):
        """Works with max_retries=1."""
        call_count = 0

        @retry_with_backoff(max_retries=1, exceptions=(ValueError,))
        def single_try():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fails")

        with pytest.raises(ValueError):
            single_try()

        assert call_count == 1

    def test_multiple_exception_types(self):
        """Catches multiple exception types."""
        call_count = 0
        exceptions = [ValueError, TypeError, RuntimeError]

        @retry_with_backoff(
            max_retries=4, base_delay=0.01, exceptions=(ValueError, TypeError, RuntimeError)
        )
        def raises_different():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise exceptions[call_count - 1]("Error")
            return "success"

        result = raises_different()
        assert result == "success"
        assert call_count == 4
