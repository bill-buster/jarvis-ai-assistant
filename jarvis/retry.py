"""Retry utilities with exponential backoff.

Provides decorators and utilities for retrying operations that may fail
transiently, such as network requests or model loading.
"""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for retrying functions with exponential backoff.

    Retries the decorated function on specified exceptions, with exponentially
    increasing delays between attempts.

    Args:
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Initial delay in seconds (default 1.0)
        max_delay: Maximum delay in seconds (default 10.0)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry with (attempt, exception)

    Returns:
        Decorated function with retry behavior

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            delay,
                            str(e),
                        )
                        if on_retry:
                            on_retry(attempt + 1, e)
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_retries,
                            func.__name__,
                            str(e),
                        )

            # This should never happen if exceptions is non-empty,
            # but mypy needs the explicit raise
            if last_exception is not None:
                raise last_exception
            msg = "No exception captured but all retries failed"
            raise RuntimeError(msg)

        return wrapper

    return decorator


def retry_async_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Async decorator for retrying functions with exponential backoff.

    Similar to retry_with_backoff but uses asyncio.sleep for async functions.

    Args:
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Initial delay in seconds (default 1.0)
        max_delay: Maximum delay in seconds (default 10.0)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry with (attempt, exception)

    Returns:
        Decorated async function with retry behavior
    """
    import asyncio

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc, no-any-return]
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            "Async retry %d/%d for %s after %.1fs: %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            delay,
                            str(e),
                        )
                        if on_retry:
                            on_retry(attempt + 1, e)
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All %d async retries exhausted for %s: %s",
                            max_retries,
                            func.__name__,
                            str(e),
                        )

            if last_exception is not None:
                raise last_exception
            msg = "No exception captured but all retries failed"
            raise RuntimeError(msg)

        return wrapper  # type: ignore[return-value]

    return decorator


class RetryContext:
    """Context manager for retry logic without decorators.

    Useful when you need more control over the retry loop or when
    decorators aren't suitable.

    Example:
        retry = RetryContext(max_retries=3, exceptions=(IOError,))
        for attempt in retry:
            with attempt:
                result = risky_operation()
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        """Initialize RetryContext.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exceptions: Tuple of exception types to catch and retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self._attempt = 0
        self._last_exception: Exception | None = None

    def __iter__(self) -> "RetryContext":
        """Start iteration over retry attempts."""
        self._attempt = 0
        self._last_exception = None
        return self

    def __next__(self) -> "RetryAttempt":
        """Get the next retry attempt."""
        if self._attempt >= self.max_retries:
            if self._last_exception is not None:
                raise self._last_exception
            raise StopIteration

        attempt = RetryAttempt(self, self._attempt)
        self._attempt += 1
        return attempt

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed attempt."""
        self._last_exception = exception
        if self._attempt < self.max_retries:
            delay = min(self.base_delay * (2 ** (self._attempt - 1)), self.max_delay)
            logger.warning(
                "Attempt %d/%d failed, retrying after %.1fs: %s",
                self._attempt,
                self.max_retries,
                delay,
                str(exception),
            )
            time.sleep(delay)

    def _record_success(self) -> None:
        """Record a successful attempt and stop iteration."""
        self._attempt = self.max_retries  # Stop iteration
        self._last_exception = None  # Clear any previous exceptions


class RetryAttempt:
    """Context manager for a single retry attempt."""

    def __init__(self, context: RetryContext, attempt_number: int):
        """Initialize RetryAttempt.

        Args:
            context: The parent RetryContext
            attempt_number: Zero-based attempt number
        """
        self._context = context
        self.attempt_number = attempt_number
        self._succeeded = False

    def __enter__(self) -> "RetryAttempt":
        """Enter the attempt context."""
        return self

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> bool:
        """Exit the attempt context, handling exceptions."""
        if exc_type is None:
            self._succeeded = True
            self._context._record_success()
            return False

        if exc_val is not None and isinstance(exc_val, self._context.exceptions):
            self._context._record_failure(exc_val)
            return True  # Suppress the exception for retry

        return False  # Don't suppress other exceptions
