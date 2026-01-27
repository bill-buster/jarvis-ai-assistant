"""Rate limiting configuration for JARVIS API.

Uses slowapi for rate limiting with configurable limits per endpoint type.
Supports per-client tracking with proper 429 responses.

Usage:
    from api.ratelimit import limiter, get_remote_address

    @app.get("/endpoint")
    @limiter.limit("60/minute")
    async def endpoint(request: Request):
        ...
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address as _get_remote_address

from jarvis.config import get_config

if TYPE_CHECKING:
    pass  # Type imports only

logger = logging.getLogger(__name__)

# Type for decorator functions
F = TypeVar("F", bound=Callable[..., Any])


def get_remote_address(request: Request) -> str:
    """Get client identifier for rate limiting.

    For local-first apps, we use a combination of IP and user-agent
    to differentiate between different local clients.

    Args:
        request: The FastAPI request object.

    Returns:
        String identifier for the client.
    """
    # Use slowapi's default for IP
    ip = _get_remote_address(request) or "unknown"

    # For localhost, add user-agent to differentiate clients
    if ip in ("127.0.0.1", "localhost", "::1"):
        user_agent = request.headers.get("user-agent", "unknown")
        # Hash the user agent for privacy
        return f"{ip}:{hash(user_agent) % 10000}"

    return ip


def _rate_limit_enabled() -> bool:
    """Check if rate limiting is enabled in config."""
    try:
        config = get_config()
        return config.rate_limit.enabled
    except Exception:
        return True


def _get_rate_limit(limit_type: str) -> str:
    """Get rate limit string based on type and config.

    Args:
        limit_type: Either "generation" or "read"

    Returns:
        Rate limit string like "10/minute" or "60/minute"
    """
    try:
        config = get_config()
        requests_per_minute = config.rate_limit.requests_per_minute

        if limit_type == "generation":
            # Generation endpoints get 1/6 of the read rate
            return f"{max(1, requests_per_minute // 6)}/minute"
        else:
            return f"{requests_per_minute}/minute"
    except Exception:
        # Defaults
        if limit_type == "generation":
            return "10/minute"
        return "60/minute"


# Create the limiter instance
# Key function extracts client identifier from request
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    enabled=True,  # Can be toggled via config at runtime
)


# Rate limit constants for different endpoint types
RATE_LIMIT_GENERATION = "10/minute"  # For CPU-intensive generation endpoints
RATE_LIMIT_READ = "60/minute"  # For read-only endpoints
RATE_LIMIT_WRITE = "30/minute"  # For write endpoints


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Handle rate limit exceeded errors with proper 429 response.

    Args:
        request: The FastAPI request object.
        exc: The rate limit exception.

    Returns:
        JSON response with 429 status and retry-after header.
    """
    from fastapi.responses import JSONResponse

    # Extract retry-after from the exception if available
    retry_after = 60  # Default to 60 seconds
    if hasattr(exc, "detail") and exc.detail:
        # Try to parse the limit from the detail message
        try:
            # Detail format: "Rate limit exceeded: X per Y minute"
            parts = str(exc.detail).split()
            for i, part in enumerate(parts):
                if part.isdigit() and i + 1 < len(parts):
                    if "minute" in parts[i + 1]:
                        retry_after = 60
                    elif "second" in parts[i + 1]:
                        retry_after = int(part)
                    break
        except (ValueError, IndexError):
            pass

    logger.warning(
        "Rate limit exceeded for %s %s from %s",
        request.method,
        request.url.path,
        get_remote_address(request),
    )

    response_body = {
        "error": "RateLimitExceeded",
        "code": "RATE_LIMIT_EXCEEDED",
        "detail": "Too many requests. Please slow down.",
        "retry_after_seconds": retry_after,
    }

    return JSONResponse(
        status_code=429,
        content=response_body,
        headers={"Retry-After": str(retry_after)},
    )


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """Decorator to add timeout to async endpoints.

    Args:
        timeout_seconds: Maximum time in seconds before timeout.

    Returns:
        Decorator function.

    Usage:
        @with_timeout(30.0)
        async def my_endpoint():
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=408,
                    detail=f"Request timed out after {timeout_seconds} seconds",
                ) from None

        return wrapper  # type: ignore[return-value]

    return decorator


def run_in_threadpool(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Decorator to run a synchronous function in a thread pool.

    Use this for CPU-bound operations like model generation to avoid
    blocking the event loop.

    Args:
        func: Synchronous function to wrap.

    Returns:
        Async function that runs the original in a thread pool.

    Usage:
        @run_in_threadpool
        def cpu_intensive_work():
            ...

        async def endpoint():
            result = await cpu_intensive_work()
    """
    from starlette.concurrency import run_in_threadpool as starlette_threadpool

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await starlette_threadpool(func, *args, **kwargs)

    return wrapper


def get_timeout_generation() -> float:
    """Get generation timeout from config.

    Returns:
        Timeout in seconds for generation operations.
    """
    try:
        config = get_config()
        return config.rate_limit.generation_timeout_seconds
    except Exception:
        return 30.0  # Default fallback


def get_timeout_read() -> float:
    """Get read timeout from config.

    Returns:
        Timeout in seconds for read operations.
    """
    try:
        config = get_config()
        return config.rate_limit.read_timeout_seconds
    except Exception:
        return 10.0  # Default fallback


# Timeout constants for different operations (for backward compatibility)
# IMPORTANT: These constants are evaluated at module import time and will NOT
# reflect runtime config changes. Always use get_timeout_generation() and
# get_timeout_read() functions instead to get the current config values.
# These constants are provided only for backwards compatibility with existing code.
TIMEOUT_GENERATION = get_timeout_generation()
TIMEOUT_READ = get_timeout_read()


# Export all public symbols
__all__ = [
    "limiter",
    "get_remote_address",
    "rate_limit_exceeded_handler",
    "with_timeout",
    "run_in_threadpool",
    "get_timeout_generation",
    "get_timeout_read",
    "RATE_LIMIT_GENERATION",
    "RATE_LIMIT_READ",
    "RATE_LIMIT_WRITE",
    "TIMEOUT_GENERATION",
    "TIMEOUT_READ",
]
