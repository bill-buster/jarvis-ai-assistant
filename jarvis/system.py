"""JARVIS System Initialization - Shared initialization logic for CLI and API.

This module contains shared system initialization code used by both
the CLI (jarvis/cli.py) and API (jarvis/api.py) components.
"""

import logging
from typing import Any

from contracts.health import DegradationPolicy
from core.health import get_degradation_controller
from core.memory import get_memory_controller

logger = logging.getLogger(__name__)

# Feature names for degradation controller
FEATURE_CHAT = "chat"
FEATURE_IMESSAGE = "imessage"

# Constants for generation and display
DEFAULT_MAX_TOKENS = 200
MESSAGE_PREVIEW_LENGTH = 80


def _handle_memory_pressure(level: str) -> None:
    """Handle memory pressure by unloading non-essential models.

    Args:
        level: Memory pressure level (e.g., "green", "yellow", "red", "critical").
    """
    if level in ("red", "critical"):
        logger.info("Memory pressure detected (%s), unloading sentence model", level)
        try:
            from models.templates import unload_sentence_model

            unload_sentence_model()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Error unloading sentence model: %s", e)


def _check_imessage_access() -> bool:
    """Check if iMessage database is accessible via the integration layer.

    This is a high-level check that verifies the entire integration stack works,
    not just permission status. For permission-only checks during setup,
    use core.health.get_permission_monitor() instead.

    Returns:
        True if accessible, False otherwise.
    """
    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            return reader.check_access()
    except PermissionError:
        logger.debug("Permission denied accessing iMessage database")
        return False
    except FileNotFoundError:
        logger.debug("iMessage database not found")
        return False
    except ImportError:
        logger.debug("iMessage integration module not available")
        return False
    except Exception as e:
        logger.debug("Error checking iMessage access: %s", e)
        return False


def _template_only_response(prompt: str) -> str:
    """Generate response using only template matching.

    Args:
        prompt: User prompt.

    Returns:
        Template response or degraded message.
    """
    try:
        from models.templates import TemplateMatcher

        matcher = TemplateMatcher()
        match = matcher.match(prompt)
        if match:
            return match.template.response
    except ImportError:
        logger.debug("Template matching module not available")
    except Exception as e:
        logger.debug("Template matching failed: %s", e)
    return "I'm operating in limited mode. Please try a simpler query."


def _fallback_response() -> str:
    """Return a fallback response when chat is unavailable.

    Returns:
        Static fallback message.
    """
    return (
        "I'm currently unable to process your request. "
        "Please check system health with 'jarvis health'."
    )


def _imessage_degraded(query: str) -> list[Any]:
    """Return degraded iMessage search result.

    Args:
        query: Search query.

    Returns:
        Empty list with logged warning.
    """
    logger.warning("iMessage search running in degraded mode")
    return []


def _imessage_fallback() -> list[Any]:
    """Return fallback for iMessage when unavailable.

    Returns:
        Empty list.
    """
    return []


def initialize_system() -> tuple[bool, list[str]]:
    """Initialize JARVIS system components.

    Initializes the memory controller and degradation controller,
    registers features, and returns status.

    Returns:
        Tuple of (success, list of warnings)
    """
    warnings: list[str] = []

    # Initialize memory controller
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()

    logger.info(
        "Memory mode: %s (%.0f MB available)",
        state.current_mode.value,
        state.available_mb,
    )

    # Register memory pressure callback to unload sentence model when under pressure
    mem_controller.register_pressure_callback(_handle_memory_pressure)

    # Initialize degradation controller and register features
    deg_controller = get_degradation_controller()

    # Register chat feature
    deg_controller.register_feature(
        DegradationPolicy(
            feature_name=FEATURE_CHAT,
            health_check=lambda: True,  # Always healthy for now
            degraded_behavior=lambda prompt: _template_only_response(prompt),
            fallback_behavior=lambda prompt: _fallback_response(),
            recovery_check=lambda: True,
            max_failures=3,
        )
    )

    # Register iMessage feature
    deg_controller.register_feature(
        DegradationPolicy(
            feature_name=FEATURE_IMESSAGE,
            health_check=_check_imessage_access,
            degraded_behavior=lambda query: _imessage_degraded(query),
            fallback_behavior=lambda query: _imessage_fallback(),
            recovery_check=_check_imessage_access,
            max_failures=3,
        )
    )

    # Check for permission issues
    if not _check_imessage_access():
        warnings.append(
            "iMessage access unavailable. Grant Full Disk Access in "
            "System Settings > Privacy & Security > Full Disk Access."
        )

    return True, warnings
