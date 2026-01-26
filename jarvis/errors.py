"""Unified exception hierarchy for JARVIS.

This module provides a consistent error handling system across CLI, API, and models.
All JARVIS-specific exceptions inherit from JarvisError, enabling consistent handling.

Exception Hierarchy:
    JarvisError (base)
    ├── ConfigurationError - Configuration and settings issues
    ├── ModelError - Model loading and generation failures
    │   ├── ModelLoadError - Failed to load model
    │   └── ModelGenerationError - Generation failed
    ├── iMessageError - iMessage access and query issues
    │   ├── iMessageAccessError - Permission/access denied
    │   └── iMessageQueryError - Database query failure
    ├── ValidationError - Input validation failures
    └── ResourceError - System resource issues
        ├── MemoryError - Insufficient memory
        └── DiskError - Disk space/access issues

Usage:
    from jarvis.errors import ModelError, iMessageError

    try:
        result = model.generate(prompt)
    except ModelError as e:
        logger.error("Model error: %s (code: %s)", e.message, e.code)
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standard error codes for JARVIS errors.

    These codes can be used to programmatically identify error types
    and are included in API error responses.
    """

    # Configuration errors (CFG_*)
    CFG_INVALID = "CFG_INVALID"
    CFG_MISSING = "CFG_MISSING"
    CFG_MIGRATION_FAILED = "CFG_MIGRATION_FAILED"

    # Model errors (MDL_*)
    MDL_LOAD_FAILED = "MDL_LOAD_FAILED"
    MDL_NOT_FOUND = "MDL_NOT_FOUND"
    MDL_GENERATION_FAILED = "MDL_GENERATION_FAILED"
    MDL_TIMEOUT = "MDL_TIMEOUT"
    MDL_INVALID_REQUEST = "MDL_INVALID_REQUEST"

    # iMessage errors (MSG_*)
    MSG_ACCESS_DENIED = "MSG_ACCESS_DENIED"
    MSG_DB_NOT_FOUND = "MSG_DB_NOT_FOUND"
    MSG_QUERY_FAILED = "MSG_QUERY_FAILED"
    MSG_SCHEMA_UNSUPPORTED = "MSG_SCHEMA_UNSUPPORTED"
    MSG_SEND_FAILED = "MSG_SEND_FAILED"

    # Validation errors (VAL_*)
    VAL_INVALID_INPUT = "VAL_INVALID_INPUT"
    VAL_MISSING_REQUIRED = "VAL_MISSING_REQUIRED"
    VAL_TYPE_ERROR = "VAL_TYPE_ERROR"

    # Resource errors (RES_*)
    RES_MEMORY_LOW = "RES_MEMORY_LOW"
    RES_MEMORY_EXHAUSTED = "RES_MEMORY_EXHAUSTED"
    RES_DISK_FULL = "RES_DISK_FULL"
    RES_DISK_ACCESS = "RES_DISK_ACCESS"

    # Experiment errors (EXP_*)
    EXP_NOT_FOUND = "EXP_NOT_FOUND"
    EXP_INVALID_CONFIG = "EXP_INVALID_CONFIG"
    EXP_VARIANT_NOT_FOUND = "EXP_VARIANT_NOT_FOUND"
    EXP_ALREADY_EXISTS = "EXP_ALREADY_EXISTS"

    # Generic errors
    UNKNOWN = "UNKNOWN"


class JarvisError(Exception):
    """Base exception for all JARVIS errors.

    All JARVIS-specific exceptions inherit from this class, enabling
    consistent error handling patterns across the codebase.

    Attributes:
        message: Human-readable error message.
        code: Machine-readable error code from ErrorCode enum.
        details: Optional additional context about the error.
        cause: Optional original exception that caused this error.
    """

    default_message: str = "An error occurred"
    default_code: ErrorCode = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a JARVIS error.

        Args:
            message: Human-readable error message.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.details = details or {}
        self.cause = cause

        # Build full message for Exception base class
        super().__init__(self.message)

        # Chain the cause if provided
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return human-readable representation."""
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        parts = [f"{self.__class__.__name__}({self.message!r}"]
        if self.code != self.default_code:
            parts.append(f", code={self.code.value!r}")
        if self.details:
            parts.append(f", details={self.details!r}")
        if self.cause:
            parts.append(f", cause={self.cause!r}")
        parts.append(")")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for API responses.

        Returns:
            Dictionary with error, code, and detail fields.
        """
        result: dict[str, Any] = {
            "error": self.__class__.__name__,
            "code": self.code.value,
            "detail": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# Configuration Errors


class ConfigurationError(JarvisError):
    """Raised for configuration and settings issues.

    Examples:
        - Invalid configuration values
        - Missing required configuration
        - Configuration file corruption
        - Migration failures
    """

    default_message = "Configuration error"
    default_code = ErrorCode.CFG_INVALID

    def __init__(
        self,
        message: str | None = None,
        *,
        config_key: str | None = None,
        config_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a configuration error.

        Args:
            message: Human-readable error message.
            config_key: The configuration key that caused the error.
            config_path: Path to the configuration file.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        if config_path:
            details["config_path"] = config_path
        super().__init__(message, code=code, details=details, cause=cause)


# Model Errors


class ModelError(JarvisError):
    """Base class for model-related errors.

    Raised when model loading or generation fails.
    """

    default_message = "Model error"
    default_code = ErrorCode.MDL_GENERATION_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        model_name: str | None = None,
        model_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a model error.

        Args:
            message: Human-readable error message.
            model_name: Name of the model that caused the error.
            model_path: Path to the model files.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, code=code, details=details, cause=cause)


class ModelLoadError(ModelError):
    """Raised when model loading fails.

    Examples:
        - Model files not found
        - Corrupted model weights
        - Insufficient memory for model
        - Incompatible model format
    """

    default_message = "Failed to load model"
    default_code = ErrorCode.MDL_LOAD_FAILED


class ModelGenerationError(ModelError):
    """Raised when text generation fails.

    Examples:
        - Generation timeout
        - Invalid generation parameters
        - Internal model errors during generation
    """

    default_message = "Text generation failed"
    default_code = ErrorCode.MDL_GENERATION_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        prompt: str | None = None,
        timeout_seconds: float | None = None,
        model_name: str | None = None,
        model_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a generation error.

        Args:
            message: Human-readable error message.
            prompt: The prompt that failed to generate (may be truncated).
            timeout_seconds: Timeout value if this was a timeout error.
            model_name: Name of the model that caused the error.
            model_path: Path to the model files.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
            code = code or ErrorCode.MDL_TIMEOUT
        if prompt is not None:
            # Truncate long prompts in error details
            details["prompt_preview"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
        super().__init__(
            message,
            model_name=model_name,
            model_path=model_path,
            code=code,
            details=details,
            cause=cause,
        )


# iMessage Errors


class iMessageError(JarvisError):  # noqa: N801 - iMessage is a brand name
    """Base class for iMessage-related errors.

    Raised when iMessage access or operations fail.
    """

    default_message = "iMessage error"
    default_code = ErrorCode.MSG_QUERY_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        db_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize an iMessage error.

        Args:
            message: Human-readable error message.
            db_path: Path to the chat.db file.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if db_path:
            details["db_path"] = db_path
        super().__init__(message, code=code, details=details, cause=cause)


class iMessageAccessError(iMessageError):  # noqa: N801 - iMessage is a brand name
    """Raised when iMessage database access is denied.

    Examples:
        - Full Disk Access not granted
        - Database file not found
        - Database locked by another process
    """

    default_message = "Cannot access iMessage database"
    default_code = ErrorCode.MSG_ACCESS_DENIED

    def __init__(
        self,
        message: str | None = None,
        *,
        db_path: str | None = None,
        requires_permission: bool = False,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize an iMessage access error.

        Args:
            message: Human-readable error message.
            db_path: Path to the chat.db file.
            requires_permission: Whether Full Disk Access is needed.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if requires_permission:
            details["requires_permission"] = True
            details["permission_instructions"] = [
                "Open System Settings",
                "Go to Privacy & Security > Full Disk Access",
                "Add and enable your terminal application",
                "Restart JARVIS",
            ]
        super().__init__(message, db_path=db_path, code=code, details=details, cause=cause)


class iMessageQueryError(iMessageError):  # noqa: N801 - iMessage is a brand name
    """Raised when iMessage database queries fail.

    Examples:
        - SQL syntax errors
        - Schema version mismatch
        - Query timeout
    """

    default_message = "iMessage query failed"
    default_code = ErrorCode.MSG_QUERY_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        query: str | None = None,
        db_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize an iMessage query error.

        Args:
            message: Human-readable error message.
            query: The SQL query that failed (may be truncated).
            db_path: Path to the chat.db file.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if query is not None:
            # Truncate long queries in error details
            details["query_preview"] = query[:200] + "..." if len(query) > 200 else query
        super().__init__(message, db_path=db_path, code=code, details=details, cause=cause)


# Validation Errors


class ValidationError(JarvisError):
    """Raised for input validation failures.

    Examples:
        - Invalid parameter types
        - Missing required fields
        - Value out of acceptable range
    """

    default_message = "Validation error"
    default_code = ErrorCode.VAL_INVALID_INPUT

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a validation error.

        Args:
            message: Human-readable error message.
            field: Name of the field that failed validation.
            value: The invalid value (will be converted to string).
            expected: Description of expected value/format.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, code=code, details=details, cause=cause)


# Resource Errors


class ResourceError(JarvisError):
    """Base class for system resource errors.

    Raised when system resources are insufficient or unavailable.
    """

    default_message = "Resource error"
    default_code = ErrorCode.RES_MEMORY_LOW

    def __init__(
        self,
        message: str | None = None,
        *,
        resource_type: str | None = None,
        available: int | float | None = None,
        required: int | float | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a resource error.

        Args:
            message: Human-readable error message.
            resource_type: Type of resource (memory, disk, etc.).
            available: Amount of resource available.
            required: Amount of resource required.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if available is not None:
            details["available"] = available
        if required is not None:
            details["required"] = required
        super().__init__(message, code=code, details=details, cause=cause)


class MemoryResourceError(ResourceError):
    """Raised when memory resources are insufficient.

    Examples:
        - Not enough RAM to load model
        - Memory pressure during generation
    """

    default_message = "Insufficient memory"
    default_code = ErrorCode.RES_MEMORY_LOW

    def __init__(
        self,
        message: str | None = None,
        *,
        available_mb: int | None = None,
        required_mb: int | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a memory resource error.

        Args:
            message: Human-readable error message.
            available_mb: Available memory in megabytes.
            required_mb: Required memory in megabytes.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        super().__init__(
            message,
            resource_type="memory",
            available=available_mb,
            required=required_mb,
            code=code,
            details=details,
            cause=cause,
        )


class DiskResourceError(ResourceError):
    """Raised when disk resources are insufficient or inaccessible.

    Examples:
        - Disk full
        - Disk access denied
        - Path not found
    """

    default_message = "Disk resource error"
    default_code = ErrorCode.RES_DISK_ACCESS

    def __init__(
        self,
        message: str | None = None,
        *,
        path: str | None = None,
        available_mb: int | None = None,
        required_mb: int | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize a disk resource error.

        Args:
            message: Human-readable error message.
            path: Path that caused the error.
            available_mb: Available disk space in megabytes.
            required_mb: Required disk space in megabytes.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if path:
            details["path"] = path
        super().__init__(
            message,
            resource_type="disk",
            available=available_mb,
            required=required_mb,
            code=code,
            details=details,
            cause=cause,
        )


# Experiment Errors


class ExperimentError(JarvisError):
    """Base class for experiment-related errors.

    Raised when A/B testing experiment operations fail.
    """

    default_message = "Experiment error"
    default_code = ErrorCode.EXP_NOT_FOUND

    def __init__(
        self,
        message: str | None = None,
        *,
        experiment_name: str | None = None,
        variant_id: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize an experiment error.

        Args:
            message: Human-readable error message.
            experiment_name: Name of the experiment that caused the error.
            variant_id: ID of the variant (if applicable).
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        details = details or {}
        if experiment_name:
            details["experiment_name"] = experiment_name
        if variant_id:
            details["variant_id"] = variant_id
        super().__init__(message, code=code, details=details, cause=cause)


class ExperimentNotFoundError(ExperimentError):
    """Raised when an experiment is not found.

    Examples:
        - Experiment name doesn't exist
        - Experiment was deleted
    """

    default_message = "Experiment not found"
    default_code = ErrorCode.EXP_NOT_FOUND


class ExperimentConfigError(ExperimentError):
    """Raised when experiment configuration is invalid.

    Examples:
        - Invalid variant weights
        - Missing required fields
        - YAML parse errors
    """

    default_message = "Invalid experiment configuration"
    default_code = ErrorCode.EXP_INVALID_CONFIG


# Convenience functions for common error scenarios


def model_not_found(model_path: str) -> ModelLoadError:
    """Create a ModelLoadError for a missing model.

    Args:
        model_path: Path where the model was expected.

    Returns:
        ModelLoadError with appropriate message and code.
    """
    return ModelLoadError(
        f"Model not found at: {model_path}",
        model_path=model_path,
        code=ErrorCode.MDL_NOT_FOUND,
    )


def model_out_of_memory(
    model_name: str, available_mb: int | None = None, required_mb: int | None = None
) -> ModelLoadError:
    """Create a ModelLoadError for out of memory during loading.

    Args:
        model_name: Name of the model.
        available_mb: Available memory in MB.
        required_mb: Required memory in MB.

    Returns:
        ModelLoadError with memory details.
    """
    details: dict[str, Any] = {}
    if available_mb is not None:
        details["available_mb"] = available_mb
    if required_mb is not None:
        details["required_mb"] = required_mb

    return ModelLoadError(
        f"Insufficient memory to load model: {model_name}",
        model_name=model_name,
        code=ErrorCode.RES_MEMORY_EXHAUSTED,
        details=details,
    )


def imessage_permission_denied(db_path: str | None = None) -> iMessageAccessError:
    """Create an iMessageAccessError for Full Disk Access requirement.

    Args:
        db_path: Path to the chat.db file.

    Returns:
        iMessageAccessError with permission instructions.
    """
    return iMessageAccessError(
        "Full Disk Access is required to read iMessages",
        db_path=db_path,
        requires_permission=True,
    )


def imessage_db_not_found(db_path: str) -> iMessageAccessError:
    """Create an iMessageAccessError for missing database.

    Args:
        db_path: Path where the database was expected.

    Returns:
        iMessageAccessError with appropriate code.
    """
    return iMessageAccessError(
        f"iMessage database not found at: {db_path}",
        db_path=db_path,
        code=ErrorCode.MSG_DB_NOT_FOUND,
    )


def validation_required(field: str) -> ValidationError:
    """Create a ValidationError for a missing required field.

    Args:
        field: Name of the missing field.

    Returns:
        ValidationError with appropriate details.
    """
    return ValidationError(
        f"Missing required field: {field}",
        field=field,
        code=ErrorCode.VAL_MISSING_REQUIRED,
    )


def validation_type_error(field: str, value: Any, expected: str) -> ValidationError:
    """Create a ValidationError for an incorrect type.

    Args:
        field: Name of the field.
        value: The invalid value.
        expected: Description of expected type.

    Returns:
        ValidationError with type details.
    """
    return ValidationError(
        f"Invalid type for '{field}': expected {expected}, got {type(value).__name__}",
        field=field,
        value=value,
        expected=expected,
        code=ErrorCode.VAL_TYPE_ERROR,
    )


# Export all public symbols
__all__ = [
    # Error codes
    "ErrorCode",
    # Base exception
    "JarvisError",
    # Configuration errors
    "ConfigurationError",
    # Model errors
    "ModelError",
    "ModelLoadError",
    "ModelGenerationError",
    # iMessage errors
    "iMessageError",
    "iMessageAccessError",
    "iMessageQueryError",
    # Validation errors
    "ValidationError",
    # Resource errors
    "ResourceError",
    "MemoryResourceError",
    "DiskResourceError",
    # Experiment errors
    "ExperimentError",
    "ExperimentNotFoundError",
    "ExperimentConfigError",
    # Convenience functions
    "model_not_found",
    "model_out_of_memory",
    "imessage_permission_denied",
    "imessage_db_not_found",
    "validation_required",
    "validation_type_error",
]
