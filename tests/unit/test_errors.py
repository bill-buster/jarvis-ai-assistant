"""Unit tests for the unified error handling system.

Tests jarvis/errors.py custom exception classes and api/errors.py handlers.
"""

import pytest

from jarvis.errors import (
    ConfigurationError,
    DiskResourceError,
    ErrorCode,
    JarvisError,
    MemoryResourceError,
    ModelError,
    ModelGenerationError,
    ModelLoadError,
    ResourceError,
    ValidationError,
    imessage_db_not_found,
    imessage_permission_denied,
    iMessageAccessError,
    iMessageError,
    iMessageQueryError,
    model_not_found,
    model_out_of_memory,
    validation_required,
    validation_type_error,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_all_error_codes_have_string_values(self):
        """All error codes have string values."""
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert len(code.value) > 0

    def test_error_code_values_are_unique(self):
        """All error code values are unique."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))

    def test_error_code_categories(self):
        """Error codes follow category prefixes."""
        for code in ErrorCode:
            if code == ErrorCode.UNKNOWN:
                continue
            # Should have a prefix like CFG_, MDL_, MSG_, VAL_, RES_, TSK_, CAL_, EXP_
            assert any(
                code.value.startswith(prefix)
                for prefix in ["CFG_", "MDL_", "MSG_", "VAL_", "RES_", "TSK_", "CAL_", "EXP_"]
            )


class TestJarvisError:
    """Tests for JarvisError base class."""

    def test_default_message(self):
        """Has sensible default message."""
        error = JarvisError()
        assert error.message == "An error occurred"
        assert str(error) == "An error occurred"

    def test_custom_message(self):
        """Can provide custom message."""
        error = JarvisError("Custom error")
        assert error.message == "Custom error"
        assert str(error) == "Custom error"

    def test_default_code(self):
        """Has default error code."""
        error = JarvisError()
        assert error.code == ErrorCode.UNKNOWN

    def test_custom_code(self):
        """Can provide custom code."""
        error = JarvisError("Error", code=ErrorCode.CFG_INVALID)
        assert error.code == ErrorCode.CFG_INVALID

    def test_details(self):
        """Can include additional details."""
        error = JarvisError("Error", details={"key": "value"})
        assert error.details == {"key": "value"}

    def test_cause(self):
        """Can chain cause exception."""
        cause = ValueError("Original error")
        error = JarvisError("Wrapper", cause=cause)
        assert error.cause is cause
        assert error.__cause__ is cause

    def test_to_dict(self):
        """Can convert to dictionary for API responses."""
        error = JarvisError("Test error", code=ErrorCode.UNKNOWN, details={"foo": "bar"})
        result = error.to_dict()
        assert result["error"] == "JarvisError"
        assert result["code"] == "UNKNOWN"
        assert result["detail"] == "Test error"
        assert result["details"] == {"foo": "bar"}

    def test_repr(self):
        """Has useful repr output."""
        error = JarvisError("Test", code=ErrorCode.CFG_INVALID)
        assert "JarvisError" in repr(error)
        assert "Test" in repr(error)

    def test_is_exception(self):
        """Is a proper exception that can be raised."""
        with pytest.raises(JarvisError):
            raise JarvisError("Test error")


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_default_message(self):
        """Has sensible default message."""
        error = ConfigurationError()
        assert "configuration" in error.message.lower()

    def test_default_code(self):
        """Has appropriate default code."""
        error = ConfigurationError()
        assert error.code == ErrorCode.CFG_INVALID

    def test_with_config_key(self):
        """Can include config key in details."""
        error = ConfigurationError("Bad config", config_key="api_key")
        assert error.details["config_key"] == "api_key"

    def test_with_config_path(self):
        """Can include config path in details."""
        error = ConfigurationError("Bad config", config_path="/path/to/config")
        assert error.details["config_path"] == "/path/to/config"

    def test_inheritance(self):
        """Inherits from JarvisError."""
        error = ConfigurationError()
        assert isinstance(error, JarvisError)


class TestModelError:
    """Tests for ModelError and subclasses."""

    def test_model_error_default_message(self):
        """ModelError has sensible default message."""
        error = ModelError()
        assert "model" in error.message.lower()

    def test_model_error_with_model_name(self):
        """Can include model name in details."""
        error = ModelError("Failed", model_name="qwen-0.5b")
        assert error.details["model_name"] == "qwen-0.5b"

    def test_model_error_with_model_path(self):
        """Can include model path in details."""
        error = ModelError("Failed", model_path="/path/to/model")
        assert error.details["model_path"] == "/path/to/model"

    def test_model_load_error_default(self):
        """ModelLoadError has appropriate defaults."""
        error = ModelLoadError()
        assert "load" in error.message.lower()
        assert error.code == ErrorCode.MDL_LOAD_FAILED

    def test_model_load_error_inheritance(self):
        """ModelLoadError inherits from ModelError."""
        error = ModelLoadError()
        assert isinstance(error, ModelError)
        assert isinstance(error, JarvisError)

    def test_model_generation_error_default(self):
        """ModelGenerationError has appropriate defaults."""
        error = ModelGenerationError()
        assert "generation" in error.message.lower()
        assert error.code == ErrorCode.MDL_GENERATION_FAILED

    def test_model_generation_error_with_timeout(self):
        """ModelGenerationError can include timeout."""
        error = ModelGenerationError("Timed out", timeout_seconds=30.0)
        assert error.details["timeout_seconds"] == 30.0
        assert error.code == ErrorCode.MDL_TIMEOUT

    def test_model_generation_error_with_prompt(self):
        """ModelGenerationError truncates long prompts."""
        long_prompt = "x" * 300
        error = ModelGenerationError("Failed", prompt=long_prompt)
        assert len(error.details["prompt_preview"]) < 300
        assert error.details["prompt_preview"].endswith("...")


class TestiMessageError:
    """Tests for iMessageError and subclasses."""

    def test_imessage_error_default(self):
        """iMessageError has sensible defaults."""
        error = iMessageError()
        assert "imessage" in error.message.lower()

    def test_imessage_error_with_db_path(self):
        """Can include db path in details."""
        error = iMessageError("Failed", db_path="/path/to/chat.db")
        assert error.details["db_path"] == "/path/to/chat.db"

    def test_imessage_access_error_default(self):
        """iMessageAccessError has appropriate defaults."""
        error = iMessageAccessError()
        assert "access" in error.message.lower()
        assert error.code == ErrorCode.MSG_ACCESS_DENIED

    def test_imessage_access_error_with_permission(self):
        """iMessageAccessError can include permission instructions."""
        error = iMessageAccessError("Permission denied", requires_permission=True)
        assert error.details["requires_permission"] is True
        assert "permission_instructions" in error.details
        assert len(error.details["permission_instructions"]) > 0

    def test_imessage_query_error_default(self):
        """iMessageQueryError has appropriate defaults."""
        error = iMessageQueryError()
        assert "query" in error.message.lower()
        assert error.code == ErrorCode.MSG_QUERY_FAILED

    def test_imessage_query_error_truncates_long_query(self):
        """iMessageQueryError truncates long queries."""
        long_query = "SELECT " + "x" * 300
        error = iMessageQueryError("Failed", query=long_query)
        assert len(error.details["query_preview"]) < 300
        assert error.details["query_preview"].endswith("...")


class TestValidationError:
    """Tests for ValidationError."""

    def test_default_message(self):
        """Has sensible default message."""
        error = ValidationError()
        assert "validation" in error.message.lower()

    def test_default_code(self):
        """Has appropriate default code."""
        error = ValidationError()
        assert error.code == ErrorCode.VAL_INVALID_INPUT

    def test_with_field(self):
        """Can include field name."""
        error = ValidationError("Invalid field", field="email")
        assert error.details["field"] == "email"

    def test_with_value(self):
        """Can include invalid value."""
        error = ValidationError("Invalid", field="age", value=-5)
        assert error.details["value"] == "-5"

    def test_with_expected(self):
        """Can include expected format."""
        error = ValidationError("Invalid", field="date", expected="YYYY-MM-DD")
        assert error.details["expected"] == "YYYY-MM-DD"


class TestResourceError:
    """Tests for ResourceError and subclasses."""

    def test_resource_error_default(self):
        """ResourceError has sensible defaults."""
        error = ResourceError()
        assert "resource" in error.message.lower()

    def test_resource_error_with_details(self):
        """Can include resource details."""
        error = ResourceError(
            "Insufficient memory",
            resource_type="memory",
            available=1024,
            required=2048,
        )
        assert error.details["resource_type"] == "memory"
        assert error.details["available"] == 1024
        assert error.details["required"] == 2048

    def test_memory_resource_error(self):
        """MemoryResourceError has appropriate defaults."""
        error = MemoryResourceError()
        assert "memory" in error.message.lower()
        assert error.code == ErrorCode.RES_MEMORY_LOW

    def test_memory_resource_error_with_mb(self):
        """MemoryResourceError includes MB values."""
        error = MemoryResourceError("Low memory", available_mb=512, required_mb=1024)
        assert error.details["available"] == 512
        assert error.details["required"] == 1024

    def test_disk_resource_error(self):
        """DiskResourceError has appropriate defaults."""
        error = DiskResourceError()
        assert error.code == ErrorCode.RES_DISK_ACCESS

    def test_disk_resource_error_with_path(self):
        """DiskResourceError can include path."""
        error = DiskResourceError("Disk full", path="/var/data")
        assert error.details["path"] == "/var/data"


class TestConvenienceFunctions:
    """Tests for convenience error factory functions."""

    def test_model_not_found(self):
        """model_not_found creates appropriate error."""
        error = model_not_found("/path/to/model")
        assert isinstance(error, ModelLoadError)
        assert error.code == ErrorCode.MDL_NOT_FOUND
        assert "/path/to/model" in error.message
        assert error.details["model_path"] == "/path/to/model"

    def test_model_out_of_memory(self):
        """model_out_of_memory creates appropriate error."""
        error = model_out_of_memory("qwen-3b", available_mb=1024, required_mb=2048)
        assert isinstance(error, ModelLoadError)
        assert error.code == ErrorCode.RES_MEMORY_EXHAUSTED
        assert "qwen-3b" in error.message
        assert error.details["available_mb"] == 1024
        assert error.details["required_mb"] == 2048

    def test_imessage_permission_denied(self):
        """imessage_permission_denied creates appropriate error."""
        error = imessage_permission_denied("/path/to/chat.db")
        assert isinstance(error, iMessageAccessError)
        assert error.details["requires_permission"] is True
        assert "permission_instructions" in error.details

    def test_imessage_db_not_found(self):
        """imessage_db_not_found creates appropriate error."""
        error = imessage_db_not_found("/path/to/chat.db")
        assert isinstance(error, iMessageAccessError)
        assert error.code == ErrorCode.MSG_DB_NOT_FOUND
        assert "/path/to/chat.db" in error.message

    def test_validation_required(self):
        """validation_required creates appropriate error."""
        error = validation_required("email")
        assert isinstance(error, ValidationError)
        assert error.code == ErrorCode.VAL_MISSING_REQUIRED
        assert "email" in error.message
        assert error.details["field"] == "email"

    def test_validation_type_error(self):
        """validation_type_error creates appropriate error."""
        error = validation_type_error("age", "not a number", "integer")
        assert isinstance(error, ValidationError)
        assert error.code == ErrorCode.VAL_TYPE_ERROR
        assert error.details["field"] == "age"
        assert error.details["expected"] == "integer"


class TestExceptionHandling:
    """Tests for exception handling patterns."""

    def test_catch_jarvis_error_catches_all_subclasses(self):
        """Catching JarvisError catches all subclasses."""
        errors = [
            ConfigurationError(),
            ModelError(),
            ModelLoadError(),
            ModelGenerationError(),
            iMessageError(),
            iMessageAccessError(),
            iMessageQueryError(),
            ValidationError(),
            ResourceError(),
            MemoryResourceError(),
            DiskResourceError(),
        ]

        for error in errors:
            caught = False
            try:
                raise error
            except JarvisError:
                caught = True
            assert caught, f"Failed to catch {type(error).__name__}"

    def test_exception_chaining_preserves_cause(self):
        """Exception chaining preserves the original cause."""
        original = ValueError("Original error")
        wrapper = JarvisError("Wrapped error", cause=original)

        try:
            raise wrapper
        except JarvisError as e:
            assert e.__cause__ is original

    def test_to_dict_excludes_empty_details(self):
        """to_dict excludes details when empty."""
        error = JarvisError("Test")
        result = error.to_dict()
        assert "details" not in result


class TestAPIErrorHandlers:
    """Tests for API error handler integration."""

    def test_error_status_code_mapping_exists(self):
        """API error handlers have status code mapping."""
        from api.errors import ERROR_STATUS_CODES

        # All error types should have a mapping
        assert JarvisError in ERROR_STATUS_CODES
        assert ModelLoadError in ERROR_STATUS_CODES
        assert iMessageAccessError in ERROR_STATUS_CODES
        assert ValidationError in ERROR_STATUS_CODES

    def test_get_status_code_for_error(self):
        """get_status_code_for_error returns appropriate codes."""
        from api.errors import get_status_code_for_error

        # 400 for validation errors
        assert get_status_code_for_error(ValidationError()) == 400

        # 403 for access denied
        assert get_status_code_for_error(iMessageAccessError()) == 403

        # 503 for model load failures
        assert get_status_code_for_error(ModelLoadError()) == 503

        # 500 for generic errors
        assert get_status_code_for_error(JarvisError()) == 500

    def test_build_error_response_format(self):
        """build_error_response returns correct format."""
        from api.errors import build_error_response

        error = ValidationError("Invalid input", field="email")
        response = build_error_response(error)

        assert "error" in response
        assert "code" in response
        assert "detail" in response
        assert response["error"] == "ValidationError"
        assert response["code"] == ErrorCode.VAL_INVALID_INPUT.value
        assert response["detail"] == "Invalid input"
        assert "details" in response
        assert response["details"]["field"] == "email"
