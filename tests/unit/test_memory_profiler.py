"""Unit tests for WS1 Memory Profiler implementation.

Tests memory profiling functionality with mocked model loading
to avoid requiring actual MLX models during testing.
"""

from unittest.mock import MagicMock

import pytest

from benchmarks.memory.models import (
    CONTEXT_LENGTHS,
    ModelSpec,
    get_context_lengths,
    get_default_model,
    get_models_for_profiling,
)
from benchmarks.memory.profiler import (
    MLXMemoryProfiler,
    _extract_model_info,
    _get_metal_memory_mb,
    _get_process_memory,
    _unload_model,
)
from contracts.memory import MemoryProfile


class TestModelInfoExtraction:
    """Tests for _extract_model_info helper."""

    def test_extracts_model_name_from_path(self):
        """Extract model identifier from HuggingFace path."""
        path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        name, _ = _extract_model_info(path)
        assert name == "Qwen2.5-0.5B-Instruct-4bit"

    def test_extracts_4bit_quantization(self):
        """Detect 4bit quantization from model name."""
        path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        _, quant = _extract_model_info(path)
        assert quant == "4bit"

    def test_extracts_8bit_quantization(self):
        """Detect 8bit quantization from model name."""
        path = "mlx-community/Model-8bit"
        _, quant = _extract_model_info(path)
        assert quant == "8bit"

    def test_extracts_fp16_quantization(self):
        """Detect fp16 quantization from model name."""
        path = "mlx-community/Model-fp16"
        _, quant = _extract_model_info(path)
        assert quant == "fp16"

    def test_unknown_quantization(self):
        """Return unknown for unrecognized quantization."""
        path = "mlx-community/SomeModel"
        _, quant = _extract_model_info(path)
        assert quant == "unknown"

    def test_handles_local_path(self):
        """Handle local file path without org prefix."""
        path = "Qwen2.5-0.5B-4bit"
        name, quant = _extract_model_info(path)
        assert name == "Qwen2.5-0.5B-4bit"
        assert quant == "4bit"


class TestProcessMemory:
    """Tests for _get_process_memory helper."""

    def test_returns_tuple(self):
        """Returns tuple of (rss_mb, vms_mb)."""
        rss, vms = _get_process_memory()
        assert isinstance(rss, float)
        assert isinstance(vms, float)

    def test_returns_positive_values(self):
        """Memory values should be positive."""
        rss, vms = _get_process_memory()
        assert rss > 0
        assert vms > 0

    def test_rss_less_than_vms(self):
        """RSS should typically be less than or equal to VMS."""
        rss, vms = _get_process_memory()
        assert rss <= vms


class TestMetalMemory:
    """Tests for _get_metal_memory_mb helper."""

    def test_returns_float(self):
        """Returns a float value."""
        result = _get_metal_memory_mb()
        assert isinstance(result, float)

    def test_returns_non_negative(self):
        """Memory should be non-negative (0 if unavailable)."""
        result = _get_metal_memory_mb()
        assert result >= 0.0

    def test_handles_mlx_unavailable(self, monkeypatch):
        """Returns 0.0 when MLX is not available."""
        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", False)
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", None)

        result = _get_metal_memory_mb()
        assert result == 0.0

    def test_handles_attribute_error(self, monkeypatch):
        """Handles missing Metal API gracefully."""
        mock_mx = MagicMock()
        mock_mx.metal.get_peak_memory.side_effect = AttributeError("No Metal")

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        result = _get_metal_memory_mb()
        assert result == 0.0


class TestUnloadModel:
    """Tests for _unload_model helper."""

    def test_calls_metal_clear_cache(self, monkeypatch):
        """Calls _mx.metal.clear_cache() when available."""
        mock_mx = MagicMock()

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        _unload_model()

        mock_mx.metal.clear_cache.assert_called_once()

    def test_handles_metal_unavailable(self, monkeypatch):
        """Handles missing Metal gracefully."""
        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", False)
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", None)

        # Should not raise
        _unload_model()

    def test_handles_clear_cache_error(self, monkeypatch):
        """Handles clear_cache error gracefully."""
        mock_mx = MagicMock()
        mock_mx.metal.clear_cache.side_effect = AttributeError("No Metal")

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        # Should not raise
        _unload_model()


class TestMLXMemoryProfiler:
    """Tests for MLXMemoryProfiler class."""

    def test_init_creates_profiler(self):
        """Profiler can be instantiated."""
        profiler = MLXMemoryProfiler()
        assert profiler is not None
        assert profiler._baseline_rss == 0.0
        assert profiler._baseline_vms == 0.0

    def test_profile_model_returns_memory_profile(self, monkeypatch):
        """profile_model returns MemoryProfile dataclass."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def mock_load(path):
            return mock_model, mock_tokenizer

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)

        # Mock _mx.eval to be no-op
        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        profiler = MLXMemoryProfiler()
        profile = profiler.profile_model("test/model-4bit", context_length=512)

        assert isinstance(profile, MemoryProfile)
        assert profile.model_name == "model-4bit"
        assert profile.quantization == "4bit"
        assert profile.context_length == 512

    def test_profile_model_measures_load_time(self, monkeypatch):
        """Load time is measured in seconds."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def mock_load(path):
            return mock_model, mock_tokenizer

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)

        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        profiler = MLXMemoryProfiler()
        profile = profiler.profile_model("test/model", context_length=1024)

        assert profile.load_time_seconds >= 0
        assert profile.load_time_seconds < 10  # Should be fast with mock

    def test_profile_model_has_timestamp(self, monkeypatch):
        """Profile includes ISO timestamp."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def mock_load(path):
            return mock_model, mock_tokenizer

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)

        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        profiler = MLXMemoryProfiler()
        profile = profiler.profile_model("test/model", context_length=512)

        assert profile.timestamp is not None
        # ISO format check
        assert "T" in profile.timestamp

    def test_profile_model_unloads_after_profile(self, monkeypatch):
        """Model is unloaded after profiling (memory safety)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        unload_called = [False]

        def mock_load(path):
            return mock_model, mock_tokenizer

        def mock_unload():
            unload_called[0] = True

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)
        monkeypatch.setattr(benchmarks.memory.profiler, "_unload_model", mock_unload)

        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        profiler = MLXMemoryProfiler()
        profiler.profile_model("test/model", context_length=512)

        assert unload_called[0], "Model should be unloaded after profiling"

    def test_profile_model_handles_file_not_found(self, monkeypatch):
        """Raises FileNotFoundError for missing model."""

        def mock_load(path):
            raise FileNotFoundError(f"Model not found: {path}")

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)

        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        profiler = MLXMemoryProfiler()

        with pytest.raises(FileNotFoundError):
            profiler.profile_model("nonexistent/model", context_length=512)

    def test_profile_model_unloads_on_error(self, monkeypatch):
        """Model is unloaded even when profiling fails."""
        unload_called = [False]

        def mock_load(path):
            raise RuntimeError("Load failed")

        def mock_unload():
            unload_called[0] = True

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)
        monkeypatch.setattr(benchmarks.memory.profiler, "_unload_model", mock_unload)

        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        profiler = MLXMemoryProfiler()

        with pytest.raises(RuntimeError):
            profiler.profile_model("test/model", context_length=512)

        assert unload_called[0], "Model should be unloaded even on error"

    def test_profile_model_raises_when_mlx_unavailable(self, monkeypatch):
        """Raises RuntimeError when MLX is not available."""
        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", False)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", None)

        profiler = MLXMemoryProfiler()

        with pytest.raises(RuntimeError, match="MLX is not available"):
            profiler.profile_model("test/model", context_length=512)


class TestProfileWithGeneration:
    """Tests for profile_with_generation method."""

    def test_runs_generation(self, monkeypatch):
        """Runs generation to measure peak memory."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def mock_load(path):
            return mock_model, mock_tokenizer

        def mock_generate(**kwargs):
            return "Generated text"

        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", True)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", mock_load)

        mock_mx = MagicMock()
        monkeypatch.setattr(benchmarks.memory.profiler, "_mx", mock_mx)

        # Patch at the point where generate is imported in the method
        # Use sys.modules to mock mlx_lm.generate without importing
        import sys

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate = mock_generate
        original_module = sys.modules.get("mlx_lm")
        sys.modules["mlx_lm"] = mock_mlx_lm

        try:
            profiler = MLXMemoryProfiler()
            profile = profiler.profile_with_generation(
                "test/model-4bit", context_length=512, prompt="Test", max_tokens=5
            )
            assert isinstance(profile, MemoryProfile)
        finally:
            if original_module:
                sys.modules["mlx_lm"] = original_module
            else:
                del sys.modules["mlx_lm"]

    def test_raises_when_mlx_unavailable(self, monkeypatch):
        """Raises RuntimeError when MLX is not available."""
        import benchmarks.memory.profiler

        monkeypatch.setattr(benchmarks.memory.profiler, "_mlx_available", False)
        monkeypatch.setattr(benchmarks.memory.profiler, "_load", None)

        profiler = MLXMemoryProfiler()

        with pytest.raises(RuntimeError, match="MLX is not available"):
            profiler.profile_with_generation(
                "test/model", context_length=512, prompt="Test", max_tokens=5
            )


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_model_spec_fields(self):
        """ModelSpec has required fields."""
        spec = ModelSpec(
            path="test/model",
            name="Test Model",
            estimated_memory_mb=1000,
            description="A test model",
        )
        assert spec.path == "test/model"
        assert spec.name == "Test Model"
        assert spec.estimated_memory_mb == 1000
        assert spec.description == "A test model"

    def test_model_spec_is_frozen(self):
        """ModelSpec is immutable."""
        spec = ModelSpec(
            path="test/model",
            name="Test",
            estimated_memory_mb=100,
            description="Test",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.path = "other/model"


class TestModelConfigurations:
    """Tests for model configuration functions."""

    def test_context_lengths_has_required_values(self):
        """CONTEXT_LENGTHS includes 512, 1024, 2048, 4096."""
        assert 512 in CONTEXT_LENGTHS
        assert 1024 in CONTEXT_LENGTHS
        assert 2048 in CONTEXT_LENGTHS
        assert 4096 in CONTEXT_LENGTHS

    def test_get_context_lengths_returns_copy(self):
        """get_context_lengths returns a copy."""
        lengths1 = get_context_lengths()
        lengths2 = get_context_lengths()
        lengths1.append(8192)
        assert 8192 not in lengths2

    def test_get_default_model_returns_model_spec(self):
        """get_default_model returns ModelSpec."""
        model = get_default_model()
        assert isinstance(model, ModelSpec)

    def test_default_model_is_qwen(self):
        """Default model is Qwen2.5-0.5B-4bit."""
        model = get_default_model()
        assert "Qwen" in model.name
        assert "4bit" in model.path

    def test_get_models_for_profiling_returns_list(self):
        """get_models_for_profiling returns list of ModelSpec."""
        models = get_models_for_profiling()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, ModelSpec) for m in models)

    def test_get_models_includes_default(self):
        """get_models_for_profiling includes default model."""
        models = get_models_for_profiling()
        default = get_default_model()
        paths = [m.path for m in models]
        assert default.path in paths


class TestMemoryProfileContract:
    """Tests for MemoryProfile dataclass compliance."""

    def test_memory_profile_has_required_fields(self):
        """MemoryProfile has all required fields."""
        profile = MemoryProfile(
            model_name="test",
            quantization="4bit",
            context_length=512,
            rss_mb=100.0,
            virtual_mb=200.0,
            metal_mb=50.0,
            load_time_seconds=1.5,
            timestamp="2024-01-01T00:00:00Z",
        )
        assert profile.model_name == "test"
        assert profile.quantization == "4bit"
        assert profile.context_length == 512
        assert profile.rss_mb == 100.0
        assert profile.virtual_mb == 200.0
        assert profile.metal_mb == 50.0
        assert profile.load_time_seconds == 1.5
        assert profile.timestamp == "2024-01-01T00:00:00Z"


class TestProtocolCompliance:
    """Verify MLXMemoryProfiler implements MemoryProfiler protocol."""

    def test_has_profile_model_method(self):
        """Profiler has profile_model method."""
        profiler = MLXMemoryProfiler()
        assert hasattr(profiler, "profile_model")
        assert callable(profiler.profile_model)

    def test_profile_model_signature(self):
        """profile_model accepts model_path and context_length."""
        import inspect

        sig = inspect.signature(MLXMemoryProfiler.profile_model)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "context_length" in params


class TestCLIModule:
    """Tests for CLI module existence."""

    def test_run_module_exists(self):
        """run.py module can be imported."""
        from benchmarks.memory import run

        assert hasattr(run, "main")

    def test_main_returns_int(self, monkeypatch):
        """main() returns an integer exit code."""
        from benchmarks.memory import run

        # Mock argparse to avoid actual argument parsing
        mock_args = MagicMock()
        mock_args.output = MagicMock()
        mock_args.output.parent = MagicMock()
        mock_args.output.write_text = MagicMock()
        mock_args.model = None
        mock_args.context_lengths = [512]
        mock_args.quick = True
        mock_args.with_generation = False
        mock_args.verbose = False

        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: mock_args)

        # Mock profiler to avoid actual model loading
        mock_profiler = MagicMock()
        mock_profile = MemoryProfile(
            model_name="test",
            quantization="4bit",
            context_length=512,
            rss_mb=100.0,
            virtual_mb=200.0,
            metal_mb=50.0,
            load_time_seconds=1.0,
            timestamp="2024-01-01T00:00:00Z",
        )
        mock_profiler.profile_model.return_value = mock_profile

        monkeypatch.setattr(run, "MLXMemoryProfiler", lambda: mock_profiler)

        result = run.main()
        assert isinstance(result, int)
        assert result == 0  # Success
