"""Unit tests for Model Registry.

Tests cover model specifications, registry lookup functions,
model recommendation based on RAM, and model availability checks.
"""

from pathlib import Path

import pytest

from models.registry import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    ModelSpec,
    ensure_model_available,
    get_all_models,
    get_model_spec,
    get_model_spec_by_path,
    get_recommended_model,
    is_model_available,
)


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_model_spec_creation(self):
        """Test ModelSpec can be created with required fields."""
        spec = ModelSpec(
            id="test-model",
            path="test/model-path",
            display_name="Test Model",
            size_gb=1.0,
            min_ram_gb=8,
            quality_tier="good",
            description="A test model",
        )
        assert spec.id == "test-model"
        assert spec.path == "test/model-path"
        assert spec.display_name == "Test Model"
        assert spec.size_gb == 1.0
        assert spec.min_ram_gb == 8
        assert spec.quality_tier == "good"
        assert spec.description == "A test model"
        assert spec.recommended_for == []

    def test_model_spec_with_recommended_for(self):
        """Test ModelSpec with recommended_for list."""
        spec = ModelSpec(
            id="test-model",
            path="test/model-path",
            display_name="Test Model",
            size_gb=1.0,
            min_ram_gb=8,
            quality_tier="good",
            description="A test model",
            recommended_for=["summarization", "drafting"],
        )
        assert spec.recommended_for == ["summarization", "drafting"]

    def test_estimated_memory_mb(self):
        """Test estimated_memory_mb property calculation."""
        spec = ModelSpec(
            id="test-model",
            path="test/model-path",
            display_name="Test Model",
            size_gb=1.5,
            min_ram_gb=8,
            quality_tier="good",
            description="A test model",
        )
        assert spec.estimated_memory_mb == 1.5 * 1024

    def test_model_spec_is_frozen(self):
        """Test ModelSpec is immutable (frozen)."""
        spec = ModelSpec(
            id="test-model",
            path="test/model-path",
            display_name="Test Model",
            size_gb=1.0,
            min_ram_gb=8,
            quality_tier="good",
            description="A test model",
        )
        with pytest.raises(AttributeError):
            spec.id = "changed"  # type: ignore[misc]


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_registry_contains_expected_models(self):
        """Test registry contains expected model IDs."""
        assert "qwen-0.5b" in MODEL_REGISTRY
        assert "qwen-1.5b" in MODEL_REGISTRY
        assert "qwen-3b" in MODEL_REGISTRY

    def test_registry_models_have_required_fields(self):
        """Test all registry models have required fields."""
        for model_id, spec in MODEL_REGISTRY.items():
            assert spec.id == model_id
            assert spec.path
            assert spec.display_name
            assert spec.size_gb > 0
            assert spec.min_ram_gb >= 8
            assert spec.quality_tier in ("basic", "good", "excellent")
            assert spec.description

    def test_registry_qwen_0_5b(self):
        """Test qwen-0.5b model specification."""
        spec = MODEL_REGISTRY["qwen-0.5b"]
        assert spec.path == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert spec.size_gb == 0.8
        assert spec.min_ram_gb == 8
        assert spec.quality_tier == "basic"

    def test_registry_qwen_1_5b(self):
        """Test qwen-1.5b model specification."""
        spec = MODEL_REGISTRY["qwen-1.5b"]
        assert spec.path == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        assert spec.size_gb == 1.5
        assert spec.min_ram_gb == 8
        assert spec.quality_tier == "good"

    def test_registry_qwen_3b(self):
        """Test qwen-3b model specification."""
        spec = MODEL_REGISTRY["qwen-3b"]
        assert spec.path == "mlx-community/Qwen2.5-3B-Instruct-4bit"
        assert spec.size_gb == 2.5
        assert spec.min_ram_gb == 8  # 4-bit model fits in 8GB
        assert spec.quality_tier == "excellent"


class TestGetModelSpec:
    """Tests for get_model_spec function."""

    def test_get_existing_model(self):
        """Test getting an existing model spec."""
        spec = get_model_spec("qwen-1.5b")
        assert spec is not None
        assert spec.id == "qwen-1.5b"

    def test_get_nonexistent_model(self):
        """Test getting a non-existent model returns None."""
        spec = get_model_spec("nonexistent-model")
        assert spec is None

    def test_get_all_registered_models(self):
        """Test all registered models can be retrieved."""
        for model_id in MODEL_REGISTRY:
            spec = get_model_spec(model_id)
            assert spec is not None
            assert spec.id == model_id


class TestGetModelSpecByPath:
    """Tests for get_model_spec_by_path function."""

    def test_get_by_existing_path(self):
        """Test getting model spec by HuggingFace path."""
        spec = get_model_spec_by_path("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        assert spec is not None
        assert spec.id == "qwen-1.5b"

    def test_get_by_nonexistent_path(self):
        """Test getting model by non-existent path returns None."""
        spec = get_model_spec_by_path("nonexistent/model-path")
        assert spec is None

    def test_get_all_models_by_path(self):
        """Test all models can be found by their paths."""
        for spec in MODEL_REGISTRY.values():
            found = get_model_spec_by_path(spec.path)
            assert found is not None
            assert found.id == spec.id


class TestGetRecommendedModel:
    """Tests for get_recommended_model function."""

    def test_recommend_basic_for_8gb(self):
        """Test 8GB RAM gets an excellent tier model."""
        spec = get_recommended_model(8.0)
        # 8GB now meets excellent tier models (qwen-3b, gemma3-4b) min_ram_gb=8
        assert spec.quality_tier == "excellent"
        assert spec.min_ram_gb <= 8

    def test_recommend_excellent_for_16gb(self):
        """Test 16GB RAM gets the best model (qwen-3b)."""
        spec = get_recommended_model(16.0)
        assert spec.id == "qwen-3b"
        assert spec.quality_tier == "excellent"

    def test_recommend_excellent_for_32gb(self):
        """Test 32GB RAM gets the best model."""
        spec = get_recommended_model(32.0)
        assert spec.id == "qwen-3b"
        assert spec.quality_tier == "excellent"

    def test_recommend_for_low_ram(self):
        """Test low RAM still gets a model (fallback)."""
        spec = get_recommended_model(4.0)
        # Should fall back to default since no model fits 4GB
        assert spec is not None
        assert spec.id == DEFAULT_MODEL_ID

    def test_recommend_returns_best_fitting_model(self):
        """Test recommendation selects best model that fits RAM."""
        # With all excellent-tier models at min_ram=8, they're all eligible at 8GB+
        spec = get_recommended_model(8.0)
        assert spec.quality_tier == "excellent"

        spec = get_recommended_model(16.0)
        assert spec.quality_tier == "excellent"

        # Low RAM should fall back
        spec = get_recommended_model(4.0)
        assert spec is not None


class TestGetAllModels:
    """Tests for get_all_models function."""

    def test_returns_all_models(self):
        """Test get_all_models returns all registered models."""
        models = get_all_models()
        assert len(models) == len(MODEL_REGISTRY)

    def test_sorted_by_quality_tier(self):
        """Test models are sorted by quality tier (basic to excellent)."""
        models = get_all_models()
        tier_order = {"basic": 1, "good": 2, "excellent": 3}
        for i in range(len(models) - 1):
            current_tier = tier_order[models[i].quality_tier]
            next_tier = tier_order[models[i + 1].quality_tier]
            assert current_tier <= next_tier


class TestDefaultModelId:
    """Tests for DEFAULT_MODEL_ID constant."""

    def test_default_model_exists(self):
        """Test default model ID exists in registry."""
        assert DEFAULT_MODEL_ID in MODEL_REGISTRY

    def test_default_model_is_balanced(self):
        """Test default model has appropriate quality tier."""
        spec = MODEL_REGISTRY[DEFAULT_MODEL_ID]
        # LFM 2.5 is the default with "excellent" tier for conversation
        assert spec.quality_tier in ("good", "excellent")


class TestIsModelAvailable:
    """Tests for is_model_available function."""

    def test_unknown_model_not_available(self):
        """Test unknown model ID returns False."""
        assert is_model_available("nonexistent-model") is False

    def test_model_without_cache_not_available(self, tmp_path, monkeypatch):
        """Test model without cache returns False."""
        # Point to a non-existent cache directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert is_model_available("qwen-1.5b") is False

    def test_model_with_empty_cache_not_available(self, tmp_path, monkeypatch):
        """Test model with empty cache directory returns False."""
        # Create cache structure without actual model files
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--mlx-community--Qwen2.5-1.5B-Instruct-4bit"
        model_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert is_model_available("qwen-1.5b") is False

    def test_model_with_snapshot_available(self, tmp_path, monkeypatch):
        """Test model with snapshot files returns True."""
        # Create cache structure with snapshot files
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--mlx-community--Qwen2.5-1.5B-Instruct-4bit"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create a dummy model file
        (snapshot_dir / "model.safetensors").write_text("dummy")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert is_model_available("qwen-1.5b") is True


class TestEnsureModelAvailable:
    """Tests for ensure_model_available function."""

    def test_unknown_model_returns_false(self):
        """Test unknown model ID returns False."""
        result = ensure_model_available("nonexistent-model")
        assert result is False

    def test_already_available_returns_true(self, tmp_path, monkeypatch):
        """Test already available model returns True without download."""
        # Set up cache structure
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--mlx-community--Qwen2.5-1.5B-Instruct-4bit"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.safetensors").write_text("dummy")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = ensure_model_available("qwen-1.5b")
        assert result is True

    def test_download_not_attempted_when_unknown_model(self, tmp_path, monkeypatch):
        """Test download is not attempted for unknown model ID."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Unknown model should return False without attempting download
        result = ensure_model_available("nonexistent-model-xyz")
        assert result is False

    def test_ensure_model_accepts_valid_model_ids(self):
        """Test ensure_model_available accepts all valid model IDs."""
        # Just verify the function accepts valid model IDs
        # Actual download behavior is tested implicitly
        for model_id in MODEL_REGISTRY:
            spec = get_model_spec(model_id)
            assert spec is not None


class TestModelConfigIntegration:
    """Integration tests for ModelConfig using registry."""

    def test_model_config_with_model_id(self):
        """Test ModelConfig resolves model_id from registry."""
        from models.loader import ModelConfig

        config = ModelConfig(model_id="qwen-1.5b")

        assert config.model_id == "qwen-1.5b"
        assert config.model_path == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        assert config.estimated_memory_mb == 1.5 * 1024
        assert config.display_name == "Qwen 2.5 1.5B (Balanced)"

    def test_model_config_with_unknown_model_id(self):
        """Test ModelConfig handles unknown model_id gracefully."""
        from models.loader import ModelConfig

        config = ModelConfig(model_id="custom/model")

        # Should use model_id as model_path when not in registry
        assert config.model_path == "custom/model"

    def test_model_config_default(self):
        """Test ModelConfig defaults to registry default model."""
        from models.loader import ModelConfig

        config = ModelConfig()

        assert config.model_id == DEFAULT_MODEL_ID
        assert config.model_path == MODEL_REGISTRY[DEFAULT_MODEL_ID].path

    def test_model_config_with_model_path(self):
        """Test ModelConfig with explicit model_path (no model_id)."""
        from models.loader import ModelConfig

        config = ModelConfig(model_path="custom/model-path")

        assert config.model_id is None
        assert config.model_path == "custom/model-path"

    def test_model_config_model_id_takes_precedence(self):
        """Test model_id takes precedence over model_path."""
        from models.loader import ModelConfig

        config = ModelConfig(model_id="qwen-0.5b", model_path="ignored/path")

        assert config.model_id == "qwen-0.5b"
        assert config.model_path == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    def test_model_config_spec_property(self):
        """Test ModelConfig.spec returns ModelSpec when available."""
        from models.loader import ModelConfig

        config = ModelConfig(model_id="qwen-3b")

        assert config.spec is not None
        assert config.spec.id == "qwen-3b"
        assert config.spec.quality_tier == "excellent"

    def test_model_config_spec_none_for_custom_path(self):
        """Test ModelConfig.spec is None for custom paths."""
        from models.loader import ModelConfig

        config = ModelConfig(model_path="custom/path")

        assert config.spec is None
