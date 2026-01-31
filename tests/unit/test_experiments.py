"""Unit tests for the A/B testing infrastructure.

Tests cover:
- Experiment configuration loading from YAML
- Deterministic variant assignment per contact
- Outcome recording and persistence
- Statistical significance calculation
- Edge cases (empty experiments, malformed YAML)
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from jarvis.experiments import (
    Experiment,
    ExperimentManager,
    ExperimentOutcome,
    UserAction,
    VariantConfig,
    VariantResults,
    get_experiment_manager,
    reset_experiment_manager,
    statistical_significance,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_experiments_yaml():
    """Return sample experiments YAML content."""
    return {
        "experiments": [
            {
                "name": "reply_tone_test",
                "enabled": True,
                "description": "Test different reply tones",
                "created_at": "2024-01-01T00:00:00",
                "variants": [
                    {
                        "id": "control",
                        "weight": 50,
                        "config": {"system_prompt": "You are helpful.", "temperature": 0.7},
                    },
                    {
                        "id": "treatment",
                        "weight": 50,
                        "config": {"system_prompt": "You are casual.", "temperature": 0.8},
                    },
                ],
            },
            {
                "name": "disabled_experiment",
                "enabled": False,
                "variants": [
                    {"id": "a", "weight": 100, "config": {}},
                ],
            },
        ]
    }


@pytest.fixture
def sample_outcomes_json():
    """Return sample outcomes JSON content."""
    return {
        "outcomes": [
            {
                "experiment_name": "reply_tone_test",
                "variant_id": "control",
                "contact_id": "user1",
                "action": "sent_unchanged",
                "timestamp": "2024-01-01T12:00:00",
            },
            {
                "experiment_name": "reply_tone_test",
                "variant_id": "treatment",
                "contact_id": "user2",
                "action": "dismissed",
                "timestamp": "2024-01-01T12:01:00",
            },
        ]
    }


@pytest.fixture
def experiment_manager(temp_dir, sample_experiments_yaml, sample_outcomes_json):
    """Create an ExperimentManager with test data."""
    experiments_path = temp_dir / "experiments.yaml"
    outcomes_path = temp_dir / "outcomes.json"

    # Write test data
    with experiments_path.open("w") as f:
        yaml.safe_dump(sample_experiments_yaml, f)

    with outcomes_path.open("w") as f:
        json.dump(sample_outcomes_json, f)

    manager = ExperimentManager(
        experiments_path=experiments_path,
        outcomes_path=outcomes_path,
    )
    return manager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    reset_experiment_manager()
    yield
    reset_experiment_manager()


# =============================================================================
# UserAction Tests
# =============================================================================


class TestUserAction:
    """Tests for UserAction enum."""

    def test_user_action_values(self):
        """Test UserAction enum values."""
        assert UserAction.SENT_UNCHANGED.value == "sent_unchanged"
        assert UserAction.SENT_EDITED.value == "sent_edited"
        assert UserAction.DISMISSED.value == "dismissed"
        assert UserAction.REGENERATED.value == "regenerated"

    def test_user_action_from_string(self):
        """Test creating UserAction from string."""
        assert UserAction("sent_unchanged") == UserAction.SENT_UNCHANGED
        assert UserAction("dismissed") == UserAction.DISMISSED


# =============================================================================
# VariantConfig Tests
# =============================================================================


class TestVariantConfig:
    """Tests for VariantConfig dataclass."""

    def test_variant_config_creation(self):
        """Test creating a VariantConfig."""
        config = VariantConfig(
            id="control",
            weight=50,
            config={"temperature": 0.7},
        )

        assert config.id == "control"
        assert config.weight == 50
        assert config.config == {"temperature": 0.7}

    def test_variant_config_default_config(self):
        """Test default empty config."""
        config = VariantConfig(id="test", weight=100)

        assert config.config == {}


# =============================================================================
# Experiment Tests
# =============================================================================


class TestExperiment:
    """Tests for Experiment dataclass."""

    def test_experiment_creation(self):
        """Test creating an Experiment."""
        variants = [
            VariantConfig(id="a", weight=50),
            VariantConfig(id="b", weight=50),
        ]
        exp = Experiment(
            name="test_exp",
            enabled=True,
            variants=variants,
            description="Test experiment",
        )

        assert exp.name == "test_exp"
        assert exp.enabled is True
        assert len(exp.variants) == 2

    def test_get_variant_by_id(self):
        """Test getting variant by ID."""
        variants = [
            VariantConfig(id="control", weight=50),
            VariantConfig(id="treatment", weight=50),
        ]
        exp = Experiment(name="test", enabled=True, variants=variants)

        assert exp.get_variant_by_id("control").id == "control"
        assert exp.get_variant_by_id("treatment").id == "treatment"
        assert exp.get_variant_by_id("nonexistent") is None


# =============================================================================
# ExperimentOutcome Tests
# =============================================================================


class TestExperimentOutcome:
    """Tests for ExperimentOutcome dataclass."""

    def test_outcome_to_dict(self):
        """Test converting outcome to dict."""
        outcome = ExperimentOutcome(
            experiment_name="test",
            variant_id="control",
            contact_id="user1",
            action=UserAction.SENT_UNCHANGED,
            timestamp="2024-01-01T00:00:00",
        )

        d = outcome.to_dict()
        assert d["experiment_name"] == "test"
        assert d["variant_id"] == "control"
        assert d["action"] == "sent_unchanged"

    def test_outcome_from_dict(self):
        """Test creating outcome from dict."""
        data = {
            "experiment_name": "test",
            "variant_id": "control",
            "contact_id": "user1",
            "action": "dismissed",
            "timestamp": "2024-01-01T00:00:00",
        }

        outcome = ExperimentOutcome.from_dict(data)
        assert outcome.experiment_name == "test"
        assert outcome.action == UserAction.DISMISSED


# =============================================================================
# VariantResults Tests
# =============================================================================


class TestVariantResults:
    """Tests for VariantResults dataclass."""

    def test_conversion_rate_zero_impressions(self):
        """Test conversion rate with zero impressions."""
        results = VariantResults(variant_id="test")
        assert results.conversion_rate == 0.0

    def test_conversion_rate_calculation(self):
        """Test conversion rate calculation."""
        results = VariantResults(
            variant_id="test",
            total_impressions=100,
            sent_unchanged=25,
        )
        assert results.conversion_rate == 25.0

    def test_conversion_rate_with_all_converted(self):
        """Test 100% conversion rate."""
        results = VariantResults(
            variant_id="test",
            total_impressions=50,
            sent_unchanged=50,
        )
        assert results.conversion_rate == 100.0


# =============================================================================
# ExperimentManager Tests
# =============================================================================


class TestExperimentManagerLoading:
    """Tests for ExperimentManager loading functionality."""

    def test_load_experiments_from_yaml(self, experiment_manager):
        """Test loading experiments from YAML file."""
        experiments = experiment_manager.get_experiments()

        assert len(experiments) == 2
        exp_names = [e.name for e in experiments]
        assert "reply_tone_test" in exp_names
        assert "disabled_experiment" in exp_names

    def test_load_outcomes_from_json(self, experiment_manager):
        """Test loading outcomes from JSON file."""
        outcomes = experiment_manager.get_outcomes_for_experiment("reply_tone_test")

        assert len(outcomes) == 2

    def test_load_nonexistent_files(self, temp_dir):
        """Test loading with nonexistent files."""
        manager = ExperimentManager(
            experiments_path=temp_dir / "missing.yaml",
            outcomes_path=temp_dir / "missing.json",
        )

        experiments = manager.get_experiments()
        assert len(experiments) == 0

    def test_load_malformed_yaml(self, temp_dir):
        """Test loading malformed YAML."""
        experiments_path = temp_dir / "bad.yaml"
        experiments_path.write_text("{ invalid yaml: [")

        manager = ExperimentManager(
            experiments_path=experiments_path,
            outcomes_path=temp_dir / "outcomes.json",
        )

        experiments = manager.get_experiments()
        assert len(experiments) == 0

    def test_load_malformed_json(self, temp_dir):
        """Test loading malformed JSON outcomes."""
        experiments_path = temp_dir / "experiments.yaml"
        outcomes_path = temp_dir / "bad.json"

        yaml.safe_dump({"experiments": []}, experiments_path.open("w"))
        outcomes_path.write_text("{ invalid json")

        manager = ExperimentManager(
            experiments_path=experiments_path,
            outcomes_path=outcomes_path,
        )

        # Should not raise, just have empty outcomes
        manager._ensure_loaded()
        assert len(manager._outcomes) == 0


class TestExperimentManagerVariants:
    """Tests for variant assignment."""

    def test_get_active_experiments(self, experiment_manager):
        """Test getting only active experiments."""
        active = experiment_manager.get_active_experiments()

        assert len(active) == 1
        assert active[0].name == "reply_tone_test"

    def test_get_experiment_by_name(self, experiment_manager):
        """Test getting experiment by name."""
        exp = experiment_manager.get_experiment("reply_tone_test")

        assert exp is not None
        assert exp.name == "reply_tone_test"
        assert len(exp.variants) == 2

    def test_get_nonexistent_experiment(self, experiment_manager):
        """Test getting nonexistent experiment."""
        exp = experiment_manager.get_experiment("nonexistent")
        assert exp is None

    def test_get_variant_for_contact(self, experiment_manager):
        """Test getting variant for a contact."""
        variant = experiment_manager.get_variant("reply_tone_test", "user@example.com")

        assert variant is not None
        assert variant.id in ("control", "treatment")

    def test_variant_assignment_is_deterministic(self, experiment_manager):
        """Test that variant assignment is deterministic."""
        contact_id = "test@example.com"

        variant1 = experiment_manager.get_variant("reply_tone_test", contact_id)
        variant2 = experiment_manager.get_variant("reply_tone_test", contact_id)
        variant3 = experiment_manager.get_variant("reply_tone_test", contact_id)

        # Same contact should always get same variant
        assert variant1.id == variant2.id == variant3.id

    def test_different_contacts_can_get_different_variants(self, experiment_manager):
        """Test that different contacts can get different variants."""
        # Generate many contacts and verify distribution
        variants = set()
        for i in range(100):
            variant = experiment_manager.get_variant("reply_tone_test", f"user{i}@test.com")
            variants.add(variant.id)

        # With 100 contacts and 50/50 split, both variants should appear
        assert len(variants) == 2

    def test_get_variant_disabled_experiment(self, experiment_manager):
        """Test getting variant for disabled experiment returns None."""
        variant = experiment_manager.get_variant("disabled_experiment", "user@test.com")
        assert variant is None

    def test_get_variant_nonexistent_experiment(self, experiment_manager):
        """Test getting variant for nonexistent experiment."""
        variant = experiment_manager.get_variant("nonexistent", "user@test.com")
        assert variant is None


class TestExperimentManagerOutcomes:
    """Tests for outcome recording."""

    def test_record_outcome(self, experiment_manager):
        """Test recording an outcome."""
        result = experiment_manager.record_outcome(
            experiment_name="reply_tone_test",
            variant_id="control",
            contact_id="new_user",
            action=UserAction.SENT_UNCHANGED,
        )

        assert result is True

        # Verify outcome was recorded
        outcomes = experiment_manager.get_outcomes_for_experiment("reply_tone_test")
        assert len(outcomes) == 3  # 2 initial + 1 new

    def test_record_outcome_with_string_action(self, experiment_manager):
        """Test recording outcome with string action."""
        result = experiment_manager.record_outcome(
            experiment_name="reply_tone_test",
            variant_id="treatment",
            contact_id="user3",
            action="sent_edited",  # String instead of enum
        )

        assert result is True

    def test_record_outcome_invalid_action(self, experiment_manager):
        """Test recording outcome with invalid action."""
        result = experiment_manager.record_outcome(
            experiment_name="reply_tone_test",
            variant_id="control",
            contact_id="user4",
            action="invalid_action",
        )

        assert result is False

    def test_clear_outcomes_for_experiment(self, experiment_manager):
        """Test clearing outcomes for specific experiment."""
        result = experiment_manager.clear_outcomes("reply_tone_test")

        assert result is True
        outcomes = experiment_manager.get_outcomes_for_experiment("reply_tone_test")
        assert len(outcomes) == 0

    def test_clear_all_outcomes(self, experiment_manager):
        """Test clearing all outcomes."""
        result = experiment_manager.clear_outcomes()

        assert result is True
        assert len(experiment_manager._outcomes) == 0


class TestExperimentManagerResults:
    """Tests for results calculation."""

    def test_calculate_results(self, experiment_manager):
        """Test calculating experiment results."""
        results = experiment_manager.calculate_results("reply_tone_test")

        assert results is not None
        assert results.experiment_name == "reply_tone_test"
        assert len(results.variants) == 2
        assert results.total_outcomes == 2

    def test_calculate_results_nonexistent(self, experiment_manager):
        """Test calculating results for nonexistent experiment."""
        results = experiment_manager.calculate_results("nonexistent")
        assert results is None

    def test_calculate_results_winner(self, experiment_manager):
        """Test winner determination."""
        # Add more outcomes to create clear winner
        for _ in range(10):
            experiment_manager.record_outcome(
                "reply_tone_test", "control", "user_x", UserAction.SENT_UNCHANGED
            )
        for _ in range(2):
            experiment_manager.record_outcome(
                "reply_tone_test", "treatment", "user_y", UserAction.SENT_UNCHANGED
            )

        results = experiment_manager.calculate_results("reply_tone_test")

        # Control should have higher conversion rate
        assert results.winner == "control"


class TestExperimentManagerCRUD:
    """Tests for experiment CRUD operations."""

    def test_create_experiment(self, experiment_manager):
        """Test creating a new experiment."""
        exp = experiment_manager.create_experiment(
            name="new_experiment",
            variants=[
                {"id": "a", "weight": 50, "config": {}},
                {"id": "b", "weight": 50, "config": {}},
            ],
            description="A new test",
        )

        assert exp.name == "new_experiment"
        assert len(exp.variants) == 2

        # Verify it's persisted
        loaded_exp = experiment_manager.get_experiment("new_experiment")
        assert loaded_exp is not None

    def test_update_experiment(self, experiment_manager):
        """Test updating an experiment."""
        result = experiment_manager.update_experiment("reply_tone_test", enabled=False)

        assert result is True

        exp = experiment_manager.get_experiment("reply_tone_test")
        assert exp.enabled is False

    def test_update_nonexistent_experiment(self, experiment_manager):
        """Test updating nonexistent experiment."""
        result = experiment_manager.update_experiment("nonexistent", enabled=True)
        assert result is False

    def test_delete_experiment(self, experiment_manager):
        """Test deleting an experiment."""
        result = experiment_manager.delete_experiment("reply_tone_test")

        assert result is True
        assert experiment_manager.get_experiment("reply_tone_test") is None

    def test_delete_nonexistent_experiment(self, experiment_manager):
        """Test deleting nonexistent experiment."""
        result = experiment_manager.delete_experiment("nonexistent")
        assert result is False

    def test_reload(self, experiment_manager, temp_dir, sample_experiments_yaml):
        """Test reloading experiments from disk."""
        # Modify the file on disk
        experiments_path = temp_dir / "experiments.yaml"
        sample_experiments_yaml["experiments"].append(
            {
                "name": "new_on_disk",
                "enabled": True,
                "variants": [{"id": "x", "weight": 100}],
            }
        )
        with experiments_path.open("w") as f:
            yaml.safe_dump(sample_experiments_yaml, f)

        # Reload
        experiment_manager.reload()

        # New experiment should be visible
        exp = experiment_manager.get_experiment("new_on_disk")
        assert exp is not None


# =============================================================================
# Statistical Significance Tests
# =============================================================================


class TestStatisticalSignificance:
    """Tests for statistical significance calculation."""

    def test_chi_squared_not_enough_samples(self, experiment_manager):
        """Test chi-squared with too few samples."""
        variant_a = VariantResults(variant_id="a", total_impressions=3, sent_unchanged=2)
        variant_b = VariantResults(variant_id="b", total_impressions=3, sent_unchanged=1)

        p_value = experiment_manager._chi_squared_test(variant_a, variant_b)
        assert p_value is None

    def test_chi_squared_with_sufficient_samples(self, experiment_manager):
        """Test chi-squared with sufficient samples."""
        variant_a = VariantResults(variant_id="a", total_impressions=100, sent_unchanged=60)
        variant_b = VariantResults(variant_id="b", total_impressions=100, sent_unchanged=40)

        p_value = experiment_manager._chi_squared_test(variant_a, variant_b)
        assert p_value is not None
        assert 0 <= p_value <= 1

    def test_statistical_significance_convenience_function(self):
        """Test the statistical_significance convenience function."""
        variant_a = VariantResults(variant_id="a", total_impressions=100, sent_unchanged=80)
        variant_b = VariantResults(variant_id="b", total_impressions=100, sent_unchanged=40)

        is_significant, p_value = statistical_significance(variant_a, variant_b)

        assert isinstance(is_significant, bool)
        # Large difference should be significant
        assert is_significant is True
        assert p_value is not None and p_value < 0.05

    def test_no_significance_when_similar(self):
        """Test no significance when conversion rates are similar."""
        variant_a = VariantResults(variant_id="a", total_impressions=100, sent_unchanged=50)
        variant_b = VariantResults(variant_id="b", total_impressions=100, sent_unchanged=51)

        is_significant, p_value = statistical_significance(variant_a, variant_b)

        # Similar rates should not be significant
        assert is_significant is False


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton management."""

    def test_get_experiment_manager_returns_singleton(self):
        """Test singleton pattern."""
        manager1 = get_experiment_manager()
        manager2 = get_experiment_manager()

        assert manager1 is manager2

    def test_reset_experiment_manager(self):
        """Test resetting singleton."""
        manager1 = get_experiment_manager()
        reset_experiment_manager()
        manager2 = get_experiment_manager()

        assert manager1 is not manager2
