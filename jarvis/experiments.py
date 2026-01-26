"""A/B Testing Infrastructure for JARVIS prompt experimentation.

This module provides experiment configuration, variant selection, and results analysis
for testing different prompt versions (few-shot examples, system prompts, temperatures).

Experiment configuration is stored in ~/.jarvis/experiments.yaml:
    experiments:
      - name: reply_tone_test
        enabled: true
        variants:
          - id: control
            weight: 50
            config:
              system_prompt: "You are a helpful assistant."
              temperature: 0.7
          - id: treatment
            weight: 50
            config:
              system_prompt: "You are a friendly, casual assistant."
              temperature: 0.8

Usage:
    from jarvis.experiments import get_experiment_manager

    manager = get_experiment_manager()
    variant = manager.get_variant("reply_tone_test", contact_id="john@example.com")
    # Use variant.config to configure prompt generation

    # Record outcome after user action
    manager.record_outcome("reply_tone_test", variant.id, contact_id, "sent_unchanged")

    # Analyze results
    results = manager.calculate_results("reply_tone_test")
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

EXPERIMENTS_PATH = Path.home() / ".jarvis" / "experiments.yaml"
OUTCOMES_PATH = Path.home() / ".jarvis" / "experiment_outcomes.json"


class UserAction(str, Enum):
    """Possible user actions for experiment outcomes."""

    SENT_UNCHANGED = "sent_unchanged"  # User sent the suggestion as-is
    SENT_EDITED = "sent_edited"  # User edited before sending
    DISMISSED = "dismissed"  # User dismissed/ignored the suggestion
    REGENERATED = "regenerated"  # User requested regeneration


@dataclass
class VariantConfig:
    """Configuration for a single experiment variant.

    Attributes:
        id: Unique identifier for this variant.
        weight: Percentage weight for allocation (0-100).
        config: Variant-specific configuration (prompts, temperatures, etc.).
    """

    id: str
    weight: int
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """An A/B test experiment definition.

    Attributes:
        name: Unique experiment name.
        enabled: Whether the experiment is currently active.
        variants: List of variant configurations.
        description: Optional description of the experiment.
        created_at: When the experiment was created.
    """

    name: str
    enabled: bool
    variants: list[VariantConfig]
    description: str = ""
    created_at: str = ""

    def get_variant_by_id(self, variant_id: str) -> VariantConfig | None:
        """Get a variant by its ID."""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None


@dataclass
class ExperimentOutcome:
    """Records a single experiment outcome.

    Attributes:
        experiment_name: Name of the experiment.
        variant_id: ID of the variant that was shown.
        contact_id: Identifier for the contact (for deterministic assignment).
        action: The user's action.
        timestamp: When the outcome was recorded.
    """

    experiment_name: str
    variant_id: str
    contact_id: str
    action: UserAction
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "variant_id": self.variant_id,
            "contact_id": self.contact_id,
            "action": self.action.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentOutcome:
        """Create from dictionary."""
        return cls(
            experiment_name=data["experiment_name"],
            variant_id=data["variant_id"],
            contact_id=data["contact_id"],
            action=UserAction(data["action"]),
            timestamp=data["timestamp"],
        )


@dataclass
class VariantResults:
    """Results for a single variant.

    Attributes:
        variant_id: The variant identifier.
        total_impressions: Total times this variant was shown.
        sent_unchanged: Times user sent without editing.
        sent_edited: Times user edited before sending.
        dismissed: Times user dismissed the suggestion.
        regenerated: Times user requested regeneration.
        conversion_rate: Percentage of sent_unchanged (primary success metric).
    """

    variant_id: str
    total_impressions: int = 0
    sent_unchanged: int = 0
    sent_edited: int = 0
    dismissed: int = 0
    regenerated: int = 0

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate (sent_unchanged / total)."""
        if self.total_impressions == 0:
            return 0.0
        return (self.sent_unchanged / self.total_impressions) * 100


@dataclass
class ExperimentResults:
    """Full results for an experiment.

    Attributes:
        experiment_name: Name of the experiment.
        variants: Results for each variant.
        total_outcomes: Total outcomes recorded.
        winner: Variant ID of the likely winner (highest conversion).
        is_significant: Whether results are statistically significant.
        p_value: P-value from chi-squared test (if applicable).
    """

    experiment_name: str
    variants: list[VariantResults]
    total_outcomes: int = 0
    winner: str | None = None
    is_significant: bool = False
    p_value: float | None = None


class ExperimentManager:
    """Manages A/B testing experiments for prompt variants.

    Thread-safe singleton that handles:
    - Loading experiment configurations from YAML
    - Deterministic variant assignment per contact
    - Recording experiment outcomes
    - Calculating results with statistical significance
    """

    def __init__(self, experiments_path: Path | None = None, outcomes_path: Path | None = None):
        """Initialize the experiment manager.

        Args:
            experiments_path: Path to experiments YAML file.
            outcomes_path: Path to outcomes JSON file.
        """
        self._experiments_path = experiments_path or EXPERIMENTS_PATH
        self._outcomes_path = outcomes_path or OUTCOMES_PATH
        self._experiments: dict[str, Experiment] = {}
        self._outcomes: list[ExperimentOutcome] = []
        self._lock = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Ensure experiments and outcomes are loaded."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load_experiments()
                    self._load_outcomes()
                    self._loaded = True

    def _load_experiments(self) -> None:
        """Load experiments from YAML file."""
        if not self._experiments_path.exists():
            logger.debug(f"Experiments file not found at {self._experiments_path}")
            self._experiments = {}
            return

        try:
            with self._experiments_path.open() as f:
                data = yaml.safe_load(f) or {}

            experiments_data = data.get("experiments", [])
            self._experiments = {}

            for exp_data in experiments_data:
                variants = [
                    VariantConfig(
                        id=v.get("id", ""),
                        weight=v.get("weight", 50),
                        config=v.get("config", {}),
                    )
                    for v in exp_data.get("variants", [])
                ]

                experiment = Experiment(
                    name=exp_data.get("name", ""),
                    enabled=exp_data.get("enabled", True),
                    variants=variants,
                    description=exp_data.get("description", ""),
                    created_at=exp_data.get("created_at", ""),
                )

                if experiment.name:
                    self._experiments[experiment.name] = experiment

            logger.info(f"Loaded {len(self._experiments)} experiments")

        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse experiments YAML: {e}")
            self._experiments = {}
        except OSError as e:
            logger.warning(f"Failed to read experiments file: {e}")
            self._experiments = {}

    def _load_outcomes(self) -> None:
        """Load outcomes from JSON file."""
        if not self._outcomes_path.exists():
            logger.debug(f"Outcomes file not found at {self._outcomes_path}")
            self._outcomes = []
            return

        try:
            with self._outcomes_path.open() as f:
                data = json.load(f)

            self._outcomes = [ExperimentOutcome.from_dict(o) for o in data.get("outcomes", [])]
            logger.info(f"Loaded {len(self._outcomes)} experiment outcomes")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse outcomes JSON: {e}")
            self._outcomes = []
        except OSError as e:
            logger.warning(f"Failed to read outcomes file: {e}")
            self._outcomes = []

    def _save_outcomes(self) -> bool:
        """Save outcomes to JSON file.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            self._outcomes_path.parent.mkdir(parents=True, exist_ok=True)

            with self._outcomes_path.open("w") as f:
                json.dump(
                    {"outcomes": [o.to_dict() for o in self._outcomes]},
                    f,
                    indent=2,
                )
            return True
        except OSError as e:
            logger.error(f"Failed to save outcomes: {e}")
            return False

    def _save_experiments(self) -> bool:
        """Save experiments to YAML file.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            self._experiments_path.parent.mkdir(parents=True, exist_ok=True)

            experiments_data = []
            for exp in self._experiments.values():
                exp_dict = {
                    "name": exp.name,
                    "enabled": exp.enabled,
                    "description": exp.description,
                    "created_at": exp.created_at,
                    "variants": [
                        {"id": v.id, "weight": v.weight, "config": v.config}
                        for v in exp.variants
                    ],
                }
                experiments_data.append(exp_dict)

            with self._experiments_path.open("w") as f:
                yaml.safe_dump({"experiments": experiments_data}, f, default_flow_style=False)
            return True
        except OSError as e:
            logger.error(f"Failed to save experiments: {e}")
            return False

    def get_experiments(self) -> list[Experiment]:
        """Get all configured experiments.

        Returns:
            List of all experiments.
        """
        self._ensure_loaded()
        return list(self._experiments.values())

    def get_active_experiments(self) -> list[Experiment]:
        """Get all active (enabled) experiments.

        Returns:
            List of enabled experiments.
        """
        self._ensure_loaded()
        return [exp for exp in self._experiments.values() if exp.enabled]

    def get_experiment(self, name: str) -> Experiment | None:
        """Get a specific experiment by name.

        Args:
            name: Experiment name.

        Returns:
            The experiment or None if not found.
        """
        self._ensure_loaded()
        return self._experiments.get(name)

    def get_variant(self, experiment_name: str, contact_id: str) -> VariantConfig | None:
        """Get the variant for a contact (deterministic assignment).

        Uses consistent hashing to ensure the same contact always gets
        the same variant for a given experiment.

        Args:
            experiment_name: Name of the experiment.
            contact_id: Identifier for the contact (phone, email, etc.).

        Returns:
            The assigned variant configuration, or None if experiment not found.
        """
        self._ensure_loaded()

        experiment = self._experiments.get(experiment_name)
        if not experiment or not experiment.enabled:
            return None

        if not experiment.variants:
            return None

        # Create deterministic hash from experiment name and contact ID
        hash_input = f"{experiment_name}:{contact_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)

        # Normalize to 0-100 range
        bucket = hash_value % 100

        # Assign based on cumulative weights
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant

        # Fallback to last variant (shouldn't happen with proper weights)
        return experiment.variants[-1]

    def record_outcome(
        self,
        experiment_name: str,
        variant_id: str,
        contact_id: str,
        action: UserAction | str,
    ) -> bool:
        """Record an experiment outcome.

        Args:
            experiment_name: Name of the experiment.
            variant_id: ID of the variant that was shown.
            contact_id: Identifier for the contact.
            action: The user's action (UserAction enum or string).

        Returns:
            True if recorded successfully, False otherwise.
        """
        self._ensure_loaded()

        # Convert string to enum if needed
        if isinstance(action, str):
            try:
                action = UserAction(action)
            except ValueError:
                logger.warning(f"Invalid action: {action}")
                return False

        outcome = ExperimentOutcome(
            experiment_name=experiment_name,
            variant_id=variant_id,
            contact_id=contact_id,
            action=action,
            timestamp=datetime.utcnow().isoformat(),
        )

        with self._lock:
            self._outcomes.append(outcome)
            return self._save_outcomes()

    def calculate_results(self, experiment_name: str) -> ExperimentResults | None:
        """Calculate results for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            ExperimentResults with win rates per variant, or None if not found.
        """
        self._ensure_loaded()

        experiment = self._experiments.get(experiment_name)
        if not experiment:
            return None

        # Initialize results for each variant
        variant_results: dict[str, VariantResults] = {}
        for variant in experiment.variants:
            variant_results[variant.id] = VariantResults(variant_id=variant.id)

        # Count outcomes
        for outcome in self._outcomes:
            if outcome.experiment_name != experiment_name:
                continue

            if outcome.variant_id not in variant_results:
                continue

            results = variant_results[outcome.variant_id]
            results.total_impressions += 1

            if outcome.action == UserAction.SENT_UNCHANGED:
                results.sent_unchanged += 1
            elif outcome.action == UserAction.SENT_EDITED:
                results.sent_edited += 1
            elif outcome.action == UserAction.DISMISSED:
                results.dismissed += 1
            elif outcome.action == UserAction.REGENERATED:
                results.regenerated += 1

        # Determine winner (highest conversion rate)
        results_list = list(variant_results.values())
        total_outcomes = sum(r.total_impressions for r in results_list)

        winner = None
        highest_rate = -1.0
        for r in results_list:
            if r.conversion_rate > highest_rate:
                highest_rate = r.conversion_rate
                winner = r.variant_id

        # Calculate statistical significance if we have two variants
        is_significant = False
        p_value = None

        if len(results_list) == 2 and all(r.total_impressions > 0 for r in results_list):
            p_value = self._chi_squared_test(results_list[0], results_list[1])
            is_significant = p_value is not None and p_value < 0.05

        return ExperimentResults(
            experiment_name=experiment_name,
            variants=results_list,
            total_outcomes=total_outcomes,
            winner=winner,
            is_significant=is_significant,
            p_value=p_value,
        )

    def _chi_squared_test(
        self, variant_a: VariantResults, variant_b: VariantResults
    ) -> float | None:
        """Perform chi-squared test for two variants.

        Tests whether conversion rates are significantly different.

        Args:
            variant_a: Results for first variant.
            variant_b: Results for second variant.

        Returns:
            P-value or None if test cannot be performed.
        """
        # Need at least 5 samples in each cell for chi-squared
        min_samples = 5
        if variant_a.total_impressions < min_samples or variant_b.total_impressions < min_samples:
            return None

        # Observed values (conversions, non-conversions)
        o_a_conv = variant_a.sent_unchanged
        o_a_non = variant_a.total_impressions - variant_a.sent_unchanged
        o_b_conv = variant_b.sent_unchanged
        o_b_non = variant_b.total_impressions - variant_b.sent_unchanged

        total = variant_a.total_impressions + variant_b.total_impressions
        total_conv = o_a_conv + o_b_conv
        total_non = o_a_non + o_b_non

        if total_conv == 0 or total_non == 0:
            return None

        # Expected values
        e_a_conv = (variant_a.total_impressions * total_conv) / total
        e_a_non = (variant_a.total_impressions * total_non) / total
        e_b_conv = (variant_b.total_impressions * total_conv) / total
        e_b_non = (variant_b.total_impressions * total_non) / total

        # Avoid division by zero
        if any(e <= 0 for e in [e_a_conv, e_a_non, e_b_conv, e_b_non]):
            return None

        # Chi-squared statistic
        chi_sq = (
            ((o_a_conv - e_a_conv) ** 2) / e_a_conv
            + ((o_a_non - e_a_non) ** 2) / e_a_non
            + ((o_b_conv - e_b_conv) ** 2) / e_b_conv
            + ((o_b_non - e_b_non) ** 2) / e_b_non
        )

        # Convert to p-value using chi-squared distribution (1 df)
        # Using simplified approximation for p-value calculation
        p_value = self._chi_squared_cdf(chi_sq, df=1)
        return 1 - p_value

    def _chi_squared_cdf(self, x: float, df: int) -> float:
        """Approximate chi-squared CDF.

        Args:
            x: Chi-squared statistic.
            df: Degrees of freedom.

        Returns:
            Approximate CDF value.
        """
        # Using Wilson-Hilferty approximation
        if x <= 0:
            return 0.0

        if df == 1:
            # For df=1, use error function approximation
            import math

            z = math.sqrt(x)
            # Standard normal CDF approximation
            return self._standard_normal_cdf(z) - self._standard_normal_cdf(-z)

        # General approximation
        import math

        a = df / 2.0
        return self._incomplete_gamma(a, x / 2.0) / math.gamma(a)

    def _standard_normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        import math

        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _incomplete_gamma(self, a: float, x: float) -> float:
        """Approximate lower incomplete gamma function."""
        import math

        # Series expansion for small x
        if x < a + 1:
            result = 0.0
            term = 1.0 / a
            for n in range(1, 100):
                term *= x / (a + n)
                result += term
                if abs(term) < 1e-10:
                    break
            return (result + 1.0 / a) * math.exp(-x + a * math.log(x))

        # Continued fraction for large x
        return math.gamma(a) - self._incomplete_gamma_upper(a, x)

    def _incomplete_gamma_upper(self, a: float, x: float) -> float:
        """Approximate upper incomplete gamma function."""
        import math

        # Continued fraction expansion
        f = 1.0 + x - a
        c = 1.0 / 1e-30
        d = 1.0 / f
        h = d

        for i in range(1, 100):
            an = -i * (i - a)
            bn = f + 2 * i
            d = bn + an * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = bn + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-10:
                break

        return math.exp(-x + a * math.log(x)) * h

    def create_experiment(
        self,
        name: str,
        variants: list[dict[str, Any]],
        description: str = "",
        enabled: bool = True,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Unique experiment name.
            variants: List of variant configurations.
            description: Optional description.
            enabled: Whether to enable the experiment.

        Returns:
            The created experiment.
        """
        self._ensure_loaded()

        variant_configs = [
            VariantConfig(
                id=v.get("id", f"variant_{i}"),
                weight=v.get("weight", 50),
                config=v.get("config", {}),
            )
            for i, v in enumerate(variants)
        ]

        experiment = Experiment(
            name=name,
            enabled=enabled,
            variants=variant_configs,
            description=description,
            created_at=datetime.utcnow().isoformat(),
        )

        with self._lock:
            self._experiments[name] = experiment
            self._save_experiments()

        return experiment

    def update_experiment(self, name: str, enabled: bool | None = None) -> bool:
        """Update an experiment.

        Args:
            name: Experiment name.
            enabled: New enabled state (optional).

        Returns:
            True if updated successfully.
        """
        self._ensure_loaded()

        experiment = self._experiments.get(name)
        if not experiment:
            return False

        with self._lock:
            if enabled is not None:
                experiment.enabled = enabled
            return self._save_experiments()

    def delete_experiment(self, name: str) -> bool:
        """Delete an experiment.

        Args:
            name: Experiment name.

        Returns:
            True if deleted successfully.
        """
        self._ensure_loaded()

        if name not in self._experiments:
            return False

        with self._lock:
            del self._experiments[name]
            return self._save_experiments()

    def get_outcomes_for_experiment(self, experiment_name: str) -> list[ExperimentOutcome]:
        """Get all outcomes for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            List of outcomes for the experiment.
        """
        self._ensure_loaded()
        return [o for o in self._outcomes if o.experiment_name == experiment_name]

    def clear_outcomes(self, experiment_name: str | None = None) -> bool:
        """Clear outcomes for an experiment or all experiments.

        Args:
            experiment_name: Specific experiment to clear, or None for all.

        Returns:
            True if cleared successfully.
        """
        with self._lock:
            if experiment_name:
                self._outcomes = [
                    o for o in self._outcomes if o.experiment_name != experiment_name
                ]
            else:
                self._outcomes = []
            return self._save_outcomes()

    def reload(self) -> None:
        """Reload experiments and outcomes from disk."""
        with self._lock:
            self._loaded = False
        self._ensure_loaded()


# Module-level singleton with thread safety
_manager: ExperimentManager | None = None
_manager_lock = threading.Lock()


def get_experiment_manager() -> ExperimentManager:
    """Get singleton experiment manager instance.

    Returns:
        Shared ExperimentManager instance.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ExperimentManager()
    return _manager


def reset_experiment_manager() -> None:
    """Reset singleton experiment manager for testing."""
    global _manager
    with _manager_lock:
        _manager = None


# Convenience functions for common operations


def get_prompt_variant(experiment_name: str, contact_id: str) -> VariantConfig | None:
    """Get the prompt variant for a contact.

    Convenience wrapper around ExperimentManager.get_variant().

    Args:
        experiment_name: Name of the experiment.
        contact_id: Identifier for the contact.

    Returns:
        The assigned variant configuration, or None if not found.
    """
    return get_experiment_manager().get_variant(experiment_name, contact_id)


def calculate_experiment_results(experiment_name: str) -> ExperimentResults | None:
    """Calculate results for an experiment.

    Convenience wrapper around ExperimentManager.calculate_results().

    Args:
        experiment_name: Name of the experiment.

    Returns:
        ExperimentResults or None if not found.
    """
    return get_experiment_manager().calculate_results(experiment_name)


def statistical_significance(
    variant_a: VariantResults, variant_b: VariantResults
) -> tuple[bool, float | None]:
    """Test statistical significance between two variants.

    Uses chi-squared test to determine if conversion rates differ significantly.

    Args:
        variant_a: Results for first variant.
        variant_b: Results for second variant.

    Returns:
        Tuple of (is_significant, p_value).
    """
    manager = get_experiment_manager()
    p_value = manager._chi_squared_test(variant_a, variant_b)
    is_significant = p_value is not None and p_value < 0.05
    return is_significant, p_value


# Export all public symbols
__all__ = [
    # Enums
    "UserAction",
    # Data classes
    "VariantConfig",
    "Experiment",
    "ExperimentOutcome",
    "VariantResults",
    "ExperimentResults",
    # Manager class
    "ExperimentManager",
    # Singleton access
    "get_experiment_manager",
    "reset_experiment_manager",
    # Convenience functions
    "get_prompt_variant",
    "calculate_experiment_results",
    "statistical_significance",
    # Paths
    "EXPERIMENTS_PATH",
    "OUTCOMES_PATH",
]
