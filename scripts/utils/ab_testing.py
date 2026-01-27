"""
A/B Testing Framework for Template Evaluation

Allows controlled rollout and measurement of template changes.
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ABTestConfig:
    """Configuration for an A/B test."""

    def __init__(
        self,
        test_name: str,
        variant_a: str,
        variant_b: str,
        traffic_split: float = 0.5,
        enabled: bool = True
    ):
        """Initialize A/B test config.

        Args:
            test_name: Unique test identifier
            variant_a: Name of control variant
            variant_b: Name of treatment variant
            traffic_split: Fraction of traffic for variant_b (0-1)
            enabled: Whether test is active
        """
        self.test_name = test_name
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.traffic_split = traffic_split
        self.enabled = enabled


class ABTestAssignment:
    """Assigns users to A/B test variants."""

    def __init__(self, config: ABTestConfig):
        """Initialize assignment logic.

        Args:
            config: A/B test configuration
        """
        self.config = config

    def get_variant(self, user_id: str) -> str:
        """Assign user to variant.

        Uses deterministic hashing for consistent assignment.

        Args:
            user_id: Unique user identifier

        Returns:
            Variant name (variant_a or variant_b)
        """
        if not self.config.enabled:
            return self.config.variant_a  # Default to control

        # Hash user_id + test_name for deterministic assignment
        hash_input = f"{user_id}_{self.config.test_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to 0-1
        assignment_value = (hash_value % 10000) / 10000.0

        if assignment_value < self.config.traffic_split:
            return self.config.variant_b
        else:
            return self.config.variant_a


class ABTestMetrics:
    """Collects and analyzes A/B test metrics."""

    def __init__(self, results_dir: Path):
        """Initialize metrics collector.

        Args:
            results_dir: Directory to store results
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = defaultdict(lambda: {
            "variant_a": defaultdict(list),
            "variant_b": defaultdict(list)
        })

    def log_event(
        self,
        test_name: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """Log a metric event.

        Args:
            test_name: Test identifier
            variant: Variant name
            metric_name: Metric name (e.g., "template_hit", "user_edited")
            value: Metric value
        """
        self.metrics[test_name][variant][metric_name].append({
            "value": value,
            "timestamp": time.time()
        })

    def calculate_statistics(self, test_name: str) -> dict[str, Any]:
        """Calculate statistics for a test.

        Args:
            test_name: Test identifier

        Returns:
            Dictionary with statistics for each variant
        """
        if test_name not in self.metrics:
            return {}

        import numpy as np

        stats = {}

        for variant in ["variant_a", "variant_b"]:
            variant_metrics = {}

            for metric_name, events in self.metrics[test_name][variant].items():
                values = [e["value"] for e in events]

                if values:
                    variant_metrics[metric_name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }

            stats[variant] = variant_metrics

        return stats

    def compare_variants(
        self,
        test_name: str,
        metric_name: str,
        min_samples: int = 30
    ) -> dict[str, Any]:
        """Compare variants on a specific metric.

        Args:
            test_name: Test identifier
            metric_name: Metric to compare
            min_samples: Minimum samples needed for comparison

        Returns:
            Comparison results with p-value and effect size
        """
        if test_name not in self.metrics:
            return {"error": "Test not found"}

        variant_a_events = self.metrics[test_name]["variant_a"].get(metric_name, [])
        variant_b_events = self.metrics[test_name]["variant_b"].get(metric_name, [])

        variant_a_values = [e["value"] for e in variant_a_events]
        variant_b_values = [e["value"] for e in variant_b_events]

        if len(variant_a_values) < min_samples or len(variant_b_values) < min_samples:
            return {
                "error": "Insufficient samples",
                "variant_a_count": len(variant_a_values),
                "variant_b_count": len(variant_b_values),
                "min_samples_required": min_samples
            }

        try:
            from scipy import stats as scipy_stats
            import numpy as np

            # T-test
            t_stat, p_value = scipy_stats.ttest_ind(variant_a_values, variant_b_values)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(variant_a_values) + np.var(variant_b_values)) / 2
            )
            effect_size = (np.mean(variant_b_values) - np.mean(variant_a_values)) / pooled_std if pooled_std > 0 else 0

            return {
                "metric": metric_name,
                "variant_a_mean": np.mean(variant_a_values),
                "variant_b_mean": np.mean(variant_b_values),
                "variant_a_count": len(variant_a_values),
                "variant_b_count": len(variant_b_values),
                "p_value": p_value,
                "effect_size": effect_size,
                "significant": p_value < 0.05,
                "winner": "variant_b" if p_value < 0.05 and effect_size > 0 else "variant_a" if p_value < 0.05 else "no_clear_winner"
            }

        except ImportError:
            logger.warning("scipy not available for statistical testing")
            return {"error": "scipy required for statistical comparison"}

    def save_results(self, test_name: str):
        """Save test results to disk.

        Args:
            test_name: Test identifier
        """
        results_file = self.results_dir / f"{test_name}_results.json"

        stats = self.calculate_statistics(test_name)

        with open(results_file, 'w') as f:
            json.dump({
                "test_name": test_name,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "raw_metrics": {
                    k: {vk: len(vv) for vk, vv in v.items()}
                    for k, v in self.metrics[test_name].items()
                }
            }, f, indent=2)

        logger.info("Saved A/B test results to: %s", results_file)


# Example usage:
"""
# Setup
config = ABTestConfig(
    test_name="template_v2_rollout",
    variant_a="baseline_templates",
    variant_b="production_templates",
    traffic_split=0.1,  # 10% to new templates
    enabled=True
)

assignment = ABTestAssignment(config)
metrics = ABTestMetrics(Path("results/ab_tests"))

# In your reply generation code:
variant = assignment.get_variant(user_id)

if variant == "variant_b":
    # Use new production templates
    template_match = production_matcher.match(query)
else:
    # Use baseline templates
    template_match = baseline_matcher.match(query)

# Log metrics
metrics.log_event(
    test_name="template_v2_rollout",
    variant=variant,
    metric_name="template_hit",
    value=1.0 if template_match else 0.0
)

metrics.log_event(
    test_name="template_v2_rollout",
    variant=variant,
    metric_name="user_accepted",
    value=1.0 if user_accepted_reply else 0.0
)

# Analyze results
comparison = metrics.compare_variants("template_v2_rollout", "user_accepted")
print(f"Winner: {comparison['winner']} (p={comparison['p_value']:.4f})")
"""
