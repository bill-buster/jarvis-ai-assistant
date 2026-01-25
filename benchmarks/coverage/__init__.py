"""Template coverage analysis benchmark (Workstream 3).

Provides semantic similarity-based template matching for email replies.
"""

from benchmarks.coverage.analyzer import TemplateCoverageAnalyzer
from benchmarks.coverage.datasets import generate_scenarios, get_scenario_metadata
from benchmarks.coverage.templates import (
    DEFAULT_TEMPLATES,
    TEMPLATES,
    get_templates_by_category,
)

__all__ = [
    "DEFAULT_TEMPLATES",
    "TEMPLATES",
    "TemplateCoverageAnalyzer",
    "generate_scenarios",
    "get_scenario_metadata",
    "get_templates_by_category",
]
