"""Template coverage analysis interfaces.

Workstream 3 implements against these contracts.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TemplateMatch:
    """Result of matching a query to templates."""

    query: str
    best_template: str | None
    similarity_score: float
    matched: bool  # True if score >= threshold


@dataclass
class CoverageResult:
    """Aggregate coverage analysis results."""

    total_queries: int
    coverage_at_50: float  # % matching at 0.5 similarity
    coverage_at_70: float  # % matching at 0.7 similarity
    coverage_at_90: float  # % matching at 0.9 similarity
    unmatched_examples: list[str]  # Sample queries that didn't match
    template_usage: dict[str, int]  # How often each template matched
    timestamp: str


class CoverageAnalyzer(Protocol):
    """Interface for template coverage analysis (Workstream 3)."""

    def match_query(self, query: str, threshold: float = 0.7) -> TemplateMatch:
        """Find best matching template for a query."""
        ...

    def analyze_dataset(self, queries: list[str]) -> CoverageResult:
        """Analyze coverage across a dataset of queries."""
        ...

    def get_templates(self) -> list[str]:
        """Return all available templates."""
        ...

    def add_template(self, template: str) -> None:
        """Add a new template to the matcher."""
        ...
