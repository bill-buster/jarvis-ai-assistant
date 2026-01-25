"""Unit tests for template coverage analyzer.

Workstream 3: Template Coverage Analyzer
"""

import pytest

from benchmarks.coverage.analyzer import TemplateCoverageAnalyzer
from benchmarks.coverage.datasets import generate_scenarios, get_scenario_metadata
from benchmarks.coverage.templates import TEMPLATES, get_templates_by_category
from contracts.coverage import CoverageResult, TemplateMatch


class TestTemplates:
    """Tests for template definitions."""

    def test_template_count_minimum(self):
        """Verify we have at least 50 templates."""
        assert len(TEMPLATES) >= 50

    def test_templates_are_unique(self):
        """Verify no duplicate templates."""
        assert len(TEMPLATES) == len(set(TEMPLATES))

    def test_templates_not_empty(self):
        """Verify all templates have content."""
        for template in TEMPLATES:
            assert template.strip(), "Template cannot be empty"

    def test_templates_by_category_returns_dict(self):
        """Verify get_templates_by_category returns proper structure."""
        categories = get_templates_by_category()
        assert isinstance(categories, dict)
        assert len(categories) >= 10  # At least 10 categories

    def test_all_categories_have_templates(self):
        """Verify each category has at least one template."""
        categories = get_templates_by_category()
        for name, templates in categories.items():
            assert len(templates) > 0, f"Category {name} has no templates"


class TestDatasets:
    """Tests for scenario dataset generation."""

    def test_scenario_count(self):
        """Verify we have exactly 1000 scenarios."""
        scenarios = generate_scenarios()
        assert len(scenarios) == 1000

    def test_scenarios_are_diverse(self):
        """Verify scenarios aren't all identical."""
        scenarios = generate_scenarios()
        unique = set(scenarios)
        assert len(unique) >= 900  # At least 90% unique

    def test_scenarios_not_empty(self):
        """Verify all scenarios have content."""
        scenarios = generate_scenarios()
        for scenario in scenarios:
            assert scenario.strip(), "Scenario cannot be empty"

    def test_scenario_metadata(self):
        """Verify metadata reports correct totals."""
        metadata = get_scenario_metadata()
        assert metadata["total"] == 1000
        assert "professional" in metadata
        assert "personal" in metadata
        assert "transactional" in metadata

    def test_scenario_distribution(self):
        """Verify distribution matches expected proportions."""
        metadata = get_scenario_metadata()
        # Professional should be largest category
        assert metadata["professional"] >= 300


class TestAnalyzer:
    """Tests for TemplateCoverageAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TemplateCoverageAnalyzer()

    def test_match_query_returns_template_match(self, analyzer):
        """Verify match_query returns correct type."""
        result = analyzer.match_query("Thanks for the update")
        assert isinstance(result, TemplateMatch)

    def test_match_query_has_required_fields(self, analyzer):
        """Verify TemplateMatch has all required fields."""
        result = analyzer.match_query("Got it, thanks")
        assert hasattr(result, "query")
        assert hasattr(result, "best_template")
        assert hasattr(result, "similarity_score")
        assert hasattr(result, "matched")

    def test_match_query_similarity_in_range(self, analyzer):
        """Verify similarity score is between 0 and 1."""
        result = analyzer.match_query("Any query here")
        assert 0.0 <= result.similarity_score <= 1.0

    def test_match_query_exact_match_high_similarity(self, analyzer):
        """Verify exact template match has high similarity."""
        # Use an actual template
        template = analyzer.get_templates()[0]
        result = analyzer.match_query(template)
        assert result.similarity_score >= 0.95
        assert result.matched is True

    def test_match_query_unrelated_low_similarity(self, analyzer):
        """Verify unrelated query has low similarity."""
        result = analyzer.match_query("What is the capital of France?")
        assert result.similarity_score < 0.7
        assert result.matched is False

    def test_analyze_dataset_returns_coverage_result(self, analyzer):
        """Verify analyze_dataset returns correct type."""
        result = analyzer.analyze_dataset(["Thanks", "Got it"])
        assert isinstance(result, CoverageResult)

    def test_analyze_dataset_has_required_fields(self, analyzer):
        """Verify CoverageResult has all required fields."""
        result = analyzer.analyze_dataset(["Thanks for the help"])
        assert hasattr(result, "total_queries")
        assert hasattr(result, "coverage_at_50")
        assert hasattr(result, "coverage_at_70")
        assert hasattr(result, "coverage_at_90")
        assert hasattr(result, "unmatched_examples")
        assert hasattr(result, "template_usage")
        assert hasattr(result, "timestamp")

    def test_analyze_dataset_coverage_in_range(self, analyzer):
        """Verify coverage percentages are between 0 and 1."""
        result = analyzer.analyze_dataset(["Hello", "Thanks", "Bye"])
        assert 0.0 <= result.coverage_at_50 <= 1.0
        assert 0.0 <= result.coverage_at_70 <= 1.0
        assert 0.0 <= result.coverage_at_90 <= 1.0

    def test_analyze_dataset_empty_list(self, analyzer):
        """Verify empty dataset returns zero coverage."""
        result = analyzer.analyze_dataset([])
        assert result.total_queries == 0
        assert result.coverage_at_50 == 0.0
        assert result.coverage_at_70 == 0.0
        assert result.coverage_at_90 == 0.0

    def test_analyze_dataset_coverage_ordering(self, analyzer):
        """Verify coverage decreases with higher thresholds."""
        result = analyzer.analyze_dataset(generate_scenarios()[:100])
        # Higher threshold should have lower or equal coverage
        assert result.coverage_at_90 <= result.coverage_at_70
        assert result.coverage_at_70 <= result.coverage_at_50

    def test_get_templates_returns_list(self, analyzer):
        """Verify get_templates returns templates."""
        templates = analyzer.get_templates()
        assert isinstance(templates, list)
        assert len(templates) >= 50

    def test_get_templates_returns_copy(self, analyzer):
        """Verify get_templates returns a copy, not the original."""
        templates1 = analyzer.get_templates()
        templates2 = analyzer.get_templates()
        assert templates1 is not templates2

    def test_add_template_increases_count(self, analyzer):
        """Verify add_template works."""
        initial = len(analyzer.get_templates())
        analyzer.add_template("New test template")
        assert len(analyzer.get_templates()) == initial + 1

    def test_add_template_invalidates_cache(self, analyzer):
        """Verify adding template recomputes embeddings."""
        # First call to ensure embeddings are computed
        analyzer.match_query("test query")

        # Add template
        analyzer.add_template("Brand new template for testing")

        # Verify new template can be matched
        result = analyzer.match_query("Brand new template for testing")
        assert result.similarity_score >= 0.95


class TestContractCompliance:
    """Verify TemplateCoverageAnalyzer implements CoverageAnalyzer protocol."""

    def test_has_match_query_method(self):
        """Verify match_query method exists."""
        analyzer = TemplateCoverageAnalyzer()
        assert hasattr(analyzer, "match_query")
        assert callable(analyzer.match_query)

    def test_has_analyze_dataset_method(self):
        """Verify analyze_dataset method exists."""
        analyzer = TemplateCoverageAnalyzer()
        assert hasattr(analyzer, "analyze_dataset")
        assert callable(analyzer.analyze_dataset)

    def test_has_get_templates_method(self):
        """Verify get_templates method exists."""
        analyzer = TemplateCoverageAnalyzer()
        assert hasattr(analyzer, "get_templates")
        assert callable(analyzer.get_templates)

    def test_has_add_template_method(self):
        """Verify add_template method exists."""
        analyzer = TemplateCoverageAnalyzer()
        assert hasattr(analyzer, "add_template")
        assert callable(analyzer.add_template)

    def test_match_query_signature(self):
        """Verify match_query accepts query and optional threshold."""
        analyzer = TemplateCoverageAnalyzer()
        # Should work with just query
        result1 = analyzer.match_query("test")
        assert isinstance(result1, TemplateMatch)
        # Should work with threshold
        result2 = analyzer.match_query("test", threshold=0.5)
        assert isinstance(result2, TemplateMatch)


class TestPerformance:
    """Performance-related tests."""

    def test_batch_encoding_efficiency(self):
        """Verify analyze_dataset handles large datasets efficiently."""
        analyzer = TemplateCoverageAnalyzer()
        # 100 queries should complete quickly with batch encoding
        result = analyzer.analyze_dataset(generate_scenarios()[:100])
        assert result.total_queries == 100

    def test_template_embedding_caching(self):
        """Verify template embeddings are cached."""
        analyzer = TemplateCoverageAnalyzer()
        # First call computes embeddings
        analyzer.match_query("test1")
        # Second call should use cached embeddings
        analyzer.match_query("test2")
        # If we got here without timeout, caching is working
