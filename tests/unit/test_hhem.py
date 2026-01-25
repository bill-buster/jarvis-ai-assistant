"""Unit tests for HHEM hallucination evaluator.

Workstream 2: HHEM Hallucination Benchmark

Tests use a mock HHEM model to avoid downloading the real model during CI.
"""

import pytest

from benchmarks.hallucination.datasets import (
    EmailTestCase,
    generate_grounded_pairs,
    generate_hallucinated_pairs,
    generate_mixed_dataset,
    generate_test_cases,
    get_dataset_metadata,
)
from benchmarks.hallucination.hhem import (
    HHEMEvaluator,
    HHEMModelProtocol,
    get_evaluator,
)
from contracts.hallucination import HHEMBenchmarkResult, HHEMResult


class MockHHEMModel:
    """Mock HHEM model for testing without downloading the real model.

    Implements HHEMModelProtocol.

    Returns predictable scores based on text length heuristics:
    - Longer summaries relative to source = more likely hallucinated (lower score)
    - Shorter summaries = more likely grounded (higher score)
    """

    def __init__(self, fixed_scores: list[float] | None = None) -> None:
        """Initialize mock model.

        Args:
            fixed_scores: Optional list of scores to return in order.
                         If provided, returns these instead of computed scores.
        """
        self._fixed_scores = fixed_scores
        self._call_count = 0

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Return mock hallucination scores.

        Args:
            pairs: List of [source, summary] pairs

        Returns:
            List of scores from 0.0 to 1.0
        """
        if self._fixed_scores is not None:
            start = self._call_count
            end = start + len(pairs)
            self._call_count = end
            # Cycle through fixed scores if we run out
            result = []
            for i in range(len(pairs)):
                idx = (start + i) % len(self._fixed_scores)
                result.append(self._fixed_scores[idx])
            return result

        # Heuristic scoring for testing
        scores = []
        for source, summary in pairs:
            # Simple heuristic: shorter summary relative to source = more grounded
            source_len = len(source)
            summary_len = len(summary)
            ratio = summary_len / max(source_len, 1)

            # Score between 0.3 and 0.9 based on ratio
            # Lower ratio (shorter summary) = higher score
            if ratio < 0.3:
                score = 0.85 + (0.3 - ratio) * 0.5  # Cap at ~1.0
            elif ratio > 0.8:
                score = 0.3 - (ratio - 0.8) * 0.5  # Can go below 0.3
            else:
                score = 0.85 - (ratio - 0.3) * (0.55 / 0.5)

            scores.append(max(0.0, min(1.0, score)))

        return scores


class TestMockHHEMModel:
    """Tests for the mock HHEM model itself."""

    def test_mock_implements_protocol(self):
        """Verify MockHHEMModel implements HHEMModelProtocol."""
        model = MockHHEMModel()
        assert isinstance(model, HHEMModelProtocol)

    def test_mock_returns_list_of_floats(self):
        """Verify mock returns correct type."""
        model = MockHHEMModel()
        scores = model.predict([["source", "summary"]])
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_mock_returns_correct_count(self):
        """Verify mock returns one score per pair."""
        model = MockHHEMModel()
        pairs = [["s1", "m1"], ["s2", "m2"], ["s3", "m3"]]
        scores = model.predict(pairs)
        assert len(scores) == len(pairs)

    def test_mock_scores_in_range(self):
        """Verify mock scores are between 0 and 1."""
        model = MockHHEMModel()
        pairs = [["short", "very long summary text here"], ["x" * 1000, "short"]]
        scores = model.predict(pairs)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_mock_with_fixed_scores(self):
        """Verify fixed scores work correctly."""
        model = MockHHEMModel(fixed_scores=[0.8, 0.3, 0.5])
        scores = model.predict([["a", "b"], ["c", "d"]])
        assert scores == [0.8, 0.3]

    def test_mock_empty_pairs(self):
        """Verify mock handles empty input."""
        model = MockHHEMModel()
        scores = model.predict([])
        assert scores == []


class TestHHEMEvaluator:
    """Tests for HHEMEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> HHEMEvaluator:
        """Create evaluator with mock model."""
        return HHEMEvaluator(model=MockHHEMModel())

    @pytest.fixture
    def fixed_evaluator(self) -> HHEMEvaluator:
        """Create evaluator with fixed scores for deterministic testing."""
        return HHEMEvaluator(model=MockHHEMModel(fixed_scores=[0.9, 0.1, 0.5, 0.7]))

    def test_evaluate_single_returns_float(self, evaluator: HHEMEvaluator):
        """Verify evaluate_single returns a float."""
        score = evaluator.evaluate_single("source text", "summary")
        assert isinstance(score, float)

    def test_evaluate_single_score_in_range(self, evaluator: HHEMEvaluator):
        """Verify single evaluation score is between 0 and 1."""
        score = evaluator.evaluate_single("source text", "summary")
        assert 0.0 <= score <= 1.0

    def test_evaluate_batch_returns_list(self, evaluator: HHEMEvaluator):
        """Verify evaluate_batch returns list of floats."""
        pairs = [("source1", "summary1"), ("source2", "summary2")]
        scores = evaluator.evaluate_batch(pairs)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_evaluate_batch_empty(self, evaluator: HHEMEvaluator):
        """Verify evaluate_batch handles empty input."""
        scores = evaluator.evaluate_batch([])
        assert scores == []

    def test_evaluate_batch_scores_in_range(self, evaluator: HHEMEvaluator):
        """Verify batch scores are between 0 and 1."""
        pairs = [
            ("short", "long summary text"),
            ("very long source text" * 10, "brief"),
        ]
        scores = evaluator.evaluate_batch(pairs)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_run_benchmark_returns_result(self, evaluator: HHEMEvaluator):
        """Verify run_benchmark returns HHEMBenchmarkResult."""
        dataset = [("source", "summary", "basic")]
        result = evaluator.run_benchmark("test-model", dataset, [])
        assert isinstance(result, HHEMBenchmarkResult)

    def test_run_benchmark_empty_dataset(self, evaluator: HHEMEvaluator):
        """Verify run_benchmark handles empty dataset."""
        result = evaluator.run_benchmark("test-model", [], [])
        assert result.num_samples == 0
        assert result.mean_score == 0.0
        assert result.results == []

    def test_run_benchmark_has_correct_fields(self, fixed_evaluator: HHEMEvaluator):
        """Verify result has all required fields."""
        dataset = [
            ("source1", "summary1", "basic"),
            ("source2", "summary2", "rag"),
        ]
        result = fixed_evaluator.run_benchmark("test-model", dataset, [])

        assert result.model_name == "test-model"
        assert result.num_samples == 2
        assert isinstance(result.mean_score, float)
        assert isinstance(result.median_score, float)
        assert isinstance(result.std_score, float)
        assert isinstance(result.pass_rate_at_05, float)
        assert isinstance(result.pass_rate_at_07, float)
        assert isinstance(result.timestamp, str)
        assert len(result.results) == 2

    def test_run_benchmark_individual_results(self, fixed_evaluator: HHEMEvaluator):
        """Verify individual HHEMResult objects are correct."""
        dataset = [("source text", "summary text", "basic")]
        result = fixed_evaluator.run_benchmark("test-model", dataset, [])

        assert len(result.results) == 1
        r = result.results[0]
        assert isinstance(r, HHEMResult)
        assert r.model_name == "test-model"
        assert r.prompt_template == "basic"
        assert r.source_text == "source text"
        assert r.generated_summary == "summary text"
        assert isinstance(r.hhem_score, float)
        assert isinstance(r.timestamp, str)

    def test_run_benchmark_template_filtering(self, fixed_evaluator: HHEMEvaluator):
        """Verify template filtering works."""
        dataset = [
            ("s1", "m1", "basic"),
            ("s2", "m2", "rag"),
            ("s3", "m3", "basic"),
            ("s4", "m4", "few_shot"),
        ]
        result = fixed_evaluator.run_benchmark("test-model", dataset, ["basic"])

        assert result.num_samples == 2
        assert all(r.prompt_template == "basic" for r in result.results)

    def test_run_benchmark_multiple_templates(self, fixed_evaluator: HHEMEvaluator):
        """Verify filtering by multiple templates."""
        dataset = [
            ("s1", "m1", "basic"),
            ("s2", "m2", "rag"),
            ("s3", "m3", "few_shot"),
        ]
        result = fixed_evaluator.run_benchmark("test-model", dataset, ["basic", "rag"])

        assert result.num_samples == 2
        templates = {r.prompt_template for r in result.results}
        assert templates == {"basic", "rag"}

    def test_run_benchmark_statistics(self, fixed_evaluator: HHEMEvaluator):
        """Verify statistics are computed correctly."""
        # Fixed scores: 0.9, 0.1, 0.5, 0.7
        dataset = [
            ("s1", "m1", "t1"),
            ("s2", "m2", "t2"),
            ("s3", "m3", "t3"),
            ("s4", "m4", "t4"),
        ]
        result = fixed_evaluator.run_benchmark("test-model", dataset, [])

        # mean = (0.9 + 0.1 + 0.5 + 0.7) / 4 = 0.55
        assert abs(result.mean_score - 0.55) < 0.01
        # median of [0.1, 0.5, 0.7, 0.9] = (0.5 + 0.7) / 2 = 0.6
        assert abs(result.median_score - 0.6) < 0.01
        # pass@0.5: 3 out of 4 (0.9, 0.5, 0.7)
        assert abs(result.pass_rate_at_05 - 0.75) < 0.01
        # pass@0.7: 2 out of 4 (0.9, 0.7)
        assert abs(result.pass_rate_at_07 - 0.50) < 0.01


class TestGetEvaluator:
    """Tests for get_evaluator factory function."""

    def test_returns_evaluator(self):
        """Verify get_evaluator returns HHEMEvaluator."""
        mock = MockHHEMModel()
        evaluator = get_evaluator(model=mock)
        assert isinstance(evaluator, HHEMEvaluator)

    def test_with_mock_model(self):
        """Verify evaluator works with injected mock."""
        mock = MockHHEMModel(fixed_scores=[0.8])
        evaluator = get_evaluator(model=mock)
        score = evaluator.evaluate_single("source", "summary")
        assert score == 0.8


class TestDatasets:
    """Tests for dataset generation functions."""

    def test_generate_test_cases_returns_list(self):
        """Verify generate_test_cases returns list of EmailTestCase."""
        cases = generate_test_cases()
        assert isinstance(cases, list)
        assert len(cases) >= 20  # At least 20 cases
        assert all(isinstance(c, EmailTestCase) for c in cases)

    def test_email_test_case_fields(self):
        """Verify EmailTestCase has all required fields."""
        cases = generate_test_cases()
        case = cases[0]
        assert hasattr(case, "source")
        assert hasattr(case, "grounded_summary")
        assert hasattr(case, "hallucinated_summary")
        assert hasattr(case, "category")
        assert hasattr(case, "template")

    def test_generate_grounded_pairs_returns_tuples(self):
        """Verify grounded pairs format."""
        pairs = generate_grounded_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) >= 20
        assert all(isinstance(p, tuple) and len(p) == 3 for p in pairs)
        # Check tuple contents are strings
        for source, summary, template in pairs:
            assert isinstance(source, str)
            assert isinstance(summary, str)
            assert isinstance(template, str)

    def test_generate_hallucinated_pairs_returns_tuples(self):
        """Verify hallucinated pairs format."""
        pairs = generate_hallucinated_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) >= 20
        assert all(isinstance(p, tuple) and len(p) == 3 for p in pairs)

    def test_grounded_and_hallucinated_same_length(self):
        """Verify both datasets have same number of items."""
        grounded = generate_grounded_pairs()
        hallucinated = generate_hallucinated_pairs()
        assert len(grounded) == len(hallucinated)

    def test_generate_mixed_dataset(self):
        """Verify mixed dataset contains both types."""
        mixed = generate_mixed_dataset()
        grounded = generate_grounded_pairs()
        # Mixed should have 2x the items (grounded + hallucinated)
        assert len(mixed) == 2 * len(grounded)

    def test_dataset_has_diverse_categories(self):
        """Verify dataset has multiple categories."""
        cases = generate_test_cases()
        categories = {case.category for case in cases}
        assert len(categories) >= 3  # professional, personal, newsletter

    def test_dataset_has_diverse_templates(self):
        """Verify dataset has multiple templates."""
        cases = generate_test_cases()
        templates = {case.template for case in cases}
        assert len(templates) >= 2  # basic, rag, few_shot

    def test_get_dataset_metadata(self):
        """Verify metadata function returns correct structure."""
        metadata = get_dataset_metadata()
        assert "total_cases" in metadata
        assert "total_pairs_mixed" in metadata
        assert "categories" in metadata
        assert "templates" in metadata
        assert metadata["total_cases"] >= 20

    def test_sources_are_not_empty(self):
        """Verify all source texts have content."""
        cases = generate_test_cases()
        for case in cases:
            assert case.source.strip(), "Source cannot be empty"
            assert len(case.source) >= 50  # Meaningful content

    def test_summaries_are_not_empty(self):
        """Verify all summaries have content."""
        cases = generate_test_cases()
        for case in cases:
            assert case.grounded_summary.strip(), "Grounded summary cannot be empty"
            assert case.hallucinated_summary.strip(), "Hallucinated summary cannot be empty"

    def test_grounded_summaries_are_concise(self):
        """Verify grounded summaries are reasonable length."""
        cases = generate_test_cases()
        for case in cases:
            # Summary should be shorter than source
            assert len(case.grounded_summary) < len(case.source)

    def test_hallucinated_summaries_differ_from_grounded(self):
        """Verify hallucinated and grounded summaries are different."""
        cases = generate_test_cases()
        for case in cases:
            assert case.grounded_summary != case.hallucinated_summary


class TestContractCompliance:
    """Verify HHEMEvaluator implements HallucinationEvaluator protocol."""

    def test_has_evaluate_single_method(self):
        """Verify evaluate_single method exists."""
        evaluator = HHEMEvaluator(model=MockHHEMModel())
        assert hasattr(evaluator, "evaluate_single")
        assert callable(evaluator.evaluate_single)

    def test_has_evaluate_batch_method(self):
        """Verify evaluate_batch method exists."""
        evaluator = HHEMEvaluator(model=MockHHEMModel())
        assert hasattr(evaluator, "evaluate_batch")
        assert callable(evaluator.evaluate_batch)

    def test_has_run_benchmark_method(self):
        """Verify run_benchmark method exists."""
        evaluator = HHEMEvaluator(model=MockHHEMModel())
        assert hasattr(evaluator, "run_benchmark")
        assert callable(evaluator.run_benchmark)

    def test_evaluate_single_signature(self):
        """Verify evaluate_single has correct signature."""
        evaluator = HHEMEvaluator(model=MockHHEMModel())
        # Should accept source and summary strings
        result = evaluator.evaluate_single("source", "summary")
        assert isinstance(result, float)

    def test_evaluate_batch_signature(self):
        """Verify evaluate_batch has correct signature."""
        evaluator = HHEMEvaluator(model=MockHHEMModel())
        # Should accept list of tuples
        result = evaluator.evaluate_batch([("s1", "m1"), ("s2", "m2")])
        assert isinstance(result, list)
        assert all(isinstance(s, float) for s in result)

    def test_run_benchmark_signature(self):
        """Verify run_benchmark has correct signature."""
        evaluator = HHEMEvaluator(model=MockHHEMModel())
        dataset = [("source", "summary", "template")]
        result = evaluator.run_benchmark("model", dataset, [])
        assert isinstance(result, HHEMBenchmarkResult)


class TestCLIRunner:
    """Tests for the HHEM CLI runner."""

    def test_main_creates_output_file(self, tmp_path, monkeypatch):
        """Test main function creates output file."""
        import sys

        # Mock the evaluator to use our mock model
        from benchmarks.hallucination import hhem

        original_get_evaluator = hhem.get_evaluator
        hhem.get_evaluator = lambda model=None: HHEMEvaluator(
            model=MockHHEMModel(fixed_scores=[0.8])
        )

        try:
            from benchmarks.hallucination.run import main

            output_file = tmp_path / "results.json"

            # Mock sys.argv
            monkeypatch.setattr(
                sys, "argv", ["run.py", "--output", str(output_file), "--dataset", "grounded"]
            )

            result = main()

            assert result == 0
            assert output_file.exists()

            # Verify output is valid JSON
            import json

            with open(output_file) as f:
                data = json.load(f)

            assert "model_name" in data
            assert "num_samples" in data
            assert "mean_score" in data
            assert "median_score" in data
            assert "pass_rate_at_05" in data
            assert "pass_rate_at_07" in data
            assert "timestamp" in data
        finally:
            hhem.get_evaluator = original_get_evaluator

    def test_main_with_model_name(self, tmp_path, monkeypatch):
        """Test main function with custom model name."""
        import sys

        from benchmarks.hallucination import hhem

        original_get_evaluator = hhem.get_evaluator
        hhem.get_evaluator = lambda model=None: HHEMEvaluator(
            model=MockHHEMModel(fixed_scores=[0.6])
        )

        try:
            from benchmarks.hallucination.run import main

            output_file = tmp_path / "results.json"

            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "run.py",
                    "--output",
                    str(output_file),
                    "--model-name",
                    "my-custom-model",
                    "--dataset",
                    "grounded",
                ],
            )

            main()

            import json

            with open(output_file) as f:
                data = json.load(f)

            assert data["model_name"] == "my-custom-model"
        finally:
            hhem.get_evaluator = original_get_evaluator

    def test_main_prints_summary(self, tmp_path, monkeypatch, capsys):
        """Test main function prints summary."""
        import sys

        from benchmarks.hallucination import hhem

        original_get_evaluator = hhem.get_evaluator
        hhem.get_evaluator = lambda model=None: HHEMEvaluator(
            model=MockHHEMModel(fixed_scores=[0.7])
        )

        try:
            from benchmarks.hallucination.run import main

            output_file = tmp_path / "results.json"

            monkeypatch.setattr(
                sys, "argv", ["run.py", "--output", str(output_file), "--dataset", "grounded"]
            )

            main()

            captured = capsys.readouterr()
            assert "HHEM BENCHMARK RESULTS" in captured.out
            assert "Mean Score:" in captured.out
            assert "Pass Rate @0.5:" in captured.out
            assert "Gate G3:" in captured.out
        finally:
            hhem.get_evaluator = original_get_evaluator
