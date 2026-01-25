"""Template coverage analyzer implementation.

Workstream 3: Template Coverage Analyzer

Implements the CoverageAnalyzer protocol from contracts/coverage.py
using semantic similarity with sentence embeddings.
"""

import logging
from datetime import UTC, datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from benchmarks.coverage.templates import TEMPLATES
from contracts.coverage import CoverageResult, TemplateMatch

logger = logging.getLogger(__name__)

# Batch size for encoding queries
_BATCH_SIZE = 32


class TemplateCoverageAnalyzer:
    """Semantic similarity-based template matcher.

    Implements CoverageAnalyzer protocol from contracts/coverage.py.

    Uses all-MiniLM-L6-v2 sentence embeddings to compute cosine
    similarity between input queries and template patterns.
    """

    def __init__(self, templates: list[str] | None = None) -> None:
        """Initialize the analyzer.

        Args:
            templates: List of template strings. Uses default templates if not provided.
        """
        self._templates = list(templates) if templates else list(TEMPLATES)
        self._model: SentenceTransformer | None = None
        self._template_embeddings: np.ndarray | None = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazy load the embedding model.

        Returns:
            The loaded SentenceTransformer model
        """
        if self._model is None:
            logger.info("Loading sentence transformer: all-MiniLM-L6-v2")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _ensure_embeddings(self) -> np.ndarray:
        """Compute and cache template embeddings.

        Returns:
            Numpy array of template embeddings
        """
        if self._template_embeddings is None:
            model = self._ensure_model()
            logger.info("Computing embeddings for %d templates", len(self._templates))
            # BATCH ENCODE - never encode one at a time
            self._template_embeddings = model.encode(
                self._templates,
                batch_size=_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return self._template_embeddings

    def _compute_cosine_similarity(
        self, query_embedding: np.ndarray, template_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and templates.

        Args:
            query_embedding: Single query embedding vector
            template_embeddings: Matrix of template embeddings

        Returns:
            Array of similarity scores
        """
        # Normalize vectors for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        template_norms = template_embeddings / np.linalg.norm(
            template_embeddings, axis=1, keepdims=True
        )
        return np.dot(template_norms, query_norm)

    def match_query(self, query: str, threshold: float = 0.7) -> TemplateMatch:
        """Find best matching template for a query.

        Args:
            query: Input query string
            threshold: Minimum similarity score for a match (default 0.7)

        Returns:
            TemplateMatch with query, best template, similarity score, and match status
        """
        model = self._ensure_model()
        template_embeddings = self._ensure_embeddings()

        # Encode query
        query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]

        # Compute similarities
        similarities = self._compute_cosine_similarity(query_embedding, template_embeddings)

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_template = self._templates[best_idx] if best_score >= threshold else None

        return TemplateMatch(
            query=query,
            best_template=best_template,
            similarity_score=best_score,
            matched=best_score >= threshold,
        )

    def analyze_dataset(self, queries: list[str]) -> CoverageResult:
        """Analyze coverage across a dataset of queries.

        Args:
            queries: List of query strings to analyze

        Returns:
            CoverageResult with coverage metrics at different thresholds
        """
        if not queries:
            return CoverageResult(
                total_queries=0,
                coverage_at_50=0.0,
                coverage_at_70=0.0,
                coverage_at_90=0.0,
                unmatched_examples=[],
                template_usage={},
                timestamp=datetime.now(UTC).isoformat(),
            )

        model = self._ensure_model()
        template_embeddings = self._ensure_embeddings()

        # BATCH ENCODE all queries at once
        logger.info("Encoding %d queries", len(queries))
        query_embeddings = model.encode(
            queries,
            batch_size=_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Compute all similarities at once
        # Shape: (num_queries, num_templates)
        query_norms = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        template_norms = template_embeddings / np.linalg.norm(
            template_embeddings, axis=1, keepdims=True
        )
        all_similarities = np.dot(query_norms, template_norms.T)

        # Get best match for each query
        best_indices = np.argmax(all_similarities, axis=1)
        best_scores = all_similarities[np.arange(len(queries)), best_indices]

        # Compute coverage at different thresholds
        coverage_50 = float(np.mean(best_scores >= 0.5))
        coverage_70 = float(np.mean(best_scores >= 0.7))
        coverage_90 = float(np.mean(best_scores >= 0.9))

        # Find unmatched examples (below 0.7 threshold)
        unmatched_mask = best_scores < 0.7
        unmatched_indices = np.where(unmatched_mask)[0]
        unmatched_examples = [queries[i] for i in unmatched_indices[:20]]

        # Count template usage (for queries matching at 0.7)
        matched_mask = best_scores >= 0.7
        template_usage: dict[str, int] = {}
        for idx, is_matched in enumerate(matched_mask):
            if is_matched:
                template = self._templates[best_indices[idx]]
                template_usage[template] = template_usage.get(template, 0) + 1

        return CoverageResult(
            total_queries=len(queries),
            coverage_at_50=coverage_50,
            coverage_at_70=coverage_70,
            coverage_at_90=coverage_90,
            unmatched_examples=unmatched_examples,
            template_usage=template_usage,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def get_templates(self) -> list[str]:
        """Return all available templates.

        Returns:
            Copy of the template list
        """
        return list(self._templates)

    def add_template(self, template: str) -> None:
        """Add a new template dynamically.

        Invalidates cached embeddings so they will be recomputed on next use.

        Args:
            template: New template string to add
        """
        self._templates.append(template)
        # Invalidate cached embeddings
        self._template_embeddings = None
        logger.info("Added template, cache invalidated. Total templates: %d", len(self._templates))
