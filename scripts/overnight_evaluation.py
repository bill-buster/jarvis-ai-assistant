#!/usr/bin/env python3
"""JARVIS Overnight Evaluation Suite.

Runs for a configurable duration (default 8 hours), executing experiments
in priority order and looping back for deeper analysis when time permits.

Experiments (in priority order):
1. Router Threshold Grid Search (CRITICAL) - Find optimal thresholds
2. Response Diversity Audit (HIGH) - Check response variety
3. Clarify Path Analysis (HIGH) - Understand why queries fail to match
4. Top-K Selection Strategy (MEDIUM) - Compare selection strategies
5. LLM Generation Comparison (MEDIUM) - Template vs LLM quality

Each experiment runs at multiple "depth" levels:
- Depth 1 (Phase 1): Quick pass with fewer samples
- Depth 2 (Phase 2): Refined search around best values
- Depth 3+ (Phase 3+): Validation with more samples

Usage:
    python scripts/overnight_evaluation.py                    # Run for 8 hours
    python scripts/overnight_evaluation.py --duration 4       # Run for 4 hours
    python scripts/overnight_evaluation.py --resume           # Resume from checkpoint
    python scripts/overnight_evaluation.py --experiment grid  # Run specific experiment
    python scripts/overnight_evaluation.py --dry-run          # Test setup without running
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from jarvis.db import JarvisDB, Pair
    from jarvis.index import TriggerIndexSearcher
    from models import MLXGenerator


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DURATION_HOURS = 8
RESULTS_DIR = Path("results/overnight")
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Memory budget constraints (8GB M2 MacBook Air)
MAX_CACHED_EMBEDDINGS = 50000  # Limit response embeddings cached
MEMORY_WARNING_THRESHOLD_GB = 6.0  # Warn if usage exceeds this
MEMORY_CRITICAL_THRESHOLD_GB = 7.0  # Unload models if exceeds this
MAX_MEMORY_SAMPLES = 1000  # Bound memory sample list to prevent growth over 8 hours

# Comparison thresholds
TIE_MARGIN = 0.05  # Similarity difference below this is considered a tie

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Memory Utilities
# =============================================================================


def get_memory_usage_gb() -> float:
    """Get current process memory usage in GB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    except ImportError:
        # Fallback if psutil not available
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    except Exception:
        return 0.0


def get_system_memory_gb() -> tuple[float, float]:
    """Get system memory (total, available) in GB."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        return 8.0, 4.0  # Default assumption for 8GB machine


# =============================================================================
# Timing Context Manager
# =============================================================================


class Timer:
    """Simple timing context manager with memory tracking."""

    def __init__(self, name: str, log: bool = True):
        self.name = name
        self.log = log
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_memory: float = 0
        self.end_memory: float = 0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage_gb()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.end_memory = get_memory_usage_gb()
        if self.log:
            logger.debug(
                "   [%s] %.2fs, memory: %.2f -> %.2f GB (delta: %+.2f GB)",
                self.name,
                self.elapsed,
                self.start_memory,
                self.end_memory,
                self.memory_delta,
            )

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    @property
    def memory_delta(self) -> float:
        return self.end_memory - self.start_memory


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    priority: str  # CRITICAL, HIGH, MEDIUM
    function: str  # Method name to call


# =============================================================================
# Main Runner Class
# =============================================================================


class OvernightRunner:
    """Orchestrates overnight evaluation experiments."""

    def __init__(
        self,
        duration_hours: float,
        resume: bool = False,
        specific_experiment: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the overnight runner.

        Args:
            duration_hours: How long to run (in hours).
            resume: Whether to resume from checkpoint.
            specific_experiment: Run only this experiment (for testing).
            seed: Random seed for reproducibility. If None, uses current timestamp.
        """
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        self.results: dict[str, Any] = self._load_checkpoint() if resume else {}
        self.phase = 1
        self.specific_experiment = specific_experiment
        self._interrupted = False

        # Set random seed for reproducibility
        self.seed = seed if seed is not None else int(self.start_time.timestamp())
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.info("Random seed: %d", self.seed)

        # Lazy-loaded resources (save memory until needed)
        self._embedder: SentenceTransformer | None = None
        self._index_searcher: TriggerIndexSearcher | None = None
        self._db: JarvisDB | None = None
        self._response_embeddings: dict[int, np.ndarray] | None = None
        self._generator: MLXGenerator | None = None
        self._pairs_cache: list[Pair] | None = None
        self._pair_id_map: dict[int, Pair] | None = None

        # Memory tracking
        self._memory_samples: list[tuple[str, float, float]] = []  # (timestamp, usage_gb, event)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals gracefully."""
        logger.warning("\n⚠️  Interrupt received! Saving checkpoint and exiting...")
        self._interrupted = True
        self._save_checkpoint()
        self._cleanup_resources()
        sys.exit(0)

    def _sample_memory(self, event: str) -> float:
        """Sample current memory usage and record it (bounded to prevent memory leak)."""
        usage = get_memory_usage_gb()
        self._memory_samples.append((datetime.now().isoformat(), usage, event))
        # Bound the list size to prevent unbounded growth over 8 hours
        if len(self._memory_samples) > MAX_MEMORY_SAMPLES:
            # Keep first 100 (startup) + last 900 (recent)
            self._memory_samples = self._memory_samples[:100] + self._memory_samples[-900:]
        return usage

    def _check_memory_pressure(self) -> bool:
        """Check if memory pressure is high and take action if needed.

        Returns True if we should continue, False if memory is critical.
        """
        usage = get_memory_usage_gb()

        if usage > MEMORY_CRITICAL_THRESHOLD_GB:
            logger.warning(
                "⚠️  CRITICAL memory usage: %.2f GB > %.2f GB threshold",
                usage,
                MEMORY_CRITICAL_THRESHOLD_GB,
            )
            self._cleanup_resources()
            gc.collect()
            new_usage = get_memory_usage_gb()
            logger.info("   After cleanup: %.2f GB", new_usage)
            if new_usage > MEMORY_CRITICAL_THRESHOLD_GB:
                logger.error("   Memory still critical after cleanup!")
                return False
        elif usage > MEMORY_WARNING_THRESHOLD_GB:
            logger.warning(
                "⚠️  High memory usage: %.2f GB > %.2f GB warning threshold",
                usage,
                MEMORY_WARNING_THRESHOLD_GB,
            )

        return True

    def _cleanup_resources(self) -> None:
        """Clean up resources to free memory."""
        logger.info("🧹 Cleaning up resources...")

        # Unload generator
        self._unload_generator()

        # Unload embedder
        self._unload_embedder()

        # Clear response embeddings cache
        if self._response_embeddings:
            logger.info("   Clearing %d cached response embeddings", len(self._response_embeddings))
            self._response_embeddings = None

        # Force garbage collection
        gc.collect()

        # Clear MLX cache if available
        try:
            import mlx.core as mx

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except ImportError:
            pass

        logger.info("   Memory after cleanup: %.2f GB", get_memory_usage_gb())

    def time_remaining(self) -> timedelta:
        """Return time remaining until end."""
        return max(self.end_time - datetime.now(), timedelta(0))

    def should_continue(self) -> bool:
        """Check if we should continue running experiments."""
        return datetime.now() < self.end_time and not self._interrupted

    # =========================================================================
    # Resource Management (Lazy Loading)
    # =========================================================================

    @property
    def db(self) -> JarvisDB:
        """Get or create the database instance."""
        if self._db is None:
            from jarvis.db import get_db

            self._db = get_db()
            self._db.init_schema()
        return self._db

    @property
    def embedder(self) -> SentenceTransformer:
        """Get or create the sentence transformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model: %s (memory: %.2f GB)",
                EMBEDDING_MODEL,
                get_memory_usage_gb(),
            )
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
            self._sample_memory("embedder_loaded")
            logger.info("   Embedding model loaded (memory: %.2f GB)", get_memory_usage_gb())
        return self._embedder

    def _unload_embedder(self) -> None:
        """Unload the sentence transformer to free memory."""
        if self._embedder is not None:
            logger.info("Unloading embedding model to free memory...")
            del self._embedder
            self._embedder = None
            gc.collect()
            self._sample_memory("embedder_unloaded")

    @property
    def index_searcher(self) -> TriggerIndexSearcher:
        """Get or create the FAISS index searcher."""
        if self._index_searcher is None:
            from jarvis.index import TriggerIndexSearcher

            self._index_searcher = TriggerIndexSearcher(self.db)
        return self._index_searcher

    def _ensure_generator_loaded(self) -> None:
        """Ensure the LLM generator is loaded."""
        if self._generator is None:
            from models import get_generator

            logger.info("Loading LLM generator... (memory: %.2f GB)", get_memory_usage_gb())
            self._generator = get_generator(skip_templates=True)
            self._generator.load()
            self._sample_memory("generator_loaded")
            logger.info("   LLM generator loaded (memory: %.2f GB)", get_memory_usage_gb())

    def _unload_generator(self) -> None:
        """Unload the generator to free memory."""
        if self._generator is not None:
            before = get_memory_usage_gb()
            logger.info("Unloading LLM generator... (memory: %.2f GB)", before)
            self._generator.unload()
            self._generator = None
            gc.collect()
            try:
                import mlx.core as mx

                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except ImportError:
                pass
            after = get_memory_usage_gb()
            self._sample_memory("generator_unloaded")
            logger.info(
                "   Generator unloaded (memory: %.2f GB, freed: %.2f GB)", after, before - after
            )

    def _get_all_pairs(self) -> list[Pair]:
        """Get all pairs from the database (cached)."""
        if self._pairs_cache is None:
            logger.info("Loading pairs from database...")
            self._pairs_cache = self.db.get_all_pairs(min_quality=0.0)
            self._pair_id_map = {p.id: p for p in self._pairs_cache}
            logger.info("Loaded %d pairs", len(self._pairs_cache))
        return self._pairs_cache

    def _get_pair_by_id(self, pair_id: int) -> Pair | None:
        """Get a pair by its ID."""
        if self._pair_id_map is None:
            self._get_all_pairs()
        return self._pair_id_map.get(pair_id) if self._pair_id_map else None

    def _get_random_pairs(self, n: int) -> list[Pair]:
        """Get n random pairs from the database."""
        all_pairs = self._get_all_pairs()
        if len(all_pairs) <= n:
            return all_pairs
        return random.sample(all_pairs, n)

    def _get_random_triggers(self, n: int) -> list[str]:
        """Get n random trigger texts."""
        pairs = self._get_random_pairs(n)
        return [p.trigger_text for p in pairs]

    # =========================================================================
    # Embedding Utilities
    # =========================================================================

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embedder.encode([text], normalize_embeddings=True)[0]

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))

    def _get_response_embedding(self, pair_id: int, response_text: str) -> np.ndarray:
        """Get or compute embedding for a response (with caching)."""
        if self._response_embeddings is None:
            self._response_embeddings = {}

        if pair_id not in self._response_embeddings:
            # Limit cache size for memory
            if len(self._response_embeddings) >= MAX_CACHED_EMBEDDINGS:
                # Remove oldest entries (simple LRU approximation)
                keys_to_remove = list(self._response_embeddings.keys())[
                    : MAX_CACHED_EMBEDDINGS // 4
                ]
                for k in keys_to_remove:
                    del self._response_embeddings[k]

            self._response_embeddings[pair_id] = self._embed_text(response_text)

        return self._response_embeddings[pair_id]

    # =========================================================================
    # Search Utilities
    # =========================================================================

    def _search_excluding(
        self, exclude_pair_id: int, k: int = 5, threshold: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search FAISS index, excluding a specific pair.

        Returns scores and faiss_ids as numpy arrays.

        IMPORTANT: This implements leave-one-out evaluation by excluding
        the source pair from results. We must convert pair_id to faiss_id
        since FAISS results contain faiss_ids, not pair_ids.
        """
        # Get the pair to search for
        pair = self._get_pair_by_id(exclude_pair_id)
        if pair is None:
            return np.array([]), np.array([])

        # Get the faiss_id for this pair so we can exclude it from results
        # This is critical: FAISS returns faiss_ids, not pair_ids!
        embedding = self.db.get_embedding_by_pair(exclude_pair_id)
        exclude_faiss_id = embedding.faiss_id if embedding else -1

        # Search with extra results to account for self-match
        results = self.index_searcher.search(pair.trigger_text, k=k + 5, threshold=threshold)

        # Filter out self-match by faiss_id (not pair_id!) and convert to arrays
        filtered = [(score, fid) for fid, score in results if fid != exclude_faiss_id]

        if not filtered:
            return np.array([]), np.array([])

        scores = np.array([s for s, _ in filtered[:k]])
        ids = np.array([i for _, i in filtered[:k]])

        return scores, ids

    def _get_response_for_faiss_id(self, faiss_id: int) -> str | None:
        """Get response text for a FAISS ID."""
        pair = self.db.get_pair_by_faiss_id(faiss_id)
        return pair.response_text if pair else None

    def _get_template_response(self, trigger: str, threshold: float = 0.85) -> str | None:
        """Get template response for a trigger if above threshold."""
        results = self.index_searcher.search(trigger, k=1, threshold=threshold)
        if results:
            faiss_id, score = results[0]
            if score >= threshold:
                return self._get_response_for_faiss_id(faiss_id)
        return None

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load results from checkpoint file."""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE) as f:
                    data = json.load(f)
                logger.info(
                    "Resuming from checkpoint with %d results", len(data.get("results", {}))
                )
                self.phase = data.get("phase", 1)
                return data.get("results", {})
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s", e)
        return {}

    def _save_checkpoint(self) -> None:
        """Save current results to checkpoint file."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "phase": self.phase,
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        logger.debug("Checkpoint saved")

    # =========================================================================
    # Experiment 1: Router Threshold Grid Search
    # =========================================================================

    def experiment_threshold_grid(self, depth: int = 1) -> dict[str, Any]:
        """Find optimal TEMPLATE_THRESHOLD, CLARIFY_THRESHOLD, and TOP_K values.

        Depth 1 (Phase 1): Coarse grid, 500 samples (~10 min)
        Depth 2 (Phase 2): Fine grid around best values, 1000 samples (~20 min)
        Depth 3 (Phase 3+): Very fine grid, 2000 samples (~40 min)
        """
        if depth == 1:
            # Coarse search
            template_thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
            clarify_thresholds = [0.30, 0.40, 0.50]
            top_k_values = [1, 5, 10]
            n_samples = 500
        elif depth == 2:
            # Fine search around Phase 1 winner
            best = self.results.get("threshold_grid_1", {}).get("best", {})
            center_t = best.get("template_threshold", 0.85)
            center_c = best.get("clarify_threshold", 0.40)
            center_k = best.get("top_k", 5)

            template_thresholds = [
                max(0.5, center_t - 0.05),
                max(0.5, center_t - 0.02),
                center_t,
                min(1.0, center_t + 0.02),
                min(1.0, center_t + 0.05),
            ]
            clarify_thresholds = [
                max(0.1, center_c - 0.05),
                center_c,
                min(0.8, center_c + 0.05),
            ]
            top_k_values = [max(1, center_k - 2), center_k, center_k + 2]
            n_samples = 1000
        else:
            # Validation with more samples
            best = self.results.get("threshold_grid_2", {}).get("best", {})
            if not best:
                best = self.results.get("threshold_grid_1", {}).get("best", {})
            template_thresholds = [best.get("template_threshold", 0.85)]
            clarify_thresholds = [best.get("clarify_threshold", 0.40)]
            top_k_values = [best.get("top_k", 5)]
            n_samples = 2000

        # Sample pairs for evaluation (exclude self when querying)
        sample_pairs = self._get_random_pairs(n_samples)

        # Pre-compute actual response embeddings (used across all configs)
        logger.info("  Pre-computing %d actual response embeddings...", len(sample_pairs))
        response_texts = [p.response_text for p in sample_pairs]
        batch_embeddings = self._embed_texts(response_texts)
        actual_embeddings: dict[int, np.ndarray] = {
            p.id: emb for p, emb in zip(sample_pairs, batch_embeddings)
        }

        results_list: list[dict[str, Any]] = []
        total_configs = len(template_thresholds) * len(clarify_thresholds) * len(top_k_values)

        for i, (t_thresh, c_thresh, k) in enumerate(
            itertools.product(template_thresholds, clarify_thresholds, top_k_values)
        ):
            if not self.should_continue():
                break

            config_result = self._evaluate_config(
                sample_pairs, t_thresh, c_thresh, k, actual_embeddings
            )
            results_list.append(config_result)

            # Progress update
            if (i + 1) % 5 == 0 or (i + 1) == total_configs:
                logger.info(
                    "  Grid search: %d/%d configs | best score: %.3f | mem: %.2f GB",
                    i + 1,
                    total_configs,
                    max(r["score"] for r in results_list) if results_list else 0,
                    get_memory_usage_gb(),
                )
                # Check memory pressure
                if not self._check_memory_pressure():
                    logger.warning("Stopping experiment due to memory pressure")
                    break

        # Find best config
        best = max(results_list, key=lambda x: x["score"]) if results_list else {}

        return {
            "depth": depth,
            "n_samples": n_samples,
            "configs_tested": len(results_list),
            "all_results": results_list,
            "best": best,
            "timestamp": datetime.now().isoformat(),
        }

    def _evaluate_config(
        self,
        pairs: list[Pair],
        template_thresh: float,
        clarify_thresh: float,
        top_k: int,
        actual_embeddings: dict[int, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single threshold configuration.

        Args:
            pairs: List of pairs to evaluate.
            template_thresh: Threshold for template matching.
            clarify_thresh: Threshold below which to clarify.
            top_k: Number of top matches to consider.
            actual_embeddings: Pre-computed embeddings for actual responses (by pair ID).
                If provided, avoids re-computing embeddings for actual responses.
        """
        template_hits = 0
        generate_hits = 0
        clarify_hits = 0
        similarities: list[float] = []

        for pair in pairs:
            if not self.should_continue():
                break

            # Query FAISS (excluding self-match)
            scores, ids = self._search_excluding(pair.id, top_k + 5)

            if len(scores) == 0:
                clarify_hits += 1
                continue

            best_score = float(scores[0])

            if best_score >= template_thresh:
                template_hits += 1
                # Select response using top-K strategy (random from top-k)
                k_actual = min(top_k, len(ids))
                selected_idx = random.randint(0, k_actual - 1)
                selected_faiss_id = int(ids[selected_idx])
                predicted_response = self._get_response_for_faiss_id(selected_faiss_id)

                if predicted_response:
                    # Compare to actual response using pre-computed embedding if available
                    actual_emb = actual_embeddings.get(pair.id) if actual_embeddings else None
                    sim = self._response_similarity(
                        predicted_response, pair.response_text, actual_emb
                    )
                    similarities.append(sim)

            elif best_score < clarify_thresh:
                clarify_hits += 1
            else:
                generate_hits += 1

        n = len(pairs)
        avg_sim = float(np.mean(similarities)) if similarities else 0
        std_sim = float(np.std(similarities)) if similarities else 0

        # Composite score: balance coverage and quality
        # We want high template rate AND high similarity
        template_rate = template_hits / n if n > 0 else 0
        score = (template_rate * 0.4) + (avg_sim * 0.6)  # Weighted combination

        return {
            "template_threshold": template_thresh,
            "clarify_threshold": clarify_thresh,
            "top_k": top_k,
            "template_rate": template_rate,
            "generate_rate": generate_hits / n if n > 0 else 0,
            "clarify_rate": clarify_hits / n if n > 0 else 0,
            "avg_similarity": avg_sim,
            "std_similarity": std_sim,
            "n_similarities": len(similarities),
            "score": score,
        }

    def _response_similarity(
        self,
        predicted: str,
        actual: str,
        actual_emb: np.ndarray | None = None,
    ) -> float:
        """Compute similarity between predicted and actual responses.

        Args:
            predicted: The predicted response text.
            actual: The actual response text.
            actual_emb: Pre-computed embedding for the actual response.
                If provided, avoids re-computing the embedding.
        """
        if predicted == actual:
            return 1.0
        pred_emb = self._embed_text(predicted)
        if actual_emb is None:
            actual_emb = self._embed_text(actual)
        return self._cosine_sim(pred_emb, actual_emb)

    # =========================================================================
    # Experiment 2: Response Diversity Audit
    # =========================================================================

    def experiment_diversity_audit(self, depth: int = 1) -> dict[str, Any]:
        """Check if the system returns diverse responses or degenerates to a few.

        Uses leave-one-out evaluation to avoid self-matching bias.

        Depth 1: Sample 5000 queries, track response distribution
        Depth 2: Analyze by trigger pattern (greetings, invitations, etc.)
        Depth 3: More comprehensive analysis with 15000 queries
        """
        n_samples = 5000 * depth
        sample_pairs = self._get_random_pairs(n_samples)  # Use pairs for leave-one-out

        response_counts: Counter[str] = Counter()
        response_examples: dict[str, list[str]] = defaultdict(list)
        no_match_count = 0

        for i, pair in enumerate(sample_pairs):
            if not self.should_continue():
                break

            # Use leave-one-out search to avoid self-matching bias
            scores, ids = self._search_excluding(pair.id, k=1, threshold=0.0)

            if len(scores) > 0 and scores[0] >= 0.7:
                faiss_id = int(ids[0])
                response = self._get_response_for_faiss_id(faiss_id)
                if response:
                    response_counts[response] += 1
                    if len(response_examples[response]) < 5:
                        response_examples[response].append(pair.trigger_text)
                else:
                    no_match_count += 1
            else:
                no_match_count += 1

            if (i + 1) % 1000 == 0:
                logger.info(
                    "  Diversity audit: %d/%d queries | mem: %.2f GB",
                    i + 1,
                    n_samples,
                    get_memory_usage_gb(),
                )

        # Analysis
        total_queries = sum(response_counts.values())
        unique_responses = len(response_counts)

        if total_queries == 0:
            return {
                "depth": depth,
                "n_queries": len(sample_pairs),
                "no_match_count": no_match_count,
                "message": "No responses matched threshold",
                "timestamp": datetime.now().isoformat(),
            }

        # Distribution analysis
        top_10 = response_counts.most_common(10)
        top_10_coverage = sum(count for _, count in top_10) / total_queries

        top_50 = response_counts.most_common(50)
        top_50_coverage = sum(count for _, count in top_50) / total_queries

        # Entropy (higher = more diverse)
        probs = np.array(list(response_counts.values())) / total_queries
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = np.log2(unique_responses) if unique_responses > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "depth": depth,
            "n_queries": len(sample_pairs),
            "n_matched": total_queries,
            "no_match_count": no_match_count,
            "unique_responses": unique_responses,
            "top_10_coverage": top_10_coverage,
            "top_50_coverage": top_50_coverage,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "top_10_responses": [
                {
                    "response": resp[:100] + ("..." if len(resp) > 100 else ""),
                    "count": count,
                    "example_triggers": response_examples.get(resp, [])[:3],
                }
                for resp, count in top_10
            ],
            "diversity_grade": self._grade_diversity(normalized_entropy, top_10_coverage),
            "timestamp": datetime.now().isoformat(),
        }

    def _grade_diversity(self, entropy: float, top_10_coverage: float) -> str:
        """Grade diversity on A-F scale."""
        if entropy > 0.8 and top_10_coverage < 0.3:
            return "A"  # Excellent diversity
        elif entropy > 0.6 and top_10_coverage < 0.5:
            return "B"  # Good diversity
        elif entropy > 0.4 and top_10_coverage < 0.7:
            return "C"  # Acceptable
        elif entropy > 0.2:
            return "D"  # Poor diversity
        else:
            return "F"  # Degenerate - returns same few responses

    # =========================================================================
    # Experiment 3: Clarify Path Analysis
    # =========================================================================

    def experiment_clarify_analysis(self, depth: int = 1) -> dict[str, Any]:
        """Analyze what types of messages fall below the clarify threshold.

        Goal: Understand WHY certain queries fail to match, so we can either:
        - Improve the matching
        - Generate better "ask for help" responses
        - Identify patterns that should route to LLM generation instead
        """
        clarify_threshold = (
            self.results.get("threshold_grid_1", {}).get("best", {}).get("clarify_threshold", 0.40)
        )

        n_samples = 10000 * depth
        all_pairs = self._get_random_pairs(n_samples)

        clarify_cases: list[dict[str, Any]] = []

        for i, pair in enumerate(all_pairs):
            if not self.should_continue():
                break

            scores, ids = self._search_excluding(pair.id, k=1)
            if len(scores) == 0 or scores[0] < clarify_threshold:
                clarify_cases.append(
                    {
                        "trigger": pair.trigger_text,
                        "actual_response": pair.response_text,
                        "best_score": float(scores[0]) if len(scores) > 0 else 0,
                        "trigger_length": len(pair.trigger_text),
                        "word_count": len(pair.trigger_text.split()),
                    }
                )

            if (i + 1) % 2000 == 0:
                logger.info(
                    "  Clarify analysis: %d/%d pairs | %d clarify cases | mem: %.2f GB",
                    i + 1,
                    n_samples,
                    len(clarify_cases),
                    get_memory_usage_gb(),
                )

        if not clarify_cases:
            return {
                "depth": depth,
                "clarify_count": 0,
                "message": "No clarify cases found",
                "timestamp": datetime.now().isoformat(),
            }

        # Analyze patterns
        clarify_rate = len(clarify_cases) / len(all_pairs) if all_pairs else 0

        # Length analysis
        avg_length = float(np.mean([c["trigger_length"] for c in clarify_cases]))
        avg_words = float(np.mean([c["word_count"] for c in clarify_cases]))

        # Common patterns (simple heuristics)
        patterns = {
            "single_word": 0,
            "question": 0,
            "contains_pronoun_reference": 0,
            "very_short": 0,
            "very_long": 0,
            "contains_emoji_only": 0,
        }

        pronoun_refs = {"it", "that", "this", "those", "these", "them", "they"}

        for case in clarify_cases:
            trigger = case["trigger"].lower()
            words = trigger.split()

            if len(words) == 1:
                patterns["single_word"] += 1
            if "?" in trigger:
                patterns["question"] += 1
            if any(p in words for p in pronoun_refs):
                patterns["contains_pronoun_reference"] += 1
            if len(trigger) < 10:
                patterns["very_short"] += 1
            if len(trigger) > 200:
                patterns["very_long"] += 1
            # Simple emoji check (not comprehensive)
            if all(ord(c) > 127 or c.isspace() for c in trigger):
                patterns["contains_emoji_only"] += 1

        # Normalize to percentages
        n_clarify = len(clarify_cases)
        pattern_rates = {k: v / n_clarify for k, v in patterns.items()}

        # Sample some examples for manual review
        examples = clarify_cases[:20]

        return {
            "depth": depth,
            "clarify_threshold": clarify_threshold,
            "total_sampled": len(all_pairs),
            "clarify_count": len(clarify_cases),
            "clarify_rate": clarify_rate,
            "avg_trigger_length": avg_length,
            "avg_word_count": avg_words,
            "pattern_rates": pattern_rates,
            "examples": [
                {
                    "trigger": e["trigger"][:100],
                    "best_score": e["best_score"],
                    "word_count": e["word_count"],
                }
                for e in examples
            ],
            "recommendations": self._clarify_recommendations(pattern_rates, clarify_rate),
            "timestamp": datetime.now().isoformat(),
        }

    def _clarify_recommendations(
        self, pattern_rates: dict[str, float], clarify_rate: float
    ) -> list[str]:
        """Generate actionable recommendations based on clarify analysis."""
        recs = []

        if clarify_rate > 0.3:
            recs.append("HIGH CLARIFY RATE: Consider lowering clarify_threshold")

        if pattern_rates.get("very_short", 0) > 0.4:
            recs.append("Many short triggers fail - consider special handling for brief messages")

        if pattern_rates.get("contains_pronoun_reference", 0) > 0.3:
            recs.append(
                "Pronoun references ('it', 'that') often fail - these genuinely need context"
            )

        if pattern_rates.get("question", 0) > 0.5:
            recs.append("Questions often fail to match - may need question-specific handling")

        if pattern_rates.get("single_word", 0) > 0.3:
            recs.append("Single-word messages often fail - consider using broader matching")

        if pattern_rates.get("contains_emoji_only", 0) > 0.1:
            recs.append("Emoji-only messages fail - need specialized emoji response handling")

        return recs if recs else ["Clarify behavior looks reasonable"]

    # =========================================================================
    # Experiment 4: Top-K Selection Strategy
    # =========================================================================

    def experiment_topk_strategy(self, depth: int = 1) -> dict[str, Any]:
        """Compare different strategies for selecting from top-K matches.

        Strategies:
        1. Always pick top-1 (highest similarity)
        2. Random from top-K
        3. Weighted random (probability proportional to similarity)
        4. Most common response among top-K (voting)

        Which produces responses most similar to actual behavior?
        """
        n_samples = 1000 * depth
        sample_pairs = self._get_random_pairs(n_samples)

        results: dict[str, dict[str, Any]] = {
            "top_1": {"similarities": [], "exact_matches": 0},
            "random_top_5": {"similarities": [], "exact_matches": 0},
            "weighted_random": {"similarities": [], "exact_matches": 0},
            "voting": {"similarities": [], "exact_matches": 0},
        }

        # Pre-compute actual response embeddings to avoid redundant computation
        # Use batch embedding for ~2x speedup
        logger.info("  Pre-computing %d actual response embeddings (batched)...", len(sample_pairs))
        response_texts = [pair.response_text for pair in sample_pairs]
        batch_embeddings = self._embed_texts(response_texts)
        actual_embeddings: dict[int, np.ndarray] = {
            pair.id: emb for pair, emb in zip(sample_pairs, batch_embeddings)
        }

        # Cache for response embeddings to avoid re-computing for same response text
        response_emb_cache: dict[str, np.ndarray] = {}

        def get_response_embedding(resp_text: str) -> np.ndarray:
            """Get or compute embedding for a response (with caching)."""
            if resp_text not in response_emb_cache:
                response_emb_cache[resp_text] = self._embed_text(resp_text)
            return response_emb_cache[resp_text]

        for i, pair in enumerate(sample_pairs):
            if not self.should_continue():
                break

            scores, ids = self._search_excluding(pair.id, k=10)
            if len(scores) < 2:
                continue

            responses = [self._get_response_for_faiss_id(int(fid)) for fid in ids]
            responses = [r for r in responses if r is not None]
            if len(responses) < 2:
                continue

            actual_emb = actual_embeddings[pair.id]

            # Strategy 1: top-1
            selected = responses[0]
            sim = self._cosine_sim(get_response_embedding(selected), actual_emb)
            results["top_1"]["similarities"].append(sim)
            if selected == pair.response_text:
                results["top_1"]["exact_matches"] += 1

            # Strategy 2: random from top-5
            k5 = min(5, len(responses))
            selected = random.choice(responses[:k5])
            sim = self._cosine_sim(get_response_embedding(selected), actual_emb)
            results["random_top_5"]["similarities"].append(sim)
            if selected == pair.response_text:
                results["random_top_5"]["exact_matches"] += 1

            # Strategy 3: weighted random
            weights = scores[: len(responses)].tolist()
            selected = random.choices(responses, weights=weights, k=1)[0]
            sim = self._cosine_sim(get_response_embedding(selected), actual_emb)
            results["weighted_random"]["similarities"].append(sim)
            if selected == pair.response_text:
                results["weighted_random"]["exact_matches"] += 1

            # Strategy 4: voting (most common response)
            response_counter = Counter(responses)
            selected = response_counter.most_common(1)[0][0]
            sim = self._cosine_sim(get_response_embedding(selected), actual_emb)
            results["voting"]["similarities"].append(sim)
            if selected == pair.response_text:
                results["voting"]["exact_matches"] += 1

            if (i + 1) % 200 == 0:
                logger.info(
                    "  Top-K strategy: %d/%d pairs | mem: %.2f GB",
                    i + 1,
                    n_samples,
                    get_memory_usage_gb(),
                )

        # Compute summary stats
        summary = {}
        for name, data in results.items():
            sims = data["similarities"]
            summary[name] = {
                "avg_similarity": float(np.mean(sims)) if sims else 0,
                "std_similarity": float(np.std(sims)) if sims else 0,
                "exact_match_rate": data["exact_matches"] / len(sims) if sims else 0,
                "n_samples": len(sims),
            }

        # Find best strategy
        best_strategy = max(summary.items(), key=lambda x: x[1]["avg_similarity"])

        return {
            "depth": depth,
            "strategies_tested": list(results.keys()),
            "results": summary,
            "best_strategy": best_strategy[0],
            "best_avg_similarity": best_strategy[1]["avg_similarity"],
            "timestamp": datetime.now().isoformat(),
        }

    # =========================================================================
    # Experiment 5: LLM Generation Comparison
    # =========================================================================

    def experiment_llm_comparison(self, depth: int = 1) -> dict[str, Any]:
        """Compare template responses vs LLM generation for borderline cases.

        This is EXPENSIVE (runs LFM 2.5 for each sample), so we limit samples.

        Depth 1: 50 samples (~2 min)
        Depth 2: 100 samples (~4 min)
        Depth 3: 200 samples (~8 min)
        """
        n_samples = 50 * depth

        # Get the current best thresholds
        best_config = self.results.get("threshold_grid_1", {}).get("best", {})
        template_thresh = best_config.get("template_threshold", 0.85)
        clarify_thresh = best_config.get("clarify_threshold", 0.40)

        # Find pairs that fall in the "generate" zone
        all_pairs = self._get_random_pairs(n_samples * 10)

        generate_zone_pairs: list[tuple[Pair, float]] = []
        for pair in all_pairs:
            scores, _ = self._search_excluding(pair.id, k=1)
            if len(scores) > 0 and clarify_thresh <= scores[0] < template_thresh:
                generate_zone_pairs.append((pair, float(scores[0])))
            if len(generate_zone_pairs) >= n_samples:
                break

        if not generate_zone_pairs:
            return {
                "depth": depth,
                "message": "No generate-zone pairs found",
                "n_found": 0,
                "timestamp": datetime.now().isoformat(),
            }

        # Pre-compute embeddings for template responses and actual responses
        # This allows us to unload the embedder before loading the LLM
        logger.info("   Pre-computing embeddings for comparison...")
        template_embeddings: dict[int, np.ndarray] = {}
        actual_embeddings: dict[int, np.ndarray] = {}

        for pair, _ in generate_zone_pairs:
            actual_embeddings[pair.id] = self._embed_text(pair.response_text)
            template_response = self._get_template_response(pair.trigger_text, threshold=0.0)
            if template_response:
                template_embeddings[pair.id] = self._embed_text(template_response)

        # Now unload embedder to make room for LLM
        logger.info("   Unloading embedder to make room for LLM...")
        self._unload_embedder()
        gc.collect()

        # Load generator (expensive, do once)
        self._ensure_generator_loaded()

        # Generate LLM responses
        llm_responses: dict[int, str] = {}
        for i, (pair, match_score) in enumerate(generate_zone_pairs):
            if not self.should_continue():
                break

            llm_response = self._generate_with_llm(pair.trigger_text)
            if llm_response:
                llm_responses[pair.id] = llm_response

            if (i + 1) % 10 == 0:
                logger.info(
                    "   Generating: %d/%d | mem: %.2f GB",
                    i + 1,
                    len(generate_zone_pairs),
                    get_memory_usage_gb(),
                )

        # Unload generator to free memory before computing LLM embeddings
        self._unload_generator()

        # Reload embedder to compute LLM response embeddings
        logger.info("   Computing LLM response embeddings...")
        llm_embeddings: dict[int, np.ndarray] = {}
        for pair_id, llm_response in llm_responses.items():
            llm_embeddings[pair_id] = self._embed_text(llm_response)

        # Now compare
        results_list: list[dict[str, Any]] = []
        template_wins = 0
        llm_wins = 0
        ties = 0

        for pair, match_score in generate_zone_pairs:
            if pair.id not in template_embeddings or pair.id not in llm_embeddings:
                continue

            actual_emb = actual_embeddings[pair.id]
            template_emb = template_embeddings[pair.id]
            llm_emb = llm_embeddings[pair.id]

            template_sim = self._cosine_sim(template_emb, actual_emb)
            llm_sim = self._cosine_sim(llm_emb, actual_emb)

            # Determine winner
            if abs(template_sim - llm_sim) < TIE_MARGIN:
                winner = "tie"
                ties += 1
            elif template_sim > llm_sim:
                winner = "template"
                template_wins += 1
            else:
                winner = "llm"
                llm_wins += 1

            # Get original responses for reporting
            template_response = self._get_template_response(pair.trigger_text, threshold=0.0) or ""
            llm_response = llm_responses.get(pair.id, "")

            results_list.append(
                {
                    "trigger": pair.trigger_text[:80],
                    "actual": pair.response_text[:80],
                    "template_response": template_response[:80],
                    "llm_response": llm_response[:80],
                    "match_score": match_score,
                    "template_similarity": template_sim,
                    "llm_similarity": llm_sim,
                    "winner": winner,
                }
            )

        logger.info(
            "   Results: Template wins: %d, LLM wins: %d, Ties: %d",
            template_wins,
            llm_wins,
            ties,
        )

        total = template_wins + llm_wins + ties

        return {
            "depth": depth,
            "n_samples": len(results_list),
            "template_wins": template_wins,
            "llm_wins": llm_wins,
            "ties": ties,
            "template_win_rate": template_wins / total if total > 0 else 0,
            "llm_win_rate": llm_wins / total if total > 0 else 0,
            "recommendation": (
                "Use LLM for generate zone"
                if llm_wins > template_wins
                else "Template is sufficient"
            ),
            "examples": results_list[:10],
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_with_llm(self, trigger: str) -> str | None:
        """Generate a response using the LLM."""
        if self._generator is None:
            return None

        try:
            from contracts.models import GenerationRequest

            request = GenerationRequest(
                prompt=f"Reply to this message naturally: {trigger}",
                max_tokens=100,
                temperature=0.7,
            )
            response = self._generator.generate(request)
            return response.text.strip()
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            return None

    # =========================================================================
    # Main Execution Loop
    # =========================================================================

    def _run_experiment_cycle(self) -> None:
        """Run one cycle of all experiments at current depth."""
        experiments = [
            ("threshold_grid", self.experiment_threshold_grid, "CRITICAL"),
            ("diversity_audit", self.experiment_diversity_audit, "HIGH"),
            ("clarify_analysis", self.experiment_clarify_analysis, "HIGH"),
            ("topk_strategy", self.experiment_topk_strategy, "MEDIUM"),
            ("llm_comparison", self.experiment_llm_comparison, "MEDIUM"),
        ]

        # Filter to specific experiment if requested
        if self.specific_experiment:
            experiments = [
                (name, fn, priority)
                for name, fn, priority in experiments
                if name.startswith(self.specific_experiment)
            ]
            if not experiments:
                logger.error("Unknown experiment: %s", self.specific_experiment)
                return

        for exp_name, exp_fn, priority in experiments:
            if not self.should_continue():
                logger.info("⏰ Time expired, stopping before %s", exp_name)
                break

            result_key = f"{exp_name}_{self.phase}"

            if result_key in self.results:
                logger.info("⏭️  Skipping %s (already completed in this phase)", exp_name)
                continue

            # Check memory before starting
            self._sample_memory(f"{exp_name}_start")
            start_memory = get_memory_usage_gb()

            logger.info("")
            logger.info("=" * 60)
            logger.info("🔬 Running: %s (Phase %d, Priority: %s)", exp_name, self.phase, priority)
            logger.info(
                "   Time remaining: %s | Memory: %.2f GB", self.time_remaining(), start_memory
            )
            logger.info("=" * 60)

            start_exp_time = time.perf_counter()
            try:
                result = exp_fn(depth=self.phase)
                elapsed = time.perf_counter() - start_exp_time

                # Record timing and memory stats
                result["elapsed_seconds"] = elapsed
                result["memory_start_gb"] = start_memory
                result["memory_end_gb"] = get_memory_usage_gb()
                result["memory_delta_gb"] = result["memory_end_gb"] - start_memory

                self.results[result_key] = result
                self._save_checkpoint()
                self._sample_memory(f"{exp_name}_end")

                logger.info(
                    "✅ Completed %s in %.1fs (memory: %.2f -> %.2f GB, delta: %+.2f GB)",
                    exp_name,
                    elapsed,
                    start_memory,
                    result["memory_end_gb"],
                    result["memory_delta_gb"],
                )
                self._print_experiment_summary(exp_name, result)

            except Exception as e:
                logger.exception("❌ Error in %s: %s", exp_name, e)
                self.results[result_key] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "memory_at_error_gb": get_memory_usage_gb(),
                    "elapsed_seconds": time.perf_counter() - start_exp_time,
                }
                self._save_checkpoint()
                self._sample_memory(f"{exp_name}_error")

            # Check memory pressure after each experiment
            if not self._check_memory_pressure():
                logger.warning("Pausing to clear memory pressure...")
                self._cleanup_resources()
                time.sleep(2)  # Brief pause to let system stabilize

    def _print_experiment_summary(self, exp_name: str, result: dict[str, Any]) -> None:
        """Print a brief summary of experiment results."""
        if exp_name == "threshold_grid":
            best = result.get("best", {})
            logger.info(
                "   Best config: template=%.2f, clarify=%.2f, k=%d",
                best.get("template_threshold", 0),
                best.get("clarify_threshold", 0),
                best.get("top_k", 0),
            )
            logger.info(
                "   Template rate: %.1f%%, Avg similarity: %.3f",
                best.get("template_rate", 0) * 100,
                best.get("avg_similarity", 0),
            )

        elif exp_name == "diversity_audit":
            logger.info(
                "   Diversity grade: %s, Unique responses: %d",
                result.get("diversity_grade", "?"),
                result.get("unique_responses", 0),
            )
            logger.info(
                "   Normalized entropy: %.3f",
                result.get("normalized_entropy", 0),
            )

        elif exp_name == "clarify_analysis":
            logger.info(
                "   Clarify rate: %.1f%%, Cases: %d",
                result.get("clarify_rate", 0) * 100,
                result.get("clarify_count", 0),
            )

        elif exp_name == "topk_strategy":
            logger.info(
                "   Best strategy: %s (similarity: %.3f)",
                result.get("best_strategy", "?"),
                result.get("best_avg_similarity", 0),
            )

        elif exp_name == "llm_comparison":
            logger.info(
                "   Template wins: %d, LLM wins: %d, Ties: %d",
                result.get("template_wins", 0),
                result.get("llm_wins", 0),
                result.get("ties", 0),
            )
            logger.info("   Recommendation: %s", result.get("recommendation", "?"))

    def run(self) -> None:
        """Main loop - runs experiments until time expires."""
        logger.info(self._header())

        # Verify database has pairs
        pair_count = self.db.count_pairs()
        if pair_count == 0:
            logger.error("❌ No pairs in database! Run 'jarvis db extract' first.")
            return

        logger.info("📊 Database contains %d pairs", pair_count)

        # Verify FAISS index exists
        active_index = self.db.get_active_index()
        if not active_index:
            logger.error("❌ No active FAISS index! Run 'jarvis db build-index' first.")
            return

        logger.info(
            "📊 Active index: %s (%d vectors)", active_index.version_id, active_index.num_vectors
        )

        while self.should_continue():
            self._run_experiment_cycle()
            self.phase += 1

            if self.should_continue():
                logger.info("")
                logger.info("=" * 60)
                logger.info(
                    "Starting Phase %d with %s remaining (memory: %.2f GB)",
                    self.phase,
                    self.time_remaining(),
                    get_memory_usage_gb(),
                )
                logger.info("=" * 60)

        # Final cleanup
        self._sample_memory("run_complete")
        self._cleanup_resources()
        self._generate_final_report()

    def _header(self) -> str:
        """Generate a header for the run."""
        total_mem, avail_mem = get_system_memory_gb()
        return f"""
{"=" * 70}
🌙 JARVIS Overnight Evaluation Suite
{"=" * 70}
Start time:  {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}
End time:    {self.end_time.strftime("%Y-%m-%d %H:%M:%S")}
Duration:    {(self.end_time - self.start_time).total_seconds() / 3600:.1f} hours
Results dir: {RESULTS_DIR}

System memory: {total_mem:.1f} GB total, {avail_mem:.1f} GB available
Process memory: {get_memory_usage_gb():.2f} GB
Random seed: {self.seed}
Warning threshold: {MEMORY_WARNING_THRESHOLD_GB:.1f} GB
Critical threshold: {MEMORY_CRITICAL_THRESHOLD_GB:.1f} GB
{"=" * 70}
"""

    # =========================================================================
    # Report Generation
    # =========================================================================

    def _generate_final_report(self) -> None:
        """Generate comprehensive morning report."""
        report_path = RESULTS_DIR / f"report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md"
        full_results_path = RESULTS_DIR / "full_results.json"

        # Get index version for reproducibility
        active_index = self.db.get_active_index()

        report = []
        report.append("# JARVIS Overnight Evaluation Report")
        report.append(f"\n**Run Duration**: {self.start_time} → {datetime.now()}")
        report.append(f"**Phases Completed**: {self.phase - 1}")
        report.append(f"**Total Experiments**: {len(self.results)}")
        report.append(f"**Random Seed**: {self.seed}")
        report.append(f"**Index Version**: {active_index.version_id if active_index else 'N/A'}")
        report.append(f"**Index Model**: {active_index.model_name if active_index else 'N/A'}")

        # Executive Summary
        report.append("\n## Executive Summary\n")

        # Best thresholds
        best_grid = None
        for key in sorted(self.results.keys(), reverse=True):
            if key.startswith("threshold_grid_") and "best" in self.results[key]:
                best_grid = self.results[key]["best"]
                break

        if best_grid:
            report.append("### Recommended Configuration")
            report.append("```python")
            report.append(f"TEMPLATE_THRESHOLD = {best_grid['template_threshold']:.3f}")
            report.append(f"CLARIFY_THRESHOLD = {best_grid['clarify_threshold']:.3f}")
            report.append(f"TOP_K = {best_grid['top_k']}")
            report.append("```")
            report.append("\nExpected routing distribution:")
            report.append(f"- Template path: {best_grid['template_rate'] * 100:.1f}%")
            report.append(f"- Generate path: {best_grid['generate_rate'] * 100:.1f}%")
            report.append(f"- Clarify path: {best_grid['clarify_rate'] * 100:.1f}%")
            report.append(f"- Average response similarity: {best_grid['avg_similarity']:.3f}")

        # Diversity grade
        for key in sorted(self.results.keys(), reverse=True):
            if key.startswith("diversity_audit_") and "diversity_grade" in self.results[key]:
                div = self.results[key]
                report.append(f"\n### Response Diversity: Grade {div['diversity_grade']}")
                report.append(f"- Unique responses: {div['unique_responses']}")
                report.append(
                    f"- Top 10 responses cover: {div['top_10_coverage'] * 100:.1f}% of queries"
                )
                report.append(f"- Normalized entropy: {div['normalized_entropy']:.3f}")
                break

        # LLM comparison result
        for key in sorted(self.results.keys(), reverse=True):
            if key.startswith("llm_comparison_") and "recommendation" in self.results[key]:
                llm = self.results[key]
                report.append(f"\n### LLM vs Template: {llm['recommendation']}")
                report.append(f"- Template wins: {llm['template_wins']}")
                report.append(f"- LLM wins: {llm['llm_wins']}")
                report.append(f"- Ties: {llm['ties']}")
                break

        # Top-K strategy
        for key in sorted(self.results.keys(), reverse=True):
            if key.startswith("topk_strategy_") and "best_strategy" in self.results[key]:
                topk = self.results[key]
                report.append(f"\n### Best Selection Strategy: `{topk['best_strategy']}`")
                report.append(f"- Average similarity: {topk['best_avg_similarity']:.3f}")
                break

        # Clarify analysis
        for key in sorted(self.results.keys(), reverse=True):
            if key.startswith("clarify_analysis_") and "recommendations" in self.results[key]:
                clar = self.results[key]
                report.append("\n### Clarify Path Analysis")
                report.append(f"- Clarify rate: {clar['clarify_rate'] * 100:.1f}%")
                report.append("- Recommendations:")
                for rec in clar["recommendations"]:
                    report.append(f"  - {rec}")
                break

        # Memory stats
        report.append("\n### Memory Usage Summary")
        if self._memory_samples:
            mem_values = [m[1] for m in self._memory_samples]
            report.append(f"- Peak memory: {max(mem_values):.2f} GB")
            report.append(f"- Average memory: {np.mean(mem_values):.2f} GB")
            report.append(f"- Final memory: {mem_values[-1]:.2f} GB")
            report.append(f"- Samples recorded: {len(self._memory_samples)}")

        # Action items
        report.append("\n## Recommended Actions\n")
        report.append("1. Update `jarvis/router.py` with optimal thresholds")
        report.append("2. Implement recommended selection strategy")
        report.append("3. Review clarify path recommendations")

        if best_grid:
            report.append("\n### Code Change for router.py")
            report.append("```python")
            report.append("# OLD")
            report.append("TEMPLATE_THRESHOLD = 0.85")
            report.append("CONTEXT_THRESHOLD = 0.60")
            report.append("GENERATE_THRESHOLD = 0.40")
            report.append("")
            report.append(
                f"# NEW (from overnight evaluation {self.start_time.strftime('%Y-%m-%d')})"
            )
            report.append(f"TEMPLATE_THRESHOLD = {best_grid['template_threshold']:.3f}")
            report.append(f"CONTEXT_THRESHOLD = {best_grid.get('context_threshold', 0.60):.3f}")
            report.append(f"GENERATE_THRESHOLD = {best_grid['clarify_threshold']:.3f}")
            report.append("```")

        # Detailed results
        report.append("\n## Detailed Results\n")
        for key in sorted(self.results.keys()):
            report.append(f"### {key}")
            report.append("```json")
            result_copy = self._truncate_for_report(self.results[key])
            report.append(json.dumps(result_copy, indent=2, default=str))
            report.append("```\n")

        # Write report
        report_text = "\n".join(report)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text)

        # Save full results
        with open(full_results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Print summary to console
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 OVERNIGHT EVALUATION COMPLETE")
        logger.info("=" * 70)
        print(report_text[:4000])
        if len(report_text) > 4000:
            logger.info("\n... (full report: %s)", report_path)
        logger.info("=" * 70)
        logger.info("📄 Report saved: %s", report_path)
        logger.info("📄 Full results: %s", full_results_path)

    def _truncate_for_report(self, result: dict[str, Any], max_items: int = 5) -> dict[str, Any]:
        """Truncate large arrays in results for report readability."""
        truncated = {}
        for key, value in result.items():
            if isinstance(value, list) and len(value) > max_items:
                truncated[key] = value[:max_items] + [f"... ({len(value) - max_items} more)"]
            elif isinstance(value, dict):
                truncated[key] = self._truncate_for_report(value, max_items)
            else:
                truncated[key] = value
        return truncated


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JARVIS Overnight Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full 8-hour overnight run
    python scripts/overnight_evaluation.py

    # Quick 15-minute test
    python scripts/overnight_evaluation.py --duration 0.25

    # Run only threshold grid experiment
    python scripts/overnight_evaluation.py --experiment threshold_grid --duration 1

    # Resume from checkpoint after interruption
    python scripts/overnight_evaluation.py --resume

    # Dry run to verify setup
    python scripts/overnight_evaluation.py --dry-run
        """,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_HOURS,
        help=f"Duration in hours (default: {DEFAULT_DURATION_HOURS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "threshold_grid",
            "diversity_audit",
            "clarify_analysis",
            "topk_strategy",
            "llm_comparison",
        ],
        help="Run specific experiment only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify setup without running experiments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: uses current timestamp)",
    )
    args = parser.parse_args()

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("🔍 Dry run - verifying setup...")

        # Check database
        from jarvis.db import get_db

        db = get_db()
        db.init_schema()
        pair_count = db.count_pairs()
        logger.info("   Database: %d pairs", pair_count)
        if pair_count == 0:
            logger.error("   ❌ No pairs found. Run 'jarvis db extract' first.")
            sys.exit(1)

        # Check FAISS index
        active_index = db.get_active_index()
        if active_index:
            logger.info(
                "   FAISS index: %s (%d vectors)", active_index.version_id, active_index.num_vectors
            )
        else:
            logger.error("   ❌ No active index. Run 'jarvis db build-index' first.")
            sys.exit(1)

        # Check embedding model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("   Loading embedding model...")
            model = SentenceTransformer(EMBEDDING_MODEL)
            test_emb = model.encode(["test"])
            logger.info("   Embedding model: OK (dim=%d)", test_emb.shape[1])
            del model
            gc.collect()
        except Exception as e:
            logger.error("   ❌ Embedding model error: %s", e)
            sys.exit(1)

        logger.info("✅ Setup verified. Ready for overnight run.")
        return

    runner = OvernightRunner(
        duration_hours=args.duration,
        resume=args.resume,
        specific_experiment=args.experiment,
        seed=args.seed,
    )
    runner.run()


if __name__ == "__main__":
    main()
