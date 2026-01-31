"""Evaluate routing thresholds against holdout pairs."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.db import get_db
from jarvis.embedding_adapter import get_embedder
from jarvis.index import TriggerIndexSearcher


@dataclass
class ThresholdResult:
    template_threshold: float
    context_threshold: float
    generate_threshold: float
    total: int
    template_rate: float
    generate_rate: float
    clarify_rate: float
    avg_best_similarity: float
    avg_template_response_similarity: float
    template_false_positive_rate: float


def parse_thresholds(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def response_similarity(embedder, a: str, b: str) -> float:
    embeddings = embedder.encode([a, b], normalize=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def evaluate_thresholds(
    template_thresholds: list[float],
    context_thresholds: list[float],
    generate_thresholds: list[float],
    limit: int,
    min_quality: float,
    response_similarity_threshold: float,
) -> list[ThresholdResult]:
    db = get_db()
    db.init_schema()
    embedder = get_embedder()
    searcher = TriggerIndexSearcher(db)
    holdout_pairs = db.get_holdout_pairs(min_quality=min_quality, limit=limit)

    results: list[ThresholdResult] = []
    for template_threshold in template_thresholds:
        for context_threshold in context_thresholds:
            for generate_threshold in generate_thresholds:
                template_count = 0
                generate_count = 0
                clarify_count = 0
                best_similarities: list[float] = []
                template_response_similarities: list[float] = []
                template_false_positives = 0

                for pair in holdout_pairs:
                    matches = searcher.search_with_pairs(
                        query=pair.trigger_text,
                        k=5,
                        threshold=generate_threshold,
                    )
                    best_similarity = matches[0]["similarity"] if matches else 0.0
                    best_similarities.append(best_similarity)

                    if best_similarity >= template_threshold:
                        template_count += 1
                        response_sim = response_similarity(
                            embedder,
                            pair.response_text,
                            matches[0]["response_text"],
                        )
                        template_response_similarities.append(response_sim)
                        if response_sim < response_similarity_threshold:
                            template_false_positives += 1
                    elif best_similarity >= context_threshold:
                        generate_count += 1
                    elif best_similarity >= generate_threshold:
                        generate_count += 1
                    else:
                        clarify_count += 1

                total = len(holdout_pairs)
                avg_best_similarity = (
                    float(np.mean(best_similarities)) if best_similarities else 0.0
                )
                avg_template_response_similarity = (
                    float(np.mean(template_response_similarities))
                    if template_response_similarities
                    else 0.0
                )
                template_fp_rate = (
                    template_false_positives / template_count if template_count > 0 else 0.0
                )

                results.append(
                    ThresholdResult(
                        template_threshold=template_threshold,
                        context_threshold=context_threshold,
                        generate_threshold=generate_threshold,
                        total=total,
                        template_rate=template_count / total if total else 0.0,
                        generate_rate=generate_count / total if total else 0.0,
                        clarify_rate=clarify_count / total if total else 0.0,
                        avg_best_similarity=avg_best_similarity,
                        avg_template_response_similarity=avg_template_response_similarity,
                        template_false_positive_rate=template_fp_rate,
                    )
                )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate routing thresholds")
    parser.add_argument(
        "--template-thresholds",
        default="0.85,0.90,0.95",
        help="Comma-separated template thresholds",
    )
    parser.add_argument(
        "--context-thresholds",
        default="0.60,0.70,0.80",
        help="Comma-separated context thresholds",
    )
    parser.add_argument(
        "--generate-thresholds",
        default="0.40,0.50,0.60",
        help="Comma-separated generate thresholds",
    )
    parser.add_argument("--limit", type=int, default=500, help="Holdout pair limit")
    parser.add_argument("--min-quality", type=float, default=0.5, help="Min quality filter")
    parser.add_argument(
        "--response-sim-threshold",
        type=float,
        default=0.5,
        help="Response similarity threshold for false positives",
    )
    args = parser.parse_args()

    results = evaluate_thresholds(
        template_thresholds=parse_thresholds(args.template_thresholds),
        context_thresholds=parse_thresholds(args.context_thresholds),
        generate_thresholds=parse_thresholds(args.generate_thresholds),
        limit=args.limit,
        min_quality=args.min_quality,
        response_similarity_threshold=args.response_sim_threshold,
    )

    print(
        "template\tcontext\tgenerate\ttemplate_rate\tgenerate_rate\tclarify_rate\tavg_best_sim\tavg_template_resp_sim\ttemplate_fp_rate"
    )
    for result in results:
        print(
            f"{result.template_threshold:.2f}\t"
            f"{result.context_threshold:.2f}\t"
            f"{result.generate_threshold:.2f}\t"
            f"{result.template_rate:.3f}\t"
            f"{result.generate_rate:.3f}\t"
            f"{result.clarify_rate:.3f}\t"
            f"{result.avg_best_similarity:.3f}\t"
            f"{result.avg_template_response_similarity:.3f}\t"
            f"{result.template_false_positive_rate:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
