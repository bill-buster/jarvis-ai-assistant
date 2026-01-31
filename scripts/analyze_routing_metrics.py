"""Analyze routing metrics stored in SQLite."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis.metrics_router import DEFAULT_METRICS_DB_PATH, load_routing_metrics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    position = (p / 100.0) * (len(values_sorted) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values_sorted) - 1)
    if lower == upper:
        return values_sorted[lower]
    weight = position - lower
    return values_sorted[lower] + (values_sorted[upper] - values_sorted[lower]) * weight


def parse_latency(row: dict[str, object]) -> dict[str, float]:
    raw = row.get("latency_json")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze routing metrics")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_METRICS_DB_PATH,
        help="Path to metrics SQLite database",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to analyze")
    args = parser.parse_args()

    rows = load_routing_metrics(args.db_path, limit=args.limit)
    if not rows:
        print(f"No routing metrics found at {args.db_path}")
        return 0

    total = len(rows)
    template_hits = sum(1 for row in rows if row.get("routing_decision") == "template")
    cache_hits = sum(1 for row in rows if row.get("cache_hit") in (1, True))
    embedding_total = sum(int(row.get("embedding_computations", 0)) for row in rows)

    latencies = []
    for row in rows:
        latency = parse_latency(row)
        total_latency = latency.get("total")
        if total_latency is not None:
            latencies.append(total_latency)

    print(f"Routing metrics: {total} requests")
    print(f"Template hit rate: {template_hits / total:.2%}")
    print(f"Cache hit rate: {cache_hits / total:.2%}")
    print(f"Avg embedding computations/request: {embedding_total / total:.2f}")

    if latencies:
        print(
            "Latency (ms): "
            f"p50={percentile(latencies, 50):.2f} "
            f"p95={percentile(latencies, 95):.2f} "
            f"p99={percentile(latencies, 99):.2f}"
        )
    else:
        print("Latency (ms): no samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
