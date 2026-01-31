# Decisions and Findings

This document consolidates the practical takeaways from prior research notes and template mining docs.

## Routing and Thresholds

- Template matching uses a default cosine similarity threshold of 0.7 in `models/templates.py`.
- Reply routing in `jarvis/router.py` uses configurable thresholds (`routing.template_threshold`, `routing.context_threshold`, `routing.generate_threshold`).
- Defaults are configured in `jarvis/config.py` and can be overridden by A/B groups.

## Template Matching Performance

- Template-first routing is the fastest path; match success depends on semantic similarity and template coverage.
- Query embedding caching in the template matcher reduces redundant embedding work.
- Cache hit rates and template hit rates should be measured with routing metrics before changing thresholds.

## Observability and Metrics

- Prometheus metrics cover API latency, request counts, and memory usage.
- Routing decisions are logged to `~/.jarvis/metrics.db` for offline analysis.
- Use `scripts/analyze_routing_metrics.py` to compute hit rates and latency percentiles.

## Evaluation and Tuning

- Use `scripts/eval_thresholds.py` to grid-search routing thresholds on holdout data.
- Prefer adjusting thresholds only after measuring real template false positives.

## Template Mining and Coverage

- Template coverage depends on the dataset and contact mix; treat coverage targets as aspirational until measured.
- Keep templates concise and categorize by intent to avoid low-similarity matches.

## Practical Guidance

- If you need higher recall: lower template threshold cautiously and monitor false positives.
- If you need higher precision: raise template threshold and rely more on generation.
- Avoid reranking until metrics show persistent template false positives.
