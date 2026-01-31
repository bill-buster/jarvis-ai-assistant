# JARVIS Scripts

Utility scripts for development, benchmarking, and automation.

## Canonical Scripts

- `eval_pipeline.py` - Train/test split and evaluation pipeline
- `score_pair_quality.py` - Pair quality analysis and updates
- `build_embedding_profiles.py` - Embedding profile builder
- `mine_response_pairs_production.py` - Production pair mining
- `validate_templates_human.py` - Interactive template validation
- `generate_report.py` - Benchmark report generator
- `check_gates.py` - Gate status evaluation
- `summarize_phase.py` - Phase summary generator
- `setup_contacts.py` - Contact setup helper
- `overnight_evaluation.py` - Long-running evaluation runner

## Experimental Scripts

Experimental and one-off scripts live in `scripts/experiments/`.
See `scripts/experiments/README.md` for details.

## Context Management

### `summarize_phase.py`

Generate concise phase summaries to reduce context usage in long conversations with Claude Code.

**Usage:**

```bash
# Summarize last 5 commits (default)
python3 scripts/summarize_phase.py --output docs/PHASE_N_SUMMARY.md

# Summarize last N commits
python3 scripts/summarize_phase.py --last-n 10 --phase-name "Phase 3: API Fixes"

# Summarize commit range
python3 scripts/summarize_phase.py --commits "abc123..def456"

# Summarize since reference
python3 scripts/summarize_phase.py --since HEAD~5

# Print to stdout
python3 scripts/summarize_phase.py --last-n 3
```

**When to use:**
- After completing a major phase of work
- Before starting a new conversation thread
- When context token usage is high (>100k)
- To create checkpoints for resuming work later

**Output includes:**
- Commit list with hashes and messages
- Files changed grouped by directory
- Line count statistics (+added/-removed)
- Space for manual "Next Steps" notes

## Benchmarking

### `overnight_eval.sh`

Run all benchmarks sequentially and generate BENCHMARKS.md report.

```bash
# Full evaluation
./scripts/overnight_eval.sh

# Quick mode (fewer iterations for testing)
./scripts/overnight_eval.sh --quick
```

### `generate_report.py`

Generate BENCHMARKS.md from benchmark results.

```bash
python -m scripts.generate_report
```

### `check_gates.py`

Evaluate gate pass/fail status from benchmark results.

```bash
python -m scripts.check_gates
```

## Development Helpers

### `health_check.py` (if exists)

Check project health status.

---

## Adding New Scripts

When adding scripts:
1. Add shebang line: `#!/usr/bin/env python3`
2. Make executable: `chmod +x scripts/your_script.py`
3. Add docstring with usage examples
4. Update this README
5. Add entry to `pyproject.toml` if it's a command
