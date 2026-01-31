# Experimental Scripts Framework

This directory contains consolidated experimental scripts using a shared framework.

## Structure

```
scripts/experiments/
├── framework/              # Shared framework code
│   ├── __init__.py
│   ├── base.py            # BaseExperiment class
│   ├── results.py         # ExperimentResult dataclass
│   └── utils.py           # Shared utilities
├── mine_pairs.py          # Consolidated pair mining (was 4 scripts)
├── reply_test.py          # Consolidated reply testing (was 4 scripts)
├── benchmark.py           # Performance benchmarks (was 4 scripts)
├── eval_quality.py        # Quality evaluation (was 3 scripts)
├── template_parameter_sweep.py
├── validate_template_quality.py
├── test_classifier_pipeline.py
├── generate_realistic_report.py
└── prompt_experiment.py
```

## Consolidated Scripts

### mine_pairs.py
Unified pair mining with multiple modes:
- `--mode basic`: Simple frequency-based mining (was mine_response_pairs.py)
- `--mode optimized`: Temporal scoring + multi-message grouping (was mine_response_pairs_optimized.py)
- `--mode semantic`: DBSCAN clustering (was mine_response_pairs_semantic.py)
- `--mode enhanced`: Full features including HDBSCAN, context, coherence (was mine_response_pairs_enhanced.py)

### reply_test.py
Unified reply generation testing:
- `--mode simple`: Basic test with hardcoded examples
- `--mode realistic`: Real iMessage conversations with template matching
- `--mode enhanced`: Context-aware with formality/topic detection

### benchmark.py
Performance and pipeline benchmarks:
- `--type quick`: Fast 200-sample evaluation
- `--type full`: Comprehensive 500-sample evaluation
- `--type router`: Router pipeline evaluation
- `--type pipeline`: Full pipeline with relationship context

### eval_quality.py
Quality evaluation with proper holdout:
- `--type improved`: Semantic similarity + fairness metrics
- `--type llm-judge`: LLM-as-judge evaluation
- `--type proper`: Train/test split per contact

## Usage

All scripts support standard arguments:
```bash
# Basic usage
uv run python scripts/experiments/mine_pairs.py --mode semantic

# With options
uv run python scripts/experiments/benchmark.py --type full --samples 500 --verbose

# Dry run to see what would happen
uv run python scripts/experiments/reply_test.py --mode realistic --dry-run
```

## Legacy Scripts

The original 20 scripts have been consolidated into 4 main scripts + 5 utilities.
If you need the old behavior, use the consolidated versions with appropriate --mode flags.
