#!/bin/bash
# Full LLM experiment - runs for ~2 hours
# Tests prompting strategies + runs improved evaluation

set -e

echo "========================================"
echo "FULL LLM EXPERIMENT"
echo "Started: $(date)"
echo "========================================"

cd "$(dirname "$0")/.."

# Create results directory
RESULTS_DIR="results/experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Phase 1: Prompt Strategy Experiment (~30-40 min)
echo "========================================"
echo "PHASE 1: Testing 5 prompting strategies"
echo "========================================"
uv run python scripts/prompt_experiment.py \
    --samples 200 \
    --verbose \
    --output-dir "$RESULTS_DIR/prompt_strategies" \
    2>&1 | tee "$RESULTS_DIR/prompt_experiment.log"

echo ""
echo "Phase 1 complete. Moving to Phase 2..."
echo ""

# Phase 2: Improved Evaluation (~60-90 min)
echo "========================================"
echo "PHASE 2: Running improved evaluation"
echo "========================================"
uv run python scripts/improved_llm_eval.py \
    --samples 500 \
    --verbose \
    --output-dir "$RESULTS_DIR/improved_eval" \
    2>&1 | tee "$RESULTS_DIR/improved_eval.log"

echo ""
echo "========================================"
echo "EXPERIMENT COMPLETE"
echo "Finished: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"

# Print quick summary
echo ""
echo "--- QUICK SUMMARY ---"
echo ""
cat "$RESULTS_DIR/prompt_strategies/"summary_*.json 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('PROMPT STRATEGY RESULTS:')
    for strategy, stats in sorted(data.items(), key=lambda x: x[1]['avg_similarity_to_actual'], reverse=True):
        print(f\"  {strategy}: sim={stats['avg_similarity_to_actual']:.3f}, words={stats['avg_word_count']:.1f}, filler={stats['pct_with_filler']:.0f}%\")
except: pass
"

echo ""
cat "$RESULTS_DIR/improved_eval/"summary_*.json 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('IMPROVED EVAL RESULTS:')
    print(f\"  Template wins: {data['template_wins']} ({100*data['template_win_rate']:.1f}%)\")
    print(f\"  LLM wins: {data['llm_wins']} ({100*data['llm_win_rate']:.1f}%)\")
    print(f\"  Similarity to actual - Template: {data['metrics']['similarity_to_actual']['template']:.3f}\")
    print(f\"  Similarity to actual - LLM: {data['metrics']['similarity_to_actual']['llm']:.3f}\")
except: pass
"
