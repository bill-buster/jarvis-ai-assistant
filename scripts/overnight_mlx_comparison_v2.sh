#!/usr/bin/env bash
# Overnight MLX Model Comparison v2 (Fixed Model Paths)
#
# Uses JARVIS's existing MLX infrastructure to test models:
# 1. Template mining (all-mpnet-base-v2)
# 2. LLM comparison (corrected model paths)
#
# Models tested:
# - Qwen2.5-1.5B-Instruct-4bit (current baseline)
# - SmolLM2-1.7B-Instruct (replaces non-existent SmolLM3-3B)
# - Qwen2.5-3B-Instruct-4bit (middle tier)
# - Phi-3-Mini-4K-Instruct-4bit (fastest)
# - Gemma 3 4B-Instruct-4bit (best instruction following)
# - Qwen3-4B-Instruct (FIXED PATH - thinking mode)
#
# Runtime: 4-6 hours
# Memory: Safe for 8GB RAM

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/overnight_mlx_v2_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/experiment.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

log "════════════════════════════════════════════════════════════"
log "JARVIS Overnight v2: MLX Model Comparison (FIXED PATHS)"
log "════════════════════════════════════════════════════════════"
log "Started: $(date)"
log "Using: JARVIS's existing MLX infrastructure"
log "Output: ${RESULTS_DIR}"
log ""
log "Tests:"
log "  Part 1: Template Mining (all-mpnet-base-v2)"
log "  Part 2: 6 LLM Models (corrected paths):"
log "    • Qwen2.5-1.5B (current baseline)"
log "    • SmolLM2-1.7B (replaces SmolLM3)"
log "    • Qwen2.5-3B (middle tier)"
log "    • Phi-3-Mini (fastest - 28 tok/s)"
log "    • Gemma 3 4B (instruction following)"
log "    • Qwen3-4B (FIXED - thinking mode)"
log ""

# ============================================================================
# Experiment 1: Template Mining (1-2 hours)
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Experiment 1: Template Mining                              │"
log "│ Model: all-mpnet-base-v2 (Best STS: 87-88)                │"
log "└────────────────────────────────────────────────────────────┘"
log ""

EXP1_START=$(date +%s)

if uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "sentence-transformers/all-mpnet-base-v2" \
    --eps 0.35 \
    --min-samples 5 \
    --min-frequency 8 \
    --output "${RESULTS_DIR}/templates.json" \
    2>&1 | tee -a "${LOG}"; then

    EXP1_END=$(date +%s)
    EXP1_DURATION=$((EXP1_END - EXP1_START))
    log "✓ Template mining complete (${EXP1_DURATION}s)"

    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates.json" 2>/dev/null || echo "N/A")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates.json" 2>/dev/null || echo "N/A")
    log "  Coverage: ${COVERAGE}%, Templates: ${TEMPLATES}"
else
    log "✗ Template mining failed"
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Experiment 2: MLX Model Quality Comparison (4-6 hours)
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Experiment 2: MLX Model Quality Tests (FIXED PATHS)       │"
log "│ Testing 6 models on iMessage reply scenarios              │"
log "└────────────────────────────────────────────────────────────┘"
log ""

EXP2_START=$(date +%s)

# Create test script with CORRECTED model paths
cat > "${RESULTS_DIR}/test_mlx_models.py" <<'PYTHON'
#!/usr/bin/env python3
"""Test MLX models for response generation quality (v2 - FIXED PATHS)."""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.loader import MLXModelLoader, ModelConfig
from models.registry import MODEL_REGISTRY


# Test scenarios
SCENARIOS = [
    {
        "id": 1,
        "context": "Friend: Hey are you free for dinner tomorrow?",
        "instructions": [
            "Say yes enthusiastically",
            "Decline politely and suggest another time",
            "Ask what time works"
        ]
    },
    {
        "id": 2,
        "context": "Mom: Can you pick up milk on your way home?",
        "instructions": [
            "Say yes",
            "Say you already got it",
            "Say you can't right now"
        ]
    },
    {
        "id": 3,
        "context": "Colleague: Did you see the project update I sent?",
        "instructions": [
            "Say yes and acknowledge",
            "Say not yet but will check soon",
            "Thank them for the update"
        ]
    },
    {
        "id": 4,
        "context": "Friend: That was so funny lol",
        "instructions": [
            "Short laugh response",
            "Agree and add comment"
        ]
    },
    {
        "id": 5,
        "context": "Friend: Running 10 mins late sorry",
        "instructions": [
            "Say no worries",
            "Say okay and you'll see them soon"
        ]
    }
]

# Models to test - 6 models for 8GB RAM (CORRECTED PATHS)
MODELS = {
    "qwen-1.5b": {
        "name": "Qwen2.5-1.5B-Instruct",
        "model_id": "qwen-1.5b",  # From registry
        "params": "1.5B",
        "memory": "~1.5GB",
        "notes": "Current JARVIS baseline"
    },
    "smollm2-1.7b": {
        "name": "SmolLM2-1.7B-Instruct",
        "model_path": "mlx-community/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "memory": "~1GB",
        "notes": "Latest SmolLM - compact and efficient (not quantized)"
    },
    "qwen-3b": {
        "name": "Qwen2.5-3B-Instruct",
        "model_id": "qwen-3b",  # From registry
        "params": "3B",
        "memory": "~2GB",
        "notes": "Middle tier - proven Qwen baseline"
    },
    "phi3-mini": {
        "name": "Phi-3-Mini-4K-Instruct",
        "model_path": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "params": "3.8B",
        "memory": "~2.5GB",
        "notes": "FASTEST (28 tok/s), excellent for coding"
    },
    "gemma3-4b": {
        "name": "Gemma 3 4B-Instruct",
        "model_path": "mlx-community/gemma-3-4b-it-4bit",
        "params": "4B",
        "memory": "~2.75GB",
        "notes": "Best instruction following, beats Gemma 2 27B"
    },
    "qwen3-4b": {
        "name": "Qwen3 4B-Instruct",
        "model_path": "Qwen/Qwen3-4B-MLX-4bit",  # FIXED PATH
        "params": "4B",
        "memory": "~2.75GB",
        "notes": "HIGHEST MMLU (74%), with thinking mode"
    }
}


def generate_reply(loader: MLXModelLoader, context: str, instruction: str) -> dict:
    """Generate a reply using MLX."""

    prompt = f"""You are helping draft an iMessage reply. Be brief and natural (1-2 sentences max).

Context: {context}

Instruction: {instruction}

Reply:"""

    start = time.time()

    try:
        # Load model first
        loader.load()

        # Generate with conservative settings using generate_sync()
        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )

        return {
            "reply": result.text,
            "latency_ms": int(result.generation_time_ms),
            "tokens": result.tokens_generated,
            "success": True
        }
    except Exception as e:
        return {
            "reply": "",
            "latency_ms": 0,
            "tokens": 0,
            "success": False,
            "error": str(e)
        }


def test_model(model_key: str, model_info: dict) -> dict:
    """Test a single model on all scenarios."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_info['name']}")
    print(f"{'='*60}\n")

    # Create config
    if "model_id" in model_info:
        config = ModelConfig(model_id=model_info["model_id"])
    else:
        config = ModelConfig(model_path=model_info["model_path"])

    # Load model
    print(f"Loading model...")
    start = time.time()

    try:
        loader = MLXModelLoader(config)
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.1f}s\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return {
            "info": model_info,
            "load_error": str(e),
            "generations": []
        }

    # Test on scenarios
    results = {
        "info": model_info,
        "load_time_s": load_time,
        "generations": [],
        "stats": {
            "total_tests": 0,
            "successes": 0,
            "failures": 0,
            "avg_latency_ms": 0,
            "avg_tokens": 0
        }
    }

    total_latency = 0
    total_tokens = 0
    successes = 0

    for scenario in SCENARIOS:
        for instruction in scenario["instructions"]:
            print(f"Scenario {scenario['id']}: {instruction[:50]}...")

            gen = generate_reply(loader, scenario["context"], instruction)

            results["generations"].append({
                "scenario_id": scenario["id"],
                "context": scenario["context"],
                "instruction": instruction,
                **gen
            })

            results["stats"]["total_tests"] += 1

            if gen["success"]:
                successes += 1
                total_latency += gen["latency_ms"]
                total_tokens += gen["tokens"]
                print(f"  ✓ {gen['latency_ms']}ms | {gen['tokens']} tokens")
                print(f"  Reply: {gen['reply'][:80]}")
            else:
                results["stats"]["failures"] += 1
                print(f"  ✗ Failed: {gen.get('error', 'Unknown')}")

            print()

    # Calculate stats
    results["stats"]["successes"] = successes

    if successes > 0:
        results["stats"]["avg_latency_ms"] = total_latency / successes
        results["stats"]["avg_tokens"] = total_tokens / successes

    # Unload model
    loader.unload()
    print(f"\n✓ Model unloaded\n")

    return results


def main():
    """Run all model tests."""

    results_dir = Path(__file__).parent

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }

    # Test each model
    for model_key, model_info in MODELS.items():
        try:
            model_results = test_model(model_key, model_info)
            all_results["models"][model_key] = model_results
        except Exception as e:
            print(f"\n✗ Error testing {model_info['name']}: {e}\n")
            all_results["models"][model_key] = {
                "info": model_info,
                "error": str(e),
                "generations": []
            }

    # Save results
    output_file = results_dir / "mlx_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
PYTHON

chmod +x "${RESULTS_DIR}/test_mlx_models.py"

log "Running MLX model comparison tests..."
log "This will take 4-6 hours (testing 6 models)"
log ""
log "NOTE: If any model fails, the script will continue with the next model"
log ""

# Run with error handling - don't stop if one model fails
if uv run python "${RESULTS_DIR}/test_mlx_models.py" 2>&1 | tee -a "${LOG}"; then
    EXP2_END=$(date +%s)
    EXP2_DURATION=$((EXP2_END - EXP2_START))
    log ""
    log "✓ MLX comparison complete (${EXP2_DURATION}s)"
else
    EXP2_END=$(date +%s)
    EXP2_DURATION=$((EXP2_END - EXP2_START))
    log "⚠ MLX comparison had errors but may have partial results (${EXP2_DURATION}s)"
    log "  Check ${RESULTS_DIR}/mlx_comparison_results.json for details"
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Generate Report
# ============================================================================

log "Generating comprehensive report..."

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - EXP1_START))
TOTAL_HOURS=$(echo "scale=1; ${TOTAL_DURATION} / 3600" | bc)

cat > "${RESULTS_DIR}/REPORT.md" <<EOF
# JARVIS Overnight Results v2: MLX Model Comparison (FIXED)

**Date:** $(date '+%Y-%m-%d')
**Duration:** ${TOTAL_HOURS} hours
**Infrastructure:** MLX (Apple Silicon optimized)

---

## Models Tested (6 models - CORRECTED PATHS)

1. **Qwen2.5-1.5B** (1.5B params, ~1.5GB) - Current baseline
2. **SmolLM2-1.7B** (1.7B params, ~1GB) - Latest SmolLM (replaces non-existent SmolLM3)
3. **Qwen2.5-3B** (3B params, ~2GB) - Middle tier
4. **Phi-3-Mini** (3.8B params, ~2.5GB) - Fastest (28 tok/s)
5. **Gemma 3 4B** (4B params, ~2.75GB) - Best instruction following
6. **Qwen3 4B** (4B params, ~2.75GB) - Thinking mode enabled

---

## Template Mining Results

EOF

if [[ -f "${RESULTS_DIR}/templates.json" ]]; then
    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates.json" 2>/dev/null || echo "0")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates.json" 2>/dev/null || echo "0")
    MESSAGES=$(jq -r '.stats.total_messages' "${RESULTS_DIR}/templates.json" 2>/dev/null || echo "0")

    cat >> "${RESULTS_DIR}/REPORT.md" <<EOF
**Messages Analyzed:** ${MESSAGES}
**Coverage:** ${COVERAGE}%
**Templates Extracted:** ${TEMPLATES}
**Baseline:** 6.2% (manual templates)
**Improvement:** $(echo "scale=1; ${COVERAGE} / 6.2" | bc 2>/dev/null || echo "N/A")× better

### Top 10 Templates:
EOF

    jq -r '.templates[:10] | to_entries[] | "**\(.key + 1).** [\(.value.frequency) uses] `\(.value.representative[:70])`"' \
        "${RESULTS_DIR}/templates.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null || echo "_No templates_" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

---

## MLX Model Comparison Results

EOF

if [[ -f "${RESULTS_DIR}/mlx_comparison_results.json" ]]; then
    cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'
### Performance Summary

| Model | Params | Memory | Avg Latency | Successes | Avg Tokens | Notes |
|-------|--------|--------|-------------|-----------|------------|-------|
EOF

    jq -r '.models | to_entries[] |
    "| \(.value.info.name) | \(.value.info.params) | \(.value.info.memory) | \(.value.stats.avg_latency_ms // 0 | floor)ms | \(.value.stats.successes // 0)/\(.value.stats.total_tests // 0) | \(.value.stats.avg_tokens // 0 | floor) | \(.value.info.notes) |"' \
        "${RESULTS_DIR}/mlx_comparison_results.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null

    cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

### Sample Generations

EOF

    # Show sample generations from each model
    jq -r '.models | to_entries[] |
    "#### \(.value.info.name)\n\n**Load Time:** \(.value.load_time_s // 0 | floor)s\n\n" +
    (.value.generations[:3] | .[] | select(.success) |
    "**Context:** \(.context)\n**Instruction:** \(.instruction)\n**Reply:** \(.reply)\n**Latency:** \(.latency_ms)ms\n\n")' \
        "${RESULTS_DIR}/mlx_comparison_results.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null || echo "_No generations_" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "_MLX tests incomplete_" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

---

## Winner & Recommendations

### For Template Matching
✅ **all-mpnet-base-v2**
- Best STS score (87-88)
- Fast embedding generation

### For Response Generation
Review the results above and pick based on:
- **Quality:** Which replies sound most natural?
- **Speed:** Acceptable latency for your use case?
- **Memory:** Fits in your system?

### Observations
- **Qwen3-4B** has thinking mode enabled - may generate longer reasoning
- **SmolLM2-1.7B** is not 4-bit quantized - expect slower performance
- **Phi-3-Mini** tends to be verbose and may hit the 50 token limit

### Next Steps

1. Review sample generations above
2. Pick winning model for JARVIS
3. Update `models/registry.py` if needed
4. Implement hybrid pipeline:
   ```
   Template Match (10ms, 30-50% hits)
        ↓ if no match
   Context Selection (30ms via embedding)
        ↓
   LLM Generation (1-2s with selected context)
   ```

---

## Files

- `templates.json` - Mined templates
- `mlx_comparison_results.json` - Full MLX test results
- `test_mlx_models.py` - Test script (v2 - fixed paths)
- `experiment.log` - Full execution log

EOF

log "Report generated: ${RESULTS_DIR}/REPORT.md"

# Create symlink
rm -f results/latest
ln -sf "$(basename "${RESULTS_DIR}")" results/latest

log ""
log "════════════════════════════════════════════════════════════"
log "EXPERIMENT COMPLETE!"
log "════════════════════════════════════════════════════════════"

# Summary of what succeeded/failed
log ""
log "Summary:"

if [[ -f "${RESULTS_DIR}/templates.json" ]]; then
    TMPL_SUCCESS=$(jq -r '.stats.templates_extracted // 0' "${RESULTS_DIR}/templates.json" 2>/dev/null)
    if [[ "${TMPL_SUCCESS}" != "0" ]]; then
        log "  ✓ Template mining: ${TMPL_SUCCESS} templates extracted"
    else
        log "  ✗ Template mining: failed or incomplete"
    fi
else
    log "  ✗ Template mining: failed"
fi

if [[ -f "${RESULTS_DIR}/mlx_comparison_results.json" ]]; then
    MODELS_TESTED=$(jq -r '.models | length' "${RESULTS_DIR}/mlx_comparison_results.json" 2>/dev/null || echo "0")
    MODELS_SUCCESS=$(jq -r '[.models[] | select(.stats.successes > 0)] | length' "${RESULTS_DIR}/mlx_comparison_results.json" 2>/dev/null || echo "0")
    log "  ✓ LLM tests: ${MODELS_SUCCESS}/${MODELS_TESTED} models completed successfully"

    # List failed models if any
    if [[ "${MODELS_SUCCESS}" != "${MODELS_TESTED}" ]]; then
        log ""
        log "  Failed models:"
        jq -r '.models | to_entries[] | select(.value.load_error or .value.error) | "    • \(.value.info.name): \(.value.load_error // .value.error | split("\n")[0])"' \
            "${RESULTS_DIR}/mlx_comparison_results.json" 2>/dev/null | head -10 | tee -a "${LOG}" || true
    fi
else
    log "  ✗ LLM tests: incomplete or failed"
fi

log ""
log "Duration: ${TOTAL_HOURS} hours"
log "Results: ${RESULTS_DIR}/"
log ""
log "View report: cat results/latest/REPORT.md"
log ""
