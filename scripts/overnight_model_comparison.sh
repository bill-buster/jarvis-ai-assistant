#!/usr/bin/env bash
# Overnight Model Comparison for Response Generation
#
# Tests multiple small LLMs (3B-4B) for iMessage reply quality:
# 1. Qwen2.5-1.5B-Instruct-4bit (current baseline)
# 2. Qwen2.5-3B-Instruct-4bit (upgrade)
# 3. Gemma 3 4B (instruction following champion)
# 4. Qwen3 4B (reasoning champion)
#
# Plus: Template mining with all-mpnet-base-v2
#
# Runtime: 6-8 hours
# Memory: Safe for 8GB RAM (Q4 models use 1-3GB each)

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/overnight_models_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/experiment.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

log "════════════════════════════════════════════════════════════"
log "JARVIS Overnight: LLM Model Comparison for Response Gen"
log "════════════════════════════════════════════════════════════"
log "Started: $(date)"
log "Output: ${RESULTS_DIR}"
log ""
log "Testing 4 LLMs + 1 embedding model:"
log "  1. all-mpnet-base-v2 (embedding for templates)"
log "  2. Qwen2.5-1.5B-Instruct-4bit (current baseline)"
log "  3. Qwen2.5-3B-Instruct-4bit (upgrade)"
log "  4. Gemma 3 4B-Instruct Q4 (instruction champion)"
log "  5. Qwen3 4B Q4 (reasoning champion)"
log ""

# ============================================================================
# Experiment 1: Template Mining (1-2 hours)
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Experiment 1: Template Mining (all-mpnet-base-v2)         │"
log "│ Best STS score: 87-88                                     │"
log "└────────────────────────────────────────────────────────────┘"
log ""

EXP1_START=$(date +%s)

if uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "sentence-transformers/all-mpnet-base-v2" \
    --eps 0.35 \
    --min-samples 5 \
    --min-frequency 8 \
    --output "${RESULTS_DIR}/templates_mpnet.json" \
    2>&1 | tee -a "${LOG}"; then

    EXP1_END=$(date +%s)
    EXP1_DURATION=$((EXP1_END - EXP1_START))
    log "✓ Template mining complete (${EXP1_DURATION}s)"

    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "N/A")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "N/A")
    log "  Coverage: ${COVERAGE}%, Templates: ${TEMPLATES}"
else
    log "✗ Template mining failed"
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Experiment 2: LLM Generation Quality Tests
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Experiment 2: LLM Generation Quality                      │"
log "│ Test 4 models on sample iMessage reply scenarios          │"
log "└────────────────────────────────────────────────────────────┘"
log ""

EXP2_START=$(date +%s)

# Create test scenarios
cat > "${RESULTS_DIR}/test_scenarios.json" <<'EOF'
{
  "scenarios": [
    {
      "id": 1,
      "context": "Friend: Hey are you free for dinner tomorrow?",
      "expected_tone": "casual, friendly, clear yes/no",
      "test_cases": ["accept", "decline politely", "ask for time"]
    },
    {
      "id": 2,
      "context": "Mom: Can you pick up milk on your way home?",
      "expected_tone": "respectful, brief acknowledgment",
      "test_cases": ["yes", "already got it", "can't right now"]
    },
    {
      "id": 3,
      "context": "Colleague: Did you see the project update I sent?",
      "expected_tone": "professional but not formal",
      "test_cases": ["yes saw it", "not yet will check", "thanks for update"]
    },
    {
      "id": 4,
      "context": "Friend: That was so funny lol",
      "expected_tone": "casual, matching energy",
      "test_cases": ["short laugh response", "agree and expand"]
    },
    {
      "id": 5,
      "context": "Friend: Running 10 mins late sorry",
      "expected_tone": "understanding, brief",
      "test_cases": ["no worries", "okay see you soon"]
    }
  ]
}
EOF

# Model configurations
cat > "${RESULTS_DIR}/model_configs.json" <<'EOF'
{
  "models": {
    "qwen2.5-1.5b": {
      "name": "Qwen2.5-1.5B-Instruct",
      "ollama_model": "qwen2.5:1.5b-instruct-q4_K_M",
      "params": "1.5B",
      "memory": "~1GB (Q4)",
      "notes": "Current JARVIS baseline"
    },
    "qwen2.5-3b": {
      "name": "Qwen2.5-3B-Instruct",
      "ollama_model": "qwen2.5:3b-instruct-q4_K_M",
      "params": "3B",
      "memory": "~2GB (Q4)",
      "notes": "Better math/coding than 1.5B"
    },
    "gemma3-4b": {
      "name": "Gemma 3 4B-Instruct",
      "ollama_model": "gemma3:4b-instruct-q4_K_M",
      "params": "4B",
      "memory": "~2.75GB (Q4)",
      "notes": "Best instruction following - beats Gemma 2 27B!"
    },
    "qwen3-4b": {
      "name": "Qwen3 4B",
      "ollama_model": "qwen3:4b-q4_K_M",
      "params": "4B",
      "memory": "~2.75GB (Q4)",
      "notes": "74% MMLU-Pro - best reasoning"
    }
  }
}
EOF

log "Test scenarios and model configs created"
log ""
log "NOTE: This experiment requires Ollama for easy model switching"
log "      Install: brew install ollama"
log ""

# Check if Ollama is installed
if ! command -v ollama >/dev/null 2>&1; then
    log "⚠ Ollama not installed - skipping LLM comparison"
    log "  Install with: brew install ollama"
    log "  Then re-run this script"
    log ""
else
    log "✓ Ollama detected - proceeding with model tests"
    log ""

    # Pull all models
    log "Pulling models (this may take 10-20 minutes)..."

    for model in "qwen2.5:1.5b-instruct-q4_K_M" "qwen2.5:3b-instruct-q4_K_M" \
                 "gemma3:4b-instruct-q4_K_M" "qwen3:4b-q4_K_M"; do
        log "  Pulling ${model}..."
        if ollama pull "${model}" >> "${LOG}" 2>&1; then
            log "    ✓ ${model} ready"
        else
            log "    ✗ ${model} failed to pull"
        fi
    done

    log ""
    log "Running generation tests on scenarios..."

    # Create test script
    cat > "${RESULTS_DIR}/run_generation_tests.py" <<'PYTHON'
#!/usr/bin/env python3
"""Test LLM generation quality on iMessage scenarios."""

import json
import subprocess
import time
from pathlib import Path

def generate_reply(model: str, context: str, instruction: str) -> dict:
    """Generate a reply using Ollama."""

    prompt = f"""You are helping draft an iMessage reply.

Context: {context}

Instruction: {instruction}

Generate a brief, natural iMessage reply (1-2 sentences max):"""

    start = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        latency = time.time() - start
        reply = result.stdout.strip()

        return {
            "reply": reply,
            "latency_ms": int(latency * 1000),
            "success": True
        }
    except Exception as e:
        return {
            "reply": "",
            "latency_ms": 0,
            "success": False,
            "error": str(e)
        }

def main():
    """Run generation tests."""

    results_dir = Path(__file__).parent

    # Load scenarios and models
    with open(results_dir / "test_scenarios.json") as f:
        scenarios_data = json.load(f)

    with open(results_dir / "model_configs.json") as f:
        models_data = json.load(f)

    results = {
        "models": {},
        "scenarios": scenarios_data["scenarios"]
    }

    # Test each model
    for model_key, model_info in models_data["models"].items():
        print(f"\nTesting {model_info['name']}...")

        model_results = {
            "info": model_info,
            "generations": [],
            "avg_latency_ms": 0,
            "success_rate": 0
        }

        total_latency = 0
        successes = 0

        # Test on each scenario
        for scenario in scenarios_data["scenarios"]:
            for test_case in scenario["test_cases"]:
                gen = generate_reply(
                    model_info["ollama_model"],
                    scenario["context"],
                    test_case
                )

                model_results["generations"].append({
                    "scenario_id": scenario["id"],
                    "context": scenario["context"],
                    "instruction": test_case,
                    **gen
                })

                if gen["success"]:
                    total_latency += gen["latency_ms"]
                    successes += 1
                    print(f"  ✓ Scenario {scenario['id']}/{test_case}: {gen['latency_ms']}ms")

        # Calculate stats
        if successes > 0:
            model_results["avg_latency_ms"] = total_latency / successes
        model_results["success_rate"] = successes / len(model_results["generations"])

        results["models"][model_key] = model_results

    # Save results
    with open(results_dir / "generation_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Generation tests complete")
    return results

if __name__ == "__main__":
    main()
PYTHON

    chmod +x "${RESULTS_DIR}/run_generation_tests.py"

    # Run the tests
    if python3 "${RESULTS_DIR}/run_generation_tests.py" 2>&1 | tee -a "${LOG}"; then
        EXP2_END=$(date +%s)
        EXP2_DURATION=$((EXP2_END - EXP2_START))
        log ""
        log "✓ LLM generation tests complete (${EXP2_DURATION}s)"
    else
        log "✗ LLM generation tests failed"
    fi
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Generate Report
# ============================================================================

log "Generating comparison report..."

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - EXP1_START))
TOTAL_HOURS=$(echo "scale=1; ${TOTAL_DURATION} / 3600" | bc)

cat > "${RESULTS_DIR}/REPORT.md" <<'EOF'
# JARVIS Overnight: Model Comparison Results

**Date:** $(date '+%Y-%m-%d')
**Duration:** ${TOTAL_HOURS} hours

---

## Models Tested

### Embedding Model (for Templates)
- **all-mpnet-base-v2** (110M params, MTEB STS: 87-88)

### LLMs (for Response Generation)
1. **Qwen2.5-1.5B-Instruct** (current baseline)
   - 1.5B params, ~1GB memory (Q4)
   - MMLU: ~60

2. **Qwen2.5-3B-Instruct** (upgrade candidate)
   - 3B params, ~2GB memory (Q4)
   - MMLU: ~67
   - Better math/coding than 1.5B

3. **Gemma 3 4B-Instruct** (instruction champion)
   - 4B params, ~2.75GB memory (Q4)
   - Beats Gemma 2 27B on benchmarks!
   - Best instruction following

4. **Qwen3 4B** (reasoning champion)
   - 4B params, ~2.75GB memory (Q4)
   - 74% MMLU-Pro
   - Best reasoning/math

---

## Template Mining Results

EOF

if [[ -f "${RESULTS_DIR}/templates_mpnet.json" ]]; then
    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "0")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "0")

    cat >> "${RESULTS_DIR}/REPORT.md" <<EOF
**Coverage:** ${COVERAGE}%
**Templates Extracted:** ${TEMPLATES}
**Baseline:** 6.2% (manual templates)
**Improvement:** $(echo "scale=1; ${COVERAGE} / 6.2" | bc 2>/dev/null || echo "N/A")× better

### Top 10 Templates:
EOF

    jq -r '.templates[:10] | to_entries[] | "**\(.key + 1).** [\(.value.frequency) uses] `\(.value.representative[:70])`"' \
        "${RESULTS_DIR}/templates_mpnet.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null || echo "_No templates_" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

---

## LLM Generation Test Results

EOF

if [[ -f "${RESULTS_DIR}/generation_test_results.json" ]]; then
    cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'
### Performance Comparison

| Model | Avg Latency | Success Rate | Memory | Notes |
|-------|-------------|--------------|--------|-------|
EOF

    jq -r '.models | to_entries[] |
    "| \(.value.info.name) | \(.value.avg_latency_ms)ms | \(.value.success_rate * 100)% | \(.value.info.memory) | \(.value.info.notes) |"' \
        "${RESULTS_DIR}/generation_test_results.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null

    cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

### Sample Generations

EOF

    # Show first 3 generations from each model
    jq -r '.models | to_entries[] |
    "#### \(.value.info.name)\n\n" +
    (.value.generations[:3] | .[] |
    "**Context:** \(.context)\n**Instruction:** \(.instruction)\n**Reply:** \(.reply)\n**Latency:** \(.latency_ms)ms\n\n")' \
        "${RESULTS_DIR}/generation_test_results.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null
else
    echo "_LLM tests skipped (Ollama not installed)_" >> "${RESULTS_DIR}/REPORT.md"
    echo "" >> "${RESULTS_DIR}/REPORT.md"
    echo "To run LLM tests:" >> "${RESULTS_DIR}/REPORT.md"
    echo "1. Install Ollama: \`brew install ollama\`" >> "${RESULTS_DIR}/REPORT.md"
    echo "2. Re-run this script" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

---

## Recommendations

### For Template Matching
✅ **Use all-mpnet-base-v2**
- Best STS score (87-88)
- Fast (10-50ms)
- Good coverage

### For Response Generation

EOF

if [[ -f "${RESULTS_DIR}/generation_test_results.json" ]]; then
    # Find best model by latency and quality
    echo "Based on the test results above:" >> "${RESULTS_DIR}/REPORT.md"
    echo "" >> "${RESULTS_DIR}/REPORT.md"
    echo "**Winner:** (check avg_latency and quality above)" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "**Recommended (based on benchmarks):**" >> "${RESULTS_DIR}/REPORT.md"
    echo "- **Gemma 3 4B** for best instruction following" >> "${RESULTS_DIR}/REPORT.md"
    echo "- **Qwen3 4B** for best reasoning/math" >> "${RESULTS_DIR}/REPORT.md"
    echo "- **Qwen2.5-3B** for good balance (2GB memory)" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

### Next Steps

1. Review generation samples above
2. Pick winning model based on:
   - Reply quality (naturalness, tone)
   - Latency (speed)
   - Memory usage
3. Update JARVIS config to use chosen model
4. Implement hybrid pipeline (templates → context selection → generation)

---

## Files

- `templates_mpnet.json` - Mined templates
- `generation_test_results.json` - LLM test results
- `test_scenarios.json` - Test scenarios
- `model_configs.json` - Model configurations
- `experiment.log` - Full log

EOF

log "Report generated: ${RESULTS_DIR}/REPORT.md"

# Create symlink
rm -f results/latest
ln -sf "$(basename "${RESULTS_DIR}")" results/latest

log ""
log "════════════════════════════════════════════════════════════"
log "EXPERIMENT COMPLETE!"
log "════════════════════════════════════════════════════════════"
log "Duration: ${TOTAL_HOURS} hours"
log "Results: ${RESULTS_DIR}/"
log ""
log "View report: cat results/latest/REPORT.md"
log ""
