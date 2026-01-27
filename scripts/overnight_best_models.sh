#!/usr/bin/env bash
# Overnight Experiment: Best Model for Each Task
#
# Tests the top-scoring model for each JARVIS use case:
# 1. all-mpnet-base-v2        (STS: 87-88)  - Template Mining
# 2. nvidia/NV-Embed-v2       (Retrieval: 62.65) - Semantic Search
# 3. Qwen/Qwen3-Embedding-8B  (Classification: ~76) - Intent Routing
#
# Runtime: 6-8 hours
# Output: results/overnight_YYYYMMDD_HHMMSS/

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/overnight_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/experiment.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

log "════════════════════════════════════════════════════════════"
log "JARVIS Overnight Experiment: Best Models on ALL Messages"
log "════════════════════════════════════════════════════════════"
log "Started: $(date)"
log "Output: ${RESULTS_DIR}"
log ""

# ============================================================================
# Task 1: Template Mining with all-mpnet-base-v2 (Best STS: 87-88)
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Task 1: Template Mining with all-mpnet-base-v2            │"
log "│ MTEB STS Score: 87-88 (HIGHEST)                           │"
log "│ Expected: Better semantic matching for templates          │"
log "└────────────────────────────────────────────────────────────┘"
log ""

TASK1_START=$(date +%s)

log "Mining templates from ALL messages..."
log "Model: sentence-transformers/all-mpnet-base-v2"
log "Parameters: eps=0.35, min_samples=5, min_frequency=8"
log ""

if uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "sentence-transformers/all-mpnet-base-v2" \
    --eps 0.35 \
    --min-samples 5 \
    --min-frequency 8 \
    --output "${RESULTS_DIR}/templates_mpnet.json" \
    2>&1 | tee -a "${LOG}"; then

    TASK1_END=$(date +%s)
    TASK1_DURATION=$((TASK1_END - TASK1_START))

    log ""
    log "✓ Task 1 Complete (Duration: ${TASK1_DURATION}s)"

    # Extract metrics
    if [[ -f "${RESULTS_DIR}/templates_mpnet.json" ]]; then
        COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "N/A")
        TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "N/A")
        MESSAGES=$(jq -r '.stats.total_messages' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "N/A")

        log "  Messages analyzed: ${MESSAGES}"
        log "  Templates extracted: ${TEMPLATES}"
        log "  Coverage: ${COVERAGE}%"
    fi
else
    log "✗ Task 1 Failed"
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Task 2: Template Mining with NV-Embed-v2 (Best Retrieval: 62.65)
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Task 2: Template Mining with NV-Embed-v2                  │"
log "│ MTEB Retrieval Score: 62.65 (HIGHEST)                     │"
log "│ Expected: Better at finding similar message patterns      │"
log "└────────────────────────────────────────────────────────────┘"
log ""

TASK2_START=$(date +%s)

log "Mining templates from ALL messages..."
log "Model: nvidia/NV-Embed-v2"
log "Parameters: eps=0.35, min_samples=5, min_frequency=8"
log ""

if uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "nvidia/NV-Embed-v2" \
    --eps 0.35 \
    --min-samples 5 \
    --min-frequency 8 \
    --output "${RESULTS_DIR}/templates_nvembed.json" \
    2>&1 | tee -a "${LOG}"; then

    TASK2_END=$(date +%s)
    TASK2_DURATION=$((TASK2_END - TASK2_START))

    log ""
    log "✓ Task 2 Complete (Duration: ${TASK2_DURATION}s)"

    # Extract metrics
    if [[ -f "${RESULTS_DIR}/templates_nvembed.json" ]]; then
        COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "N/A")
        TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "N/A")
        MESSAGES=$(jq -r '.stats.total_messages' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "N/A")

        log "  Messages analyzed: ${MESSAGES}"
        log "  Templates extracted: ${TEMPLATES}"
        log "  Coverage: ${COVERAGE}%"
    fi
else
    log "✗ Task 2 Failed"
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Task 3: Template Mining with Qwen3-Embedding-8B (Best Classification: ~76)
# ============================================================================

log "┌────────────────────────────────────────────────────────────┐"
log "│ Task 3: Template Mining with Qwen3-Embedding-8B           │"
log "│ MTEB Classification Score: ~76 (HIGHEST)                  │"
log "│ Expected: Better at distinguishing message categories     │"
log "└────────────────────────────────────────────────────────────┘"
log ""

TASK3_START=$(date +%s)

log "Mining templates from ALL messages..."
log "Model: Qwen/Qwen3-Embedding-8B"
log "Parameters: eps=0.35, min_samples=5, min_frequency=8"
log ""

if uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "Qwen/Qwen3-Embedding-8B" \
    --eps 0.35 \
    --min-samples 5 \
    --min-frequency 8 \
    --output "${RESULTS_DIR}/templates_qwen3.json" \
    2>&1 | tee -a "${LOG}"; then

    TASK3_END=$(date +%s)
    TASK3_DURATION=$((TASK3_END - TASK3_START))

    log ""
    log "✓ Task 3 Complete (Duration: ${TASK3_DURATION}s)"

    # Extract metrics
    if [[ -f "${RESULTS_DIR}/templates_qwen3.json" ]]; then
        COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "N/A")
        TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "N/A")
        MESSAGES=$(jq -r '.stats.total_messages' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "N/A")

        log "  Messages analyzed: ${MESSAGES}"
        log "  Templates extracted: ${TEMPLATES}"
        log "  Coverage: ${COVERAGE}%"
    fi
else
    log "✗ Task 3 Failed"
fi

log ""
log "════════════════════════════════════════════════════════════"
log ""

# ============================================================================
# Generate Comparison Report
# ============================================================================

log "Generating comparison report..."

TOTAL_END=$(date +%s)
TOTAL_START=$(date -r "${RESULTS_DIR}" +%s 2>/dev/null || echo "${TASK1_START}")
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$(echo "scale=1; ${TOTAL_DURATION} / 3600" | bc)

cat > "${RESULTS_DIR}/REPORT.md" <<'EOF'
# JARVIS Overnight Experiment Results
## Best Models on All Messages

**Date:** $(date '+%Y-%m-%d')
**Duration:** ${TOTAL_HOURS} hours
**Total Messages:** $(jq -r '.stats.total_messages' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "N/A")

---

## Experiment Overview

Tested the **top-scoring model** for each JARVIS task on ALL iMessage messages:

| Model | MTEB Score | Strength | Use Case |
|-------|------------|----------|----------|
| all-mpnet-base-v2 | **87-88** STS | Semantic similarity | Template matching |
| NV-Embed-v2 | **62.65** Retrieval | Finding similar texts | Semantic search |
| Qwen3-Embedding-8B | **~76** Classification | Categorizing texts | Intent routing |

---

## Results Summary

### Template Mining Performance

| Model | Coverage | Templates Extracted | Top Pattern Frequency | Avg Patterns/Template |
|-------|----------|---------------------|----------------------|----------------------|
EOF

# Add mpnet results
if [[ -f "${RESULTS_DIR}/templates_mpnet.json" ]]; then
    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "0")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "0")
    TOTAL_PATTERNS=$(jq -r '.stats.total_patterns // 0' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "0")
    TOP_FREQ=$(jq -r '.templates[0].frequency // 0' "${RESULTS_DIR}/templates_mpnet.json" 2>/dev/null || echo "0")

    if [[ "${TEMPLATES}" != "0" ]]; then
        AVG_PATTERNS=$(echo "scale=1; ${TOTAL_PATTERNS} / ${TEMPLATES}" | bc 2>/dev/null || echo "0")
    else
        AVG_PATTERNS="0"
    fi

    printf "| all-mpnet-base-v2 | %.1f%% | %s | %s | %s |\n" "${COVERAGE}" "${TEMPLATES}" "${TOP_FREQ}" "${AVG_PATTERNS}" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "| all-mpnet-base-v2 | Failed | Failed | Failed | Failed |" >> "${RESULTS_DIR}/REPORT.md"
fi

# Add NV-Embed results
if [[ -f "${RESULTS_DIR}/templates_nvembed.json" ]]; then
    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "0")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "0")
    TOTAL_PATTERNS=$(jq -r '.stats.total_patterns // 0' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "0")
    TOP_FREQ=$(jq -r '.templates[0].frequency // 0' "${RESULTS_DIR}/templates_nvembed.json" 2>/dev/null || echo "0")

    if [[ "${TEMPLATES}" != "0" ]]; then
        AVG_PATTERNS=$(echo "scale=1; ${TOTAL_PATTERNS} / ${TEMPLATES}" | bc 2>/dev/null || echo "0")
    else
        AVG_PATTERNS="0"
    fi

    printf "| NV-Embed-v2 | %.1f%% | %s | %s | %s |\n" "${COVERAGE}" "${TEMPLATES}" "${TOP_FREQ}" "${AVG_PATTERNS}" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "| NV-Embed-v2 | Failed | Failed | Failed | Failed |" >> "${RESULTS_DIR}/REPORT.md"
fi

# Add Qwen3 results
if [[ -f "${RESULTS_DIR}/templates_qwen3.json" ]]; then
    COVERAGE=$(jq -r '.stats.coverage * 100' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "0")
    TEMPLATES=$(jq -r '.stats.templates_extracted' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "0")
    TOTAL_PATTERNS=$(jq -r '.stats.total_patterns // 0' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "0")
    TOP_FREQ=$(jq -r '.templates[0].frequency // 0' "${RESULTS_DIR}/templates_qwen3.json" 2>/dev/null || echo "0")

    if [[ "${TEMPLATES}" != "0" ]]; then
        AVG_PATTERNS=$(echo "scale=1; ${TOTAL_PATTERNS} / ${TEMPLATES}" | bc 2>/dev/null || echo "0")
    else
        AVG_PATTERNS="0"
    fi

    printf "| Qwen3-Embedding-8B | %.1f%% | %s | %s | %s |\n" "${COVERAGE}" "${TEMPLATES}" "${TOP_FREQ}" "${AVG_PATTERNS}" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "| Qwen3-Embedding-8B | Failed | Failed | Failed | Failed |" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

**Coverage** = % of messages that matched a template (higher is better)
**Templates Extracted** = Number of unique patterns found
**Top Pattern Frequency** = How many messages matched the most common pattern

---

## Top 10 Templates by Model

### all-mpnet-base-v2 (Best STS Score)
EOF

if [[ -f "${RESULTS_DIR}/templates_mpnet.json" ]]; then
    jq -r '.templates[:10] | to_entries[] | "**\(.key + 1). [\(.value.frequency) uses]** `\(.value.representative[:80])`"' \
        "${RESULTS_DIR}/templates_mpnet.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null || echo "_No templates found_" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "_Mining failed_" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

### NV-Embed-v2 (Best Retrieval Score)
EOF

if [[ -f "${RESULTS_DIR}/templates_nvembed.json" ]]; then
    jq -r '.templates[:10] | to_entries[] | "**\(.key + 1). [\(.value.frequency) uses]** `\(.value.representative[:80])`"' \
        "${RESULTS_DIR}/templates_nvembed.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null || echo "_No templates found_" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "_Mining failed_" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

### Qwen3-Embedding-8B (Best Classification Score)
EOF

if [[ -f "${RESULTS_DIR}/templates_qwen3.json" ]]; then
    jq -r '.templates[:10] | to_entries[] | "**\(.key + 1). [\(.value.frequency) uses]** `\(.value.representative[:80])`"' \
        "${RESULTS_DIR}/templates_qwen3.json" >> "${RESULTS_DIR}/REPORT.md" 2>/dev/null || echo "_No templates found_" >> "${RESULTS_DIR}/REPORT.md"
else
    echo "_Mining failed_" >> "${RESULTS_DIR}/REPORT.md"
fi

cat >> "${RESULTS_DIR}/REPORT.md" <<'EOF'

---

## Analysis Questions for Tomorrow

### 1. Coverage Comparison
- Which model found the most message patterns?
- Did higher MTEB scores translate to better coverage?

### 2. Template Quality
- Which templates are most useful for JARVIS?
- Are there patterns all models found? (high confidence)
- Are there patterns only one model found? (unique insights)

### 3. Practical Application
- Which model's templates would you actually use?
- Did the "best STS" model actually produce the best templates?

### 4. Speed vs Quality Trade-off
- Was the slower model worth the extra time?
- Should we use different models for different template types?

---

## Raw Data Files

- `templates_mpnet.json` - all-mpnet-base-v2 results
- `templates_nvembed.json` - NV-Embed-v2 results
- `templates_qwen3.json` - Qwen3-Embedding-8B results
- `experiment.log` - Full execution log

---

## Next Steps

1. **Review Report** - Read this file tomorrow
2. **Compare Templates** - Look at top 20 from each model
3. **Test Coverage** - Run `benchmarks/templates/run.py --mode real` with each model's templates
4. **Choose Winner** - Pick the model that works best for JARVIS's actual use cases

EOF

log ""
log "Report generated: ${RESULTS_DIR}/REPORT.md"

# Create symlink to latest
rm -f results/latest
ln -sf "$(basename "${RESULTS_DIR}")" results/latest

log ""
log "════════════════════════════════════════════════════════════"
log "EXPERIMENT COMPLETE!"
log "════════════════════════════════════════════════════════════"
log "Finished: $(date)"
log "Duration: ${TOTAL_HOURS} hours"
log "Results: ${RESULTS_DIR}/"
log "Report: ${RESULTS_DIR}/REPORT.md"
log "Symlink: results/latest -> ${RESULTS_DIR}"
log ""
log "To view results tomorrow:"
log "  cat ${RESULTS_DIR}/REPORT.md"
log "  or: cat results/latest/REPORT.md"
log ""
log "════════════════════════════════════════════════════════════"
