#!/usr/bin/env bash
# Overnight Embedding Model Comparison
# Tests top models on JARVIS's actual use cases:
# 1. Template mining (STS task)
# 2. Semantic search indexing (Retrieval task)
# 3. Intent classification training (Classification task)
#
# Runtime: 6-8 hours on Apple Silicon
# Output: results/embedding_comparison_YYYYMMDD/

set -euo pipefail

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/embedding_comparison_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/experiment.log"

# Models to test (in order of priority)
MODELS=(
    "nvidia/NV-Embed-v2"                    # Best retrieval (62.65)
    "Qwen/Qwen3-Embedding-8B"               # Best classification (~76)
    "sentence-transformers/all-mpnet-base-v2"  # Best STS (87-88)
    "intfloat/e5-mistral-7b-instruct"       # E5-Mistral baseline
    "BAAI/bge-large-en-v1.5"                # BGE-Large baseline
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

log "========================================="
log "JARVIS Embedding Model Comparison"
log "========================================="
log "Output directory: ${RESULTS_DIR}"
log "Models to test: ${#MODELS[@]}"
log ""

# Task 1: Template Mining (STS-focused)
# Mine templates from all iMessage data
log "Task 1: Template Mining (STS task)"
log "─────────────────────────────────────"

for model in "${MODELS[@]}"; do
    model_name=$(echo "${model}" | tr '/' '_')
    output="${RESULTS_DIR}/templates_${model_name}.json"

    log "Mining templates with: ${model}"

    if uv run python -m benchmarks.templates.mine \
        --sample-size 100000 \
        --model "${model}" \
        --eps 0.35 \
        --min-samples 5 \
        --min-frequency 8 \
        --output "${output}" \
        >> "${LOG}" 2>&1; then

        log "✓ Mining completed: ${output}"

        # Extract key metrics
        coverage=$(jq -r '.stats.coverage' "${output}")
        templates=$(jq -r '.stats.templates_extracted' "${output}")
        log "  Coverage: ${coverage}, Templates: ${templates}"
    else
        log "✗ Mining failed for ${model}"
    fi

    log ""
done

# Task 2: Semantic Search Indexing (Retrieval-focused)
# Index all messages and measure search quality
log "Task 2: Semantic Search Indexing (Retrieval task)"
log "───────────────────────────────────────────────────"

for model in "${MODELS[@]}"; do
    model_name=$(echo "${model}" | tr '/' '_')
    output="${RESULTS_DIR}/search_${model_name}.json"

    log "Building search index with: ${model}"

    # TODO: Create benchmarks/search/evaluate.py script
    # This would:
    # 1. Build embedding cache with model
    # 2. Run test queries (e.g., "dinner plans", "meeting tomorrow")
    # 3. Measure retrieval quality (precision@10, recall@10, MRR)

    log "  (Search evaluation script not yet implemented)"
    log ""
done

# Task 3: Intent Classification Training (Classification-focused)
log "Task 3: Intent Classification (Classification task)"
log "──────────────────────────────────────────────────"

for model in "${MODELS[@]}"; do
    model_name=$(echo "${model}" | tr '/' '_')
    output="${RESULTS_DIR}/intent_${model_name}.json"

    log "Training intent classifier with: ${model}"

    # TODO: Create benchmarks/intent/evaluate.py script
    # This would:
    # 1. Encode intent examples with model
    # 2. Test on held-out queries
    # 3. Measure classification accuracy

    log "  (Intent evaluation script not yet implemented)"
    log ""
done

# Generate comparison report
log "========================================="
log "Generating Comparison Report"
log "========================================="

cat > "${RESULTS_DIR}/REPORT.md" <<EOF
# JARVIS Embedding Model Comparison

**Date:** $(date '+%Y-%m-%d %H:%M:%S')
**Duration:** 6-8 hours
**Models Tested:** ${#MODELS[@]}

## Models Evaluated

EOF

for model in "${MODELS[@]}"; do
    cat >> "${RESULTS_DIR}/REPORT.md" <<EOF
- \`${model}\`
EOF
done

cat >> "${RESULTS_DIR}/REPORT.md" <<EOF

## Task 1: Template Mining (STS)

Measures how well models find semantically similar message patterns.

| Model | Coverage | Templates | Top Pattern Frequency |
|-------|----------|-----------|----------------------|
EOF

for model in "${MODELS[@]}"; do
    model_name=$(echo "${model}" | tr '/' '_')
    template_file="${RESULTS_DIR}/templates_${model_name}.json"

    if [[ -f "${template_file}" ]]; then
        coverage=$(jq -r '.stats.coverage' "${template_file}" 2>/dev/null || echo "N/A")
        templates=$(jq -r '.stats.templates_extracted' "${template_file}" 2>/dev/null || echo "N/A")
        top_freq=$(jq -r '.templates[0].frequency' "${template_file}" 2>/dev/null || echo "N/A")

        printf "| %s | %.1f%% | %s | %s |\n" \
            "${model}" \
            "$(echo "${coverage} * 100" | bc)" \
            "${templates}" \
            "${top_freq}" \
            >> "${RESULTS_DIR}/REPORT.md"
    else
        printf "| %s | Failed | Failed | Failed |\n" "${model}" >> "${RESULTS_DIR}/REPORT.md"
    fi
done

cat >> "${RESULTS_DIR}/REPORT.md" <<EOF

## Task 2: Semantic Search (Retrieval)

*To be implemented*

## Task 3: Intent Classification

*To be implemented*

## Recommendations

Based on the results:

1. **Best for Template Mining (STS):** TBD
2. **Best for Semantic Search (Retrieval):** TBD
3. **Best for Intent Classification:** TBD

## Raw Data

All raw results are in: \`${RESULTS_DIR}/\`

EOF

log "Report generated: ${RESULTS_DIR}/REPORT.md"
log ""
log "========================================="
log "Experiment Complete!"
log "========================================="
log "Results: ${RESULTS_DIR}/"
log "View report: cat ${RESULTS_DIR}/REPORT.md"

# Create symlink to latest results
rm -f results/latest_comparison
ln -sf "$(basename "${RESULTS_DIR}")" results/latest_comparison

log "Symlink: results/latest_comparison -> ${RESULTS_DIR}"
