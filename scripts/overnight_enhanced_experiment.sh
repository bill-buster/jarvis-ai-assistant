#!/bin/bash
#
# Enhanced Overnight Experiment
#
# IMPROVEMENTS OVER PREVIOUS VERSION:
# 1. ✓ Context-aware template mining
# 2. ✓ HDBSCAN clustering with auto-eps
# 3. ✓ Coherence filtering
# 4. ✓ Adaptive temporal decay
# 5. ✓ Template quality validation
# 6. ✓ Enhanced evaluation metrics
#
# Duration: 6-8 hours
# Memory: Safe for 8GB RAM
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="$RESULTS_DIR/enhanced_experiment_$TIMESTAMP"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"

# Log file
LOG_FILE="$EXPERIMENT_DIR/experiment.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=================================="
log "ENHANCED OVERNIGHT EXPERIMENT"
log "=================================="
log ""
log "Improvements:"
log "  ✓ Context-aware mining (sender, group, time)"
log "  ✓ HDBSCAN clustering (automatic eps)"
log "  ✓ Coherence filtering (no contradictions)"
log "  ✓ Adaptive temporal decay"
log "  ✓ Template quality validation"
log "  ✓ Enhanced evaluation metrics"
log ""
log "Experiment directory: $EXPERIMENT_DIR"
log ""

# Change to project directory
cd "$PROJECT_DIR"

# ============================================================================
# PHASE 1: Enhanced Template Mining (2-3 hours)
# ============================================================================

log "PHASE 1: Enhanced Template Mining"
log "--------------------------------"
log "This will extract context-aware templates with:"
log "  - Sender/group/time metadata"
log "  - Coherence filtering"
log "  - HDBSCAN clustering"
log "  - Adaptive decay"
log ""

log "Starting enhanced template mining..."
uv run python scripts/mine_response_pairs_enhanced.py \
    --use-hdbscan \
    --output "$EXPERIMENT_DIR/templates_enhanced.json" \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "✓ Enhanced template mining complete!"
else
    log "✗ Enhanced template mining failed"
    exit 1
fi

# ============================================================================
# PHASE 2: Template Quality Validation (30-60 min)
# ============================================================================

log ""
log "PHASE 2: Template Quality Validation"
log "------------------------------------"
log "Validating template quality using LLM-as-judge..."
log ""

uv run python scripts/validate_template_quality.py \
    "$EXPERIMENT_DIR/templates_enhanced.json" \
    --output "$EXPERIMENT_DIR/templates_validated.json" \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "✓ Template validation complete!"
else
    log "✗ Template validation failed (non-fatal, continuing)"
fi

# ============================================================================
# PHASE 3: Enhanced Evaluation (1-2 hours)
# ============================================================================

log ""
log "PHASE 3: Enhanced Evaluation"
log "----------------------------"
log "Running enhanced evaluation with:"
log "  - Context awareness"
log "  - Appropriateness scoring"
log "  - Tone matching"
log ""

uv run python scripts/test_realistic_reply_generation_enhanced.py \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "✓ Enhanced evaluation complete!"

    # Move results to experiment directory
    LATEST_RESULT=$(ls -t "$RESULTS_DIR"/realistic_reply_enhanced_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        cp "$LATEST_RESULT" "$EXPERIMENT_DIR/evaluation_results.json"
        log "  Saved to: $EXPERIMENT_DIR/evaluation_results.json"
    fi
else
    log "✗ Enhanced evaluation failed (non-fatal)"
fi

# ============================================================================
# PHASE 4: Generate Comparison Report
# ============================================================================

log ""
log "PHASE 4: Generating Comparison Report"
log "-------------------------------------"

# Create a summary report
REPORT_FILE="$EXPERIMENT_DIR/ENHANCED_REPORT.md"

cat > "$REPORT_FILE" << 'EOF'
# Enhanced Template Mining Experiment Results

## Overview

This experiment used **enhanced template mining** with the following improvements:

### Improvements Over Previous Version

1. **Context-Aware Mining**
   - Tracks sender ID (who sent the message)
   - Tracks group vs. direct chat
   - Tracks time of day (hour)
   - Enables context-specific template matching

2. **Coherence Filtering**
   - Filters contradictory multi-message responses
   - Example: Removes "yeah wait actually can't"
   - Ensures templates make semantic sense

3. **HDBSCAN Clustering**
   - Automatic eps selection (no manual tuning)
   - Better handling of varying density clusters
   - Fallback to DBSCAN with silhouette scoring

4. **Adaptive Temporal Decay**
   - Adjusts based on messaging frequency
   - Heavy texters: 1-year half-life
   - Light texters: 2-year half-life
   - Better captures evolving communication style

5. **Expanded Filtering**
   - Filters stickers, memojis, animojis
   - Filters payment requests (Venmo, Apple Pay)
   - Filters calendar invites
   - Cleaner training data

6. **Conversation Segmentation**
   - Splits chats by 24-hour gaps
   - Treats each conversation separately
   - Better temporal context

7. **Template Quality Validation**
   - LLM-as-judge appropriateness scoring
   - Naturalness checks
   - Safety filters
   - Only keeps templates with score >= 0.7

8. **Enhanced Evaluation**
   - Appropriateness metrics
   - Tone matching (formal vs. casual)
   - Hit rate by context breakdown
   - Better quality measurement

---

## Results

EOF

# Add template stats
log "Extracting template statistics..."

if [ -f "$EXPERIMENT_DIR/templates_enhanced.json" ]; then
    TOTAL_CLUSTERS=$(jq '.total_clusters' "$EXPERIMENT_DIR/templates_enhanced.json")

    echo "### Template Mining Results" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "- **Total clusters**: $TOTAL_CLUSTERS" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    # Top 10 templates
    echo "#### Top 10 Templates by Score" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    jq -r '.patterns[:10] | .[] | "[\(.combined_score | floor)] \"\(.representative_incoming[:50])\" → \"\(.representative_response[:50])\""' \
        "$EXPERIMENT_DIR/templates_enhanced.json" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Add validation stats
if [ -f "$EXPERIMENT_DIR/templates_validated.json" ]; then
    TOTAL=$(jq '.total_templates' "$EXPERIMENT_DIR/templates_validated.json")
    PASSED=$(jq '.passed' "$EXPERIMENT_DIR/templates_validated.json")
    PASS_RATE=$(jq '.pass_rate * 100' "$EXPERIMENT_DIR/templates_validated.json")

    echo "### Template Validation Results" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "- **Total validated**: $TOTAL" >> "$REPORT_FILE"
    echo "- **Passed quality check**: $PASSED" >> "$REPORT_FILE"
    echo "- **Pass rate**: ${PASS_RATE}%" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Add evaluation results
if [ -f "$EXPERIMENT_DIR/evaluation_results.json" ]; then
    echo "### Evaluation Results" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    TEMPLATE_HIT_RATE=$(jq '.template_matching.hit_rate * 100' "$EXPERIMENT_DIR/evaluation_results.json")
    TEMPLATE_APP=$(jq '.quality_metrics.template_quality.avg_appropriateness' "$EXPERIMENT_DIR/evaluation_results.json")
    TEMPLATE_TONE=$(jq '.quality_metrics.template_quality.avg_tone_match' "$EXPERIMENT_DIR/evaluation_results.json")

    echo "#### Template Matching" >> "$REPORT_FILE"
    echo "- **Hit rate**: ${TEMPLATE_HIT_RATE}%" >> "$REPORT_FILE"
    echo "- **Appropriateness**: ${TEMPLATE_APP}/1.0" >> "$REPORT_FILE"
    echo "- **Tone match**: ${TEMPLATE_TONE}/1.0" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << 'EOF'

---

## Next Steps

1. **Deploy validated templates** to production
2. **Compare with baseline** (old mining approach)
3. **Monitor performance** in real usage
4. **Iterate** based on user feedback

---

## Files Generated

- `templates_enhanced.json` - Raw mined templates with context
- `templates_validated.json` - Quality-validated templates
- `evaluation_results.json` - Comprehensive evaluation metrics
- `experiment.log` - Full experiment log
- `ENHANCED_REPORT.md` - This report

EOF

log "✓ Report generated: $REPORT_FILE"

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log ""
log "=================================="
log "EXPERIMENT COMPLETE!"
log "=================================="
log ""
log "Results directory: $EXPERIMENT_DIR"
log ""
log "Files generated:"
log "  - templates_enhanced.json"
log "  - templates_validated.json (if validation succeeded)"
log "  - evaluation_results.json (if evaluation succeeded)"
log "  - ENHANCED_REPORT.md"
log "  - experiment.log"
log ""
log "View report:"
log "  cat $EXPERIMENT_DIR/ENHANCED_REPORT.md"
log ""
log "Compare with baseline:"
log "  # TODO: Create comparison script"
log ""

# Create a symlink to latest results
ln -sfn "$EXPERIMENT_DIR" "$RESULTS_DIR/latest_enhanced"

log "Symlink created: $RESULTS_DIR/latest_enhanced -> $EXPERIMENT_DIR"
log ""
log "All done! 🎉"
