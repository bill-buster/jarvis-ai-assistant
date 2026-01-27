#!/bin/bash
#
# Production Template Mining - Easy Runner
#
# This script runs the complete production template mining pipeline with
# progress updates and error handling.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $*"
}

success() {
    echo -e "${GREEN}✓${NC} $*"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

error() {
    echo -e "${RED}✗${NC} $*"
}

# Check if we're in the right directory
cd "$PROJECT_DIR"

echo ""
echo "=========================================="
echo "  PRODUCTION TEMPLATE MINING"
echo "=========================================="
echo ""
log "Project: jarvis-ai-assistant"
log "Results: $RESULTS_DIR"
echo ""

# Check dependencies
log "Checking dependencies..."

if ! python -c "import sentence_transformers" 2>/dev/null; then
    warning "sentence-transformers not found"
    log "Installing sentence-transformers..."
    uv pip install sentence-transformers
fi

if ! python -c "import sklearn" 2>/dev/null; then
    warning "scikit-learn not found"
    log "Installing scikit-learn..."
    uv pip install scikit-learn
fi

# Check for optional dependencies
if ! python -c "import hdbscan" 2>/dev/null; then
    warning "hdbscan not found (optional, will use DBSCAN fallback)"
    read -p "Install hdbscan? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv pip install hdbscan
    fi
fi

success "Dependencies checked"
echo ""

# Check for iMessage database
IMESSAGE_DB="$HOME/Library/Messages/chat.db"
if [ ! -f "$IMESSAGE_DB" ]; then
    error "iMessage database not found at: $IMESSAGE_DB"
    echo ""
    echo "Make sure:"
    echo "  1. You're on macOS"
    echo "  2. iMessage is set up"
    echo "  3. Full Disk Access is granted to Terminal"
    echo ""
    exit 1
fi

success "iMessage database found"

# Count messages
MSG_COUNT=$(sqlite3 "$IMESSAGE_DB" "SELECT COUNT(*) FROM message WHERE text IS NOT NULL" 2>/dev/null || echo "0")
log "Messages in database: $MSG_COUNT"
echo ""

# Estimate time
if [ "$MSG_COUNT" -gt 50000 ]; then
    warning "Large message database detected (~2-3 hours)"
elif [ "$MSG_COUNT" -gt 20000 ]; then
    log "Medium database (~1-2 hours)"
else
    log "Small database (~30-60 minutes)"
fi

echo ""
read -p "Continue with mining? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    log "Aborted by user"
    exit 0
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Output file
OUTPUT_FILE="$RESULTS_DIR/templates_production_$TIMESTAMP.json"

echo ""
log "Starting production mining..."
log "Output: $OUTPUT_FILE"
echo ""

# Run mining
if python scripts/mine_response_pairs_production.py \
    --output "$OUTPUT_FILE" \
    --min-senders 3; then

    echo ""
    success "Mining complete!"
    success "Results saved to: $OUTPUT_FILE"
    echo ""

    # Show statistics
    log "Template Statistics:"
    echo ""

    TOTAL_PATTERNS=$(jq '.total_patterns' "$OUTPUT_FILE")
    echo "  Total patterns mined: $TOTAL_PATTERNS"

    if command -v jq &> /dev/null; then
        # Show context distribution
        echo ""
        log "Context distribution:"
        jq -r '.metadata.context_distribution | to_entries | .[] | "  \(.key): \(.value)"' "$OUTPUT_FILE" 2>/dev/null || true

        # Show top 5 patterns
        echo ""
        log "Top 5 patterns by score:"
        echo ""
        jq -r '.patterns[:5] | .[] | "  [\(.adaptive_weight | floor)] \"\(.representative_incoming[:50])\" → \"\(.representative_response[:50])\""' "$OUTPUT_FILE" 2>/dev/null || true
    fi

    echo ""
    success "✓ Mining completed successfully!"
    echo ""

    # Ask about human validation
    echo "=========================================="
    echo "  NEXT STEP: Human Validation"
    echo "=========================================="
    echo ""
    log "It's recommended to validate a sample of templates"
    log "This ensures quality before deployment"
    echo ""
    read -p "Run human validation now? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        log "Starting human validation (50 templates)..."
        log "This will be interactive - you'll rate templates"
        echo ""
        sleep 2

        python scripts/validate_templates_human.py \
            "$OUTPUT_FILE" \
            --sample-size 50

        if [ $? -eq 0 ]; then
            success "Human validation complete!"
            echo ""
            log "Validated results saved to:"
            log "  ${OUTPUT_FILE%.json}_humanvalidated.json"
        fi
    else
        log "Skipping human validation"
        log "You can run it later with:"
        echo ""
        echo "  python scripts/validate_templates_human.py \\"
        echo "      $OUTPUT_FILE \\"
        echo "      --sample-size 50"
        echo ""
    fi

    # Next steps
    echo ""
    echo "=========================================="
    echo "  NEXT STEPS"
    echo "=========================================="
    echo ""
    echo "1. Review results:"
    echo "   cat $OUTPUT_FILE | jq '.patterns[:10]'"
    echo ""
    echo "2. Compare with baseline templates:"
    echo "   # TODO: Create comparison script"
    echo ""
    echo "3. Set up A/B test:"
    echo "   # See docs/TEMPLATE_MINING_PRODUCTION.md"
    echo ""
    echo "4. Read documentation:"
    echo "   cat docs/TEMPLATE_MINING_PRODUCTION.md"
    echo ""

else
    echo ""
    error "Mining failed!"
    echo ""
    log "Check the error messages above"
    log "Common issues:"
    echo "  - Full Disk Access not granted"
    echo "  - Missing dependencies (run: uv sync)"
    echo "  - Out of memory (close other apps)"
    echo ""
    exit 1
fi
