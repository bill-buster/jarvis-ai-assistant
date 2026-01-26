#!/bin/bash
# scripts/overnight_eval.sh
# JARVIS Overnight Benchmark Evaluation Suite
#
# Runs all benchmarks sequentially (memory-safe for 8GB machines),
# generates a comprehensive report, and evaluates decision gates.
#
# Usage: ./scripts/overnight_eval.sh [--quick]
#
# Options:
#   --quick    Run in quick mode (reduced iterations for testing)

set -o pipefail  # Catch errors in pipes

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Log file for full output
LOG_FILE="$RESULTS_DIR/eval.log"

# Helper function to log with timestamps
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# Helper function to log errors
log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$msg" | tee -a "$LOG_FILE" >&2
}

# Helper function to run a benchmark with error handling
run_benchmark() {
    local name="$1"
    local command="$2"
    local output_file="$3"
    local requires_mlx="${4:-false}"

    log "Starting $name benchmark..."

    # Check MLX requirement
    if [[ "$requires_mlx" == "true" ]] && [[ "$HAS_MLX" != "true" ]]; then
        log "SKIPPING $name (MLX not available)"
        echo '{"skipped": true, "reason": "MLX not available"}' > "$output_file"
        return 1
    fi

    # Run benchmark and capture exit code
    local start_time=$(date +%s)
    if eval "$command" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "COMPLETED $name in ${duration}s"
        return 0
    else
        local exit_code=$?
        log_error "$name failed with exit code $exit_code"
        return $exit_code
    fi
}

# Convert latency format from {scenarios: {}} to {results: []}
convert_latency_format() {
    local input_file="$1"
    python3 << EOF
import json
import sys
from pathlib import Path

input_path = Path("$input_file")
if not input_path.exists():
    sys.exit(1)

data = json.loads(input_path.read_text())

# If already in expected format, skip conversion
if "results" in data:
    sys.exit(0)

# Convert from scenarios dict to results list
if "scenarios" in data:
    results = []
    for scenario_name, scenario_data in data["scenarios"].items():
        result = {"scenario": scenario_name}
        result.update(scenario_data)
        results.append(result)

    output_data = {
        "timestamp": data.get("timestamp", ""),
        "results": results
    }

    input_path.write_text(json.dumps(output_data, indent=2))
EOF
}

# Print banner
echo ""
echo "========================================"
echo "  JARVIS Overnight Evaluation Suite"
echo "========================================"
echo ""
log "Started: $(date)"
log "Results directory: $RESULTS_DIR"
if [[ "$QUICK_MODE" == "true" ]]; then
    log "Running in QUICK mode (reduced iterations)"
fi
echo ""

# Track benchmark status
MEMORY_OK=false
HHEM_OK=false
LATENCY_OK=false
BENCHMARKS_RUN=0
BENCHMARKS_FAILED=0

# ============================================
# Pre-flight checks
# ============================================
log "Running pre-flight checks..."

# Check Python is available
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not available"
    exit 1
fi

# Check for MLX availability (required for memory/latency benchmarks)
HAS_MLX=false
if python3 -c "import mlx" 2>/dev/null; then
    HAS_MLX=true
    MLX_VERSION=$(python3 -c "import mlx; print(mlx.__version__)" 2>/dev/null)
    log "MLX version: $MLX_VERSION"
else
    log "MLX not available (memory/latency benchmarks will be skipped)"
fi

# Check available memory
if command -v python3 &> /dev/null; then
    AVAILABLE_RAM=$(python3 -c "import psutil; m = psutil.virtual_memory(); print(f'{m.available / 1e9:.1f}GB')" 2>/dev/null || echo "unknown")
    log "Available RAM: $AVAILABLE_RAM"
fi

echo ""
log "Pre-flight checks complete"
echo ""

# ============================================
# Benchmark 1: Memory Profiling (G1)
# ============================================
log "[1/3] Memory Profiling..."
MEMORY_CMD="python3 -m benchmarks.memory.run --output $RESULTS_DIR/memory.json"
if [[ "$QUICK_MODE" == "true" ]]; then
    MEMORY_CMD="$MEMORY_CMD --quick"
fi

if run_benchmark "Memory" "$MEMORY_CMD" "$RESULTS_DIR/memory.json" "true"; then
    MEMORY_OK=true
    ((BENCHMARKS_RUN++))
else
    ((BENCHMARKS_FAILED++))
    ((BENCHMARKS_RUN++))
fi
echo ""

# ============================================
# Benchmark 2: HHEM Hallucination Evaluation (G2)
# ============================================
log "[2/3] HHEM Hallucination Benchmark..."
HHEM_CMD="python3 -m benchmarks.hallucination.run --output $RESULTS_DIR/hhem.json"
if [[ "$QUICK_MODE" == "true" ]]; then
    HHEM_CMD="$HHEM_CMD --verbose"
fi

if run_benchmark "HHEM" "$HHEM_CMD" "$RESULTS_DIR/hhem.json" "false"; then
    HHEM_OK=true
    ((BENCHMARKS_RUN++))
else
    ((BENCHMARKS_FAILED++))
    ((BENCHMARKS_RUN++))
fi
echo ""

# ============================================
# Benchmark 3: Latency (G3, G4)
# ============================================
log "[3/3] Latency Benchmarks..."
LATENCY_CMD="python3 -m benchmarks.latency.run --output $RESULTS_DIR/latency.json --scenario all"
if [[ "$QUICK_MODE" == "true" ]]; then
    LATENCY_CMD="$LATENCY_CMD --runs 3"
fi

if run_benchmark "Latency" "$LATENCY_CMD" "$RESULTS_DIR/latency.json" "true"; then
    LATENCY_OK=true
    ((BENCHMARKS_RUN++))
    # Convert latency output format for compatibility with report/gate scripts
    log "Converting latency output format..."
    convert_latency_format "$RESULTS_DIR/latency.json"
else
    ((BENCHMARKS_FAILED++))
    ((BENCHMARKS_RUN++))
fi
echo ""

# ============================================
# Generate Report
# ============================================
log "Generating benchmark report..."
if python3 scripts/generate_report.py \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/BENCHMARKS.md" >> "$LOG_FILE" 2>&1; then
    log "Report generated: $RESULTS_DIR/BENCHMARKS.md"
    # Also copy to docs for easy access
    cp "$RESULTS_DIR/BENCHMARKS.md" docs/BENCHMARKS.md 2>/dev/null || true
else
    log_error "Failed to generate report"
fi
echo ""

# ============================================
# Check Gates
# ============================================
log "Evaluating decision gates..."
GATE_EXIT_CODE=0

if python3 scripts/check_gates.py "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"; then
    log "Gate evaluation complete"
else
    GATE_EXIT_CODE=$?
    log "Gate evaluation found issues (exit code: $GATE_EXIT_CODE)"
fi
echo ""

# ============================================
# Summary
# ============================================
echo ""
echo "========================================"
echo "           EVALUATION SUMMARY"
echo "========================================"
echo ""
log "Completed: $(date)"
log "Results directory: $RESULTS_DIR"
echo ""
echo "Benchmarks:"
echo "  - Memory:    $([[ "$MEMORY_OK" == "true" ]] && echo 'COMPLETED' || echo 'FAILED/SKIPPED')"
echo "  - HHEM:      $([[ "$HHEM_OK" == "true" ]] && echo 'COMPLETED' || echo 'FAILED/SKIPPED')"
echo "  - Latency:   $([[ "$LATENCY_OK" == "true" ]] && echo 'COMPLETED' || echo 'FAILED/SKIPPED')"
echo ""
echo "Total: $BENCHMARKS_RUN benchmarks run, $BENCHMARKS_FAILED failed"
echo ""
echo "Output files:"
echo "  - Log:       $LOG_FILE"
echo "  - Report:    $RESULTS_DIR/BENCHMARKS.md"
[[ -f "$RESULTS_DIR/memory.json" ]] && echo "  - Memory:    $RESULTS_DIR/memory.json"
[[ -f "$RESULTS_DIR/hhem.json" ]] && echo "  - HHEM:      $RESULTS_DIR/hhem.json"
[[ -f "$RESULTS_DIR/latency.json" ]] && echo "  - Latency:   $RESULTS_DIR/latency.json"
echo ""
echo "========================================"

# Create a symlink to latest results
ln -sfn "$TIMESTAMP" results/latest 2>/dev/null || true

# Exit with gate status code
# 0 = all gates pass
# 1 = one gate failed (reassess)
# 2 = two or more gates failed (consider cancellation)
exit $GATE_EXIT_CODE
