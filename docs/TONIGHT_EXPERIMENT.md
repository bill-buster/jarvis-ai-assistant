# Tonight's Overnight Experiment

**Script:** `./scripts/overnight_mlx_comparison.sh`
**Duration:** 6-8 hours
**Memory:** Safe for 8GB RAM
**Using:** MLX (Apple Silicon native) - No Ollama needed!

---

## What Will Run

### Part 1: Template Mining (1-2 hours) 🔴 CRITICAL

**Model:** all-mpnet-base-v2
- **Task:** Mine templates from 22,507 messages
- **Current coverage:** 6.2% (33/528 messages)
- **Expected coverage:** 30-50% (10× better!)
- **Output:** 80-120 data-driven templates

**Why this matters:**
- Instant replies (10ms) for 30-50% of queries
- Replace 91 manual templates (79% unused)
- Deploy immediately tomorrow

---

### Part 2: LLM Comparison (4-6 hours) 🟡 VALUABLE

Testing **6 models** on real iMessage scenarios:

| # | Model | Size | Memory | Benchmark | Strength |
|---|-------|------|--------|-----------|----------|
| 1 | **Qwen2.5-1.5B** | 1.5B | 1.5GB | 60 MMLU | Current baseline |
| 2 | **SmolLM3-3B** | 3B | 2GB | Beats Qwen2.5-3B | Dual reasoning mode |
| 3 | **Ministral-3B** | 3.4B | 2.3GB | **Beats Gemma 3 4B!** | 256K context + vision |
| 4 | **Phi-3-Mini** | 3.8B | 2.5GB | **28 tok/s** | Fastest + coding |
| 5 | **Gemma 3 4B** | 4B | 2.75GB | 70 MMLU | Best instructions |
| 6 | **Qwen3 4B** | 4B | 2.75GB | **74% MMLU-Pro** | Best reasoning |

**Test Scenarios:**
1. Dinner invite (accept/decline politely)
2. Parent request (respectful acknowledgment)
3. Work message (professional tone)
4. Friend banter (matching energy)
5. Running late (understanding response)

**Why this matters:**
- See actual reply quality (not just benchmarks)
- Measure real latency on your hardware
- Pick the model that sounds most natural
- Upgrade path for JARVIS

---

## How to Start

```bash
# Navigate to project
cd ~/coding/jarvis-ai-assistant

# Start experiment (runs in background)
nohup ./scripts/overnight_mlx_comparison.sh > overnight.log 2>&1 &

# Get process ID (save this!)
echo $!

# Watch it start (Ctrl+C after you see it running)
tail -f overnight.log
```

---

## Timeline

```
Tonight  10:00pm  → Start experiment
         10:05pm  → Template mining begins

Tomorrow 12:00am  → Template mining done ✓
         12:05am  → Testing Qwen2.5-1.5B
         1:00am   → Testing SmolLM3-3B
         2:15am   → Testing Ministral-3B
         3:30am   → Testing Phi-3-Mini
         4:45am   → Testing Gemma 3 4B
         6:00am   → Testing Qwen3 4B
         7:00am   → Report generated ✓

         8:00am   → YOU: Wake up! ☀️
         8:01am   → cat results/latest/REPORT.md
         8:15am   → Pick winning model
         8:30am   → Update JARVIS config
```

---

## What You'll Get Tomorrow

### 1. Template Library
```
✅ 80-120 mined templates
✅ 30-50% coverage (vs 6.2% now)
✅ Ready to deploy immediately
✅ Top patterns by frequency

Example:
  1. [281 uses] "Bro 😭"
  2. [79 uses] "Loved an image"
  3. [54 uses] "Idk"
```

### 2. LLM Comparison Report
```
✅ Actual reply samples from each model
✅ Latency measurements (ms per reply)
✅ Success rate and quality
✅ Memory usage

Example:
  Scenario: "Friend: Running 10 mins late sorry"

  Qwen2.5-1.5B: "No worries! See you soon" (850ms)
  SmolLM3-3B:   "All good, take your time!" (1100ms)
  Ministral-3B: "No problem at all! Drive safe" (1200ms)
  Phi-3-Mini:   "No worries! Text when close" (900ms)
  Gemma 3 4B:   "Not a problem! See you when you get here" (1400ms)
  Qwen3 4B:     "No worries at all! Let me know when close" (1300ms)

  → Pick the one that sounds best!
```

### 3. Complete System Design
```
✅ Hybrid pipeline architecture
✅ Performance estimates
✅ Deployment recommendations
✅ Next steps for implementation
```

---

## Expected Improvements

| Metric | Current | After Experiment | Improvement |
|--------|---------|------------------|-------------|
| **Template Coverage** | 6.2% | 30-50% | **10× better** |
| **Instant Replies** | 6.2% | 30-50% | **5-8× more** |
| **Reply Quality** | Good | Best model | **TBD** |
| **Template Quality** | Manual (79% unused) | Data-driven | **All useful** |

---

## Memory Safety

All models fit in 8GB RAM (one at a time):

```
System RAM:        8GB
macOS overhead:    -2GB
Available:         ~6GB

Peak usage:
- Template mining: 420MB
- Largest LLM:     2.75GB (Qwen3 4B or Gemma 3 4B)
- Total:           ~3.2GB ✅ Safe!
```

Models are loaded one at a time and unloaded between tests.

---

## If Something Goes Wrong

### Monitor progress:
```bash
tail -f results/latest/experiment.log
```

### Stop if needed:
```bash
# Find process
ps aux | grep overnight_mlx

# Kill it
kill <PID>
```

### Restart just templates:
```bash
# If LLM tests fail, at least get templates
uv run python -m benchmarks.templates.mine \
    --sample-size 100000 \
    --model "sentence-transformers/all-mpnet-base-v2" \
    --output results/templates_manual.json
```

---

## Tomorrow Morning Checklist

- [ ] Read report: `cat results/latest/REPORT.md`
- [ ] Review template coverage vs baseline
- [ ] Read LLM generation samples
- [ ] Pick winning model based on reply quality
- [ ] Note latency (speed) of winner
- [ ] Decide: Deploy templates immediately?
- [ ] Decide: Upgrade to new LLM model?

---

## Quick Commands

```bash
# Start experiment
nohup ./scripts/overnight_mlx_comparison.sh > overnight.log 2>&1 &

# Check progress
tail -f overnight.log

# View results tomorrow
cat results/latest/REPORT.md

# See just the template stats
jq '.stats' results/latest/templates.json

# See just the model performance
jq '.models | .[] | {name: .info.name, latency: .stats.avg_latency_ms}' \
   results/latest/mlx_comparison_results.json
```

---

## Why These 6 Models?

Based on 2025 benchmarks and research:

1. **Qwen2.5-1.5B** - Need baseline for comparison
2. **SmolLM3-3B** - Beats models in same class (Llama-3.2-3B, Qwen2.5-3B)
3. **Ministral-3B** - Beats models twice its size! (Gemma 3 4B)
4. **Phi-3-Mini** - Fastest inference (28 tok/s), great for coding
5. **Gemma 3 4B** - Instruction following champion, beats Gemma 2 27B
6. **Qwen3 4B** - Highest MMLU-Pro (74%), best reasoning

We're testing the **best small LLM in each category** to find the winner!

---

**Ready? Let's go!** 🚀

```bash
nohup ./scripts/overnight_mlx_comparison.sh > overnight.log 2>&1 &
```

Then go to bed! 😴 You'll have results in the morning! ☀️
