# Evaluation Pipeline & Known Issues

This document describes the evaluation system for JARVIS reply generation and documents all known issues and limitations.

**Last Updated**: 2026-01-30

---

## Table of Contents

1. [Evaluation Pipeline](#evaluation-pipeline)
2. [Platform Requirements](#platform-requirements)
3. [Known Issues](#known-issues)
4. [Feature Limitations](#feature-limitations)
5. [Performance Characteristics](#performance-characteristics)
6. [Pair Quality Analysis](#pair-quality-analysis)
7. [Reporting New Issues](#reporting-new-issues)

---

## Evaluation Pipeline

### Overview

The evaluation system tests reply generation quality on held-out data:

1. **Train/Test Split**: 80/20 split by contact (all pairs for a contact go to same set)
2. **Index Building**: FAISS index built only from training pairs
3. **Evaluation**: Run triggers through router, compare to actual responses

### Usage

```bash
# Step 1: Create train/test split
python -m scripts.eval_pipeline --setup --holdout-ratio 0.2 --seed 42

# Step 2: Rebuild index (excludes holdout pairs)
python -m scripts.eval_pipeline --rebuild-index

# Step 3: Run evaluation
python -m scripts.eval_pipeline --limit 100 --verbose

# Save detailed results
python -m scripts.eval_pipeline --output results.json
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Semantic Similarity** | Cosine similarity between generated and actual response embeddings |
| **Length Ratio** | `len(generated) / len(actual)` - should be ~1.0 |
| **Route Distribution** | % of messages routed to template/generate/clarify/acknowledgment |
| **Quality Threshold** | % of pairs with similarity >= 0.5 |
| **Latency** | Time per generation in milliseconds |

### Current Results (as of 2026-01-30)

```
Total pairs evaluated: 30
Route distribution:
  acknowledgment :   12 ( 40.0%)
  generated      :   17 ( 56.7%)
  clarify        :    1 (  3.3%)

Metrics:
  Avg semantic similarity: 0.590
  Avg length ratio:        2.483
  Quality threshold (>= 0.5): 90.0%
```

---

## Platform Requirements

### macOS Only

JARVIS requires macOS because:
- iMessage database (`~/Library/Messages/chat.db`) is macOS-specific
- MLX acceleration requires Apple Silicon
- AddressBook contacts integration is macOS-specific
- Calendar integration uses macOS Calendar database

**Will NOT work on**: Linux, Windows, Intel Macs

### Memory Requirements

| Model | Min RAM | Recommended |
|-------|---------|-------------|
| Qwen2.5-0.5B-4bit | 8GB | 8GB |
| Qwen2.5-1.5B-4bit | 8GB | 8GB |
| Qwen2.5-3B-4bit | 8GB | 12GB |
| LFM2.5-1.2B-4bit | 8GB | 8GB (default) |

The default model (LFM2.5-1.2B-Instruct-MLX-4bit) targets 8GB Macs.

---

## Known Issues

### HIGH Priority

#### 1. iMessage Database Access Requires Full Disk Access

**Symptom**: "Permission denied" or empty conversation list

**Cause**: macOS Sequoia+ requires explicit Full Disk Access for applications reading `~/Library/Messages/chat.db`

**Solution**:
1. Open System Settings > Privacy & Security > Full Disk Access
2. Add your Terminal app (or IDE)
3. Restart the Terminal/IDE

**Verification**: `uv run python -m jarvis.setup --check`

#### 2. Model Download Required Before First Use

**Symptom**: "FileNotFoundError: Model not found"

**Cause**: MLX models must be downloaded from HuggingFace before use

**Solution**:
```bash
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit
```

Or run setup wizard:
```bash
uv run python -m jarvis.setup
```

#### 3. First Generation is Slow (Cold Start)

**Symptom**: First reply takes 10-15 seconds

**Cause**: Model loading and Metal shader compilation on first run

**Solution**: This is expected behavior. Subsequent generations use cached model (~2-3s).

**Mitigation**: The API server pre-loads the model on startup.

#### 4. Pair Quality Problem

**Issue**: Many extracted (trigger, response) pairs are NOT true Q&A pairs.

**Example**:
```
Context: [conversation about walnuts/freezer]
Trigger: "Ok"
Response: "There are 3 bags of almonds"
```

The "Ok" acknowledged something earlier. The "almonds" message is a NEW TOPIC, not a response to "Ok". But the extractor pairs them because they're consecutive (them → me) turns.

**Impact**:
- ~48% of pairs are "mediocre" or "bad" quality
- Training data contains false patterns
- Evaluation metrics are misleading

**Root Cause**: The pair extractor (`jarvis/extract.py`) pairs consecutive speaker turns without verifying conversational coherence.

**Mitigations Implemented**:
1. Embedding similarity filter (>= 0.55)
2. Reaction removal ("Liked", "Loved", etc.)
3. Topic shift detection ("btw", "anyway", etc.)
4. Quality scoring via `scripts/score_pair_quality.py`

### MEDIUM Priority

#### 5. iMessage Sender is Unreliable (DEPRECATED)

**Symptom**: Sending messages fails or requires constant permission prompts

**Cause**: Apple restricts AppleScript automation for Messages.app. Known issues:
- Requires Automation permission
- May be blocked by SIP
- Requires Messages.app to be running
- Breaks with macOS updates

**Solution**: The `IMessageSender` class is deprecated. JARVIS generates reply suggestions but does NOT send them automatically.

**Location**: `integrations/imessage/sender.py` (marked DEPRECATED)

#### 6. Group Chat Handling is Limited

**Symptom**: Reply suggestions for group chats may be less accurate

**Cause**:
- Template matching uses group size but context is still limited
- Multiple participant threads are harder to track
- RAG search doesn't distinguish group context well

**Mitigation**: The intent classifier has GROUP_COORDINATION, GROUP_RSVP, GROUP_CELEBRATION intents, but quality varies.

#### 7. Generated Responses Too Formal

**Issue**: Model generates formal text when user's actual style is casual.

**Example**:
```
Actual:  "Yea"
Generated: "Hey! Yeah, that sounds good!"

Actual:  "LOL"
Generated: "Okay!"
```

**Impact**: Generated responses don't match user's texting voice.

**Root Cause**:
- LLM (LFM 2.5 1.2B) has formal training bias
- No explicit style constraints in prompt

**Mitigations**:
1. Length constraints in prompts
2. User's actual message examples in prompt
3. Low temperature (0.1)

#### 8. Acknowledgment Handling

**Issue**: Short acknowledgments ("Ok", "Yes", "Sure") get generic emoji responses, but actual responses are often new information.

**Example**:
```
Trigger: "Ok"
Actual: "Got link for Stanford immunology will call tmrw morning"
Generated: "👍"
```

**Root Cause**: "Ok" triggers acknowledgment handler before context is considered.

**Mitigation**: Context-aware acknowledgment routing via `_should_generate_after_acknowledgment()`.

#### 9. HHEM Model Requires Separate Download

**Symptom**: HHEM benchmark fails with model not found

**Cause**: Vectara HHEM model must be downloaded separately

**Solution**:
```bash
huggingface-cli download vectara/hallucination_evaluation_model
```

### LOW Priority

#### 10. Contact Resolution May Miss Some Contacts

**Symptom**: Phone numbers shown instead of names

**Cause**: AddressBook database structure varies across macOS versions and sync sources (iCloud, Google, etc.)

**Workaround**: JARVIS tries multiple AddressBook sources but may miss contacts synced from certain providers.

#### 11. Schema Detection for Older macOS

**Symptom**: Query errors on older macOS versions

**Cause**: JARVIS supports macOS Sonoma (v14) and Sequoia (v15) schemas. Older versions may have incompatible schemas.

**Solution**: Upgrade to macOS Sonoma or later.

#### 12. PDF Export Requires Additional Dependencies

**Symptom**: PDF export fails

**Cause**: PDF generation requires `weasyprint` which has system dependencies

**Solution**: Install system dependencies:
```bash
brew install pango gdk-pixbuf libffi
pip install weasyprint
```

#### 13. Context Not Stored Historically

**Issue**: Pairs extracted before 2026-01-30 don't have `context_text` stored.

**Status**: FIXED in `jarvis/extract.py` - context is now saved during extraction.

**Migration**: Re-run `jarvis db extract` to populate context for existing pairs.

#### 14. Contact-Pair Linking

**Issue**: Pairs weren't linked to contacts due to chat_id format mismatch.

**Details**:
- Contacts stored: `+14085090232`
- Pairs stored: `iMessage;-;+14085090232`

**Status**: Fixed via phone number extraction. ~76% of pairs now have contact_id.

**Remaining**: Group chats and some edge cases still unlinked.

---

## Feature Limitations

### What JARVIS Does NOT Do

1. **Send messages automatically** - Generates suggestions only
2. **Access other messaging apps** - iMessage only
3. **Work offline completely** - Model download requires internet initially
4. **Fine-tune models** - Uses RAG + few-shot (fine-tuning increases hallucinations)
5. **Store conversation history** - Reads iMessage database directly, no duplication
6. **Sync across devices** - Local-only, per-machine

### Template Coverage

The template system covers common responses (~25 templates) but won't match every message. When templates don't match:
1. System falls back to LLM generation
2. Quality depends on conversation context
3. Very specific or unusual messages may produce generic responses

### RAG Limitations

- RAG search requires embedding index build (first run)
- Cross-conversation search requires relationship registry setup
- Embedding similarity thresholds are configurable; defaults may still miss relevant context

---

## Performance Characteristics

### Expected Latencies

| Operation | Cold Start | Warm Start |
|-----------|------------|------------|
| Model load | 10-15s | N/A |
| Template match | <50ms | <50ms |
| LLM generation | N/A | 2-3s |
| iMessage query | 50-200ms | 50-200ms |
| Embedding search | 100-500ms | 100-500ms |

### Memory Usage

| Component | Usage |
|-----------|-------|
| Qwen2.5-1.5B-4bit | ~1.5GB |
| Embedding model | ~400MB |
| Python runtime | ~200MB |
| Total typical | ~2.5GB |
| Peak during generation | ~4GB |

---

## Pair Quality Analysis

Run the quality analyzer to see current distribution:

```bash
python -m scripts.score_pair_quality --analyze --limit 500
```

**Quality Criteria**:

| Verdict | Criteria |
|---------|----------|
| **GOOD** | Trigger-response similarity >= 0.6, not a reaction, not topic shift |
| **MEDIOCRE** | Similarity 0.45-0.6, or starts with topic shift indicator |
| **BAD** | Reaction ("Liked..."), acknowledgment trigger with low similarity, similarity < 0.45 |

**Update quality scores**:

```bash
# Dry run (preview changes)
python -m scripts.score_pair_quality --update

# Actually commit changes
python -m scripts.score_pair_quality --update --commit
```

### Embedding Profiles

Embedding-based relationship profiles analyze communication patterns:

```bash
# Build for single contact
python -m scripts.build_embedding_profiles --contact "Name" --verbose

# Build for all contacts
python -m scripts.build_embedding_profiles --limit 50
```

**Profile Contents**:
- Topic clusters (K-means on message embeddings)
- Communication dynamics (style similarity, initiation patterns)
- Response semantic shift

**Storage**: `~/.jarvis/embedding_profiles/{contact_hash}.json`

---

## Reporting New Issues

1. Check if issue is listed above
2. Run `make health` and capture output
3. Run `uv run python -m jarvis.setup --check` and capture output
4. Include macOS version, Python version, available RAM
5. Report at: https://github.com/anthropics/claude-code/issues

---

## Workarounds Summary

| Issue | Workaround |
|-------|------------|
| Permission denied | Grant Full Disk Access |
| Model not found | `huggingface-cli download <model>` |
| Slow first generation | Expected (cold start) |
| Sending fails | Don't use sender (deprecated) |
| Wrong contact names | Check AddressBook sync |
| Group chat quality | Use simpler responses |
| PDF fails | Install weasyprint deps |

---

## Recommended Workflow

1. **Extract pairs with context**:
   ```bash
   jarvis db extract
   ```

2. **Score pair quality**:
   ```bash
   python -m scripts.score_pair_quality --update --commit
   ```

3. **Create train/test split**:
   ```bash
   python -m scripts.eval_pipeline --setup
   ```

4. **Build index** (training only):
   ```bash
   python -m scripts.eval_pipeline --rebuild-index
   ```

5. **Run evaluation**:
   ```bash
   python -m scripts.eval_pipeline --limit 100
   ```

6. **Build embedding profiles**:
   ```bash
   python -m scripts.build_embedding_profiles
   ```

---

## Files Reference

| File | Purpose |
|------|---------|
| `jarvis/db.py` | Train/test split methods (`split_train_test()`, `get_training_pairs()`, `get_holdout_pairs()`) |
| `jarvis/index.py` | `include_holdout` parameter for index building |
| `jarvis/extract.py` | Pair extraction with context |
| `jarvis/embedding_profile.py` | Embedding-based relationship profiles |
| `scripts/eval_pipeline.py` | Train/test split and evaluation pipeline |
| `scripts/build_embedding_profiles.py` | Build embedding profiles for contacts |
| `scripts/score_pair_quality.py` | Analyze and update pair quality scores |
