# Evaluation Pipeline & Known Issues

This document describes the evaluation system for JARVIS reply generation and documents known issues that need to be addressed.

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

## Known Issues

### CRITICAL: Pair Quality Problem

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

**Root Cause**:
The pair extractor (`jarvis/extract.py`) pairs consecutive speaker turns without verifying conversational coherence.

**Proposed Solutions**:
1. **Embedding similarity filter**: Require trigger-response similarity >= 0.55
2. **Reaction removal**: Filter out "Liked", "Loved", etc. tapback reactions
3. **Topic shift detection**: Flag responses starting with "btw", "anyway", etc.
4. **LLM coherence judge**: Use LLM to verify if response addresses trigger

**Script**: `scripts/score_pair_quality.py` implements basic coherence scoring.

---

### Issue: Generated Responses Too Formal

**Issue**: Model generates formal text when user's actual style is casual.

**Example**:
```
Actual:  "Yea"
Generated: "Hey! Yeah, that sounds good!"

Actual:  "LOL"
Generated: "Okay!"

Actual:  "Nah bro im telling u SF is DIFFERENT"
Generated: "Hey there! So yeah, I'm all in and ready..."
```

**Impact**: Generated responses don't match user's texting voice.

**Root Cause**:
- LLM (LFM 2.5 1.2B) has formal training bias
- No explicit style constraints in prompt
- Relationship profiles not enforced strongly enough

**Proposed Solutions**:
1. Add explicit length constraints ("respond in 1-5 words")
2. Include user's actual message examples in prompt
3. Lower temperature further (currently 0.1)
4. Fine-tune style matching in prompt template

---

### Issue: Acknowledgment Handling

**Issue**: Short acknowledgments ("Ok", "Yes", "Sure") get generic emoji responses, but actual responses are often new information.

**Example**:
```
Trigger: "Ok"
Actual: "Got link for Stanford immunology will call tmrw morning"
Generated: "👍"
```

**Impact**: System assumes "Ok" needs an acknowledgment, but conversation flow requires substantive response.

**Root Cause**:
- "Ok" triggers acknowledgment handler before context is considered
- No way to distinguish "Ok (end of topic)" from "Ok (I'll share more)"

**Proposed Solutions**:
1. Use context to determine if acknowledgment trigger needs substantive response
2. Check if user typically follows "Ok" with new info
3. Fallback to generation instead of canned acknowledgments

---

### Issue: Context Not Stored Historically

**Issue**: Pairs extracted before 2026-01-30 don't have `context_text` stored.

**Status**: FIXED in `jarvis/extract.py` - context is now saved during extraction.

**Migration**: Re-run `jarvis db extract` to populate context for existing pairs.

---

### Issue: Contact-Pair Linking

**Issue**: Pairs weren't linked to contacts due to chat_id format mismatch.

**Details**:
- Contacts stored: `+14085090232`
- Pairs stored: `iMessage;-;+14085090232`

**Status**: Fixed via phone number extraction. ~76% of pairs now have contact_id.

**Remaining**: Group chats and some edge cases still unlinked.

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

---

## Embedding Profiles

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

## Files Added/Modified

| File | Purpose |
|------|---------|
| `jarvis/db.py` | Added `is_holdout` column, `split_train_test()`, `get_training_pairs()`, `get_holdout_pairs()` |
| `jarvis/index.py` | Added `include_holdout` parameter to `build_index_from_db()` |
| `jarvis/extract.py` | Fixed `context_text` not being saved in `extract_all_pairs()` |
| `jarvis/embedding_profile.py` | NEW - Embedding-based relationship profiles |
| `scripts/eval_pipeline.py` | NEW - Train/test split and evaluation pipeline |
| `scripts/build_embedding_profiles.py` | NEW - Build embedding profiles for contacts |
| `scripts/score_pair_quality.py` | NEW - Analyze and update pair quality scores |
