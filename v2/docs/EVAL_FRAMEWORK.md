# Evaluation Framework & Methodology

## Overview

This document outlines the proper approach to evaluating reply generation quality for JARVIS. It identifies what data we have, what's broken with our current approach, and the systematic methodology to fix it.

---

## TL;DR - The Process

```
STEP 1: Profile your contacts (one-time, ~30 min)
        └── Label 50-100 contacts with specific relationships
        └── System infers group chats from members

STEP 2: Validate profiles cover your data
        └── Check: Do profiles cover 80%+ of test samples?

STEP 3: Establish eval ground truth
        └── Human rates 30-50 generated responses
        └── Calibrate LLM-as-judge against human ratings

STEP 4: Run experiments with validated metrics
        └── Now we can trust our measurements
```

---

## Contact Profiling System

### Why Specific Relationships Matter

Broad categories like "family" don't capture how you actually text:

| Contact | Broad Category | Specific Relationship | How You Text |
|---------|---------------|----------------------|--------------|
| Dad | family | dad | Respectful but casual |
| Mihir Shah | family | brother | Very casual, jokes |
| Mom | family | mom | Warm, updates |
| Mihir | friend | close_friend | Casual banter |

### Relationship Types

```
FAMILY (you text each differently):
├── dad        - respectful but casual
├── mom        - warm, caring
├── brother    - casual, joking
├── sister     - casual, supportive
├── cousin     - friendly casual
├── uncle/aunt - respectful
└── grandparent - respectful loving

FRIENDS (levels of closeness):
├── best_friend  - very casual, inside jokes
├── close_friend - casual, comfortable
├── friend       - friendly casual
└── acquaintance - polite casual

ROMANTIC:
├── partner - intimate, affectionate
├── dating  - flirty, interested
└── ex      - varies

WORK/SCHOOL:
├── coworker  - professional friendly
├── boss      - professional respectful
├── classmate - casual friendly
└── professor - formal respectful
```

### Group Chat Inference

Once you profile individuals, the system infers group chat relationships:

```
You profile:
  "Het Patel" → close_friend
  "Meethre Bharot" → close_friend
  "Rishiraj Taylor" → friend

System infers:
  "Het Patel, Meethre Bharot, Rishiraj Taylor +3" → friend_group
```

### Profiling Script

```bash
# Interactive profiling (profiles sorted by message count)
python scripts/profile_my_contacts.py

# Check progress
python scripts/profile_my_contacts.py --stats

# Infer group relationships from member profiles
python scripts/profile_my_contacts.py --infer-groups

# Export for use in generation prompts
python scripts/profile_my_contacts.py --export
```

### Coverage Analysis

```
186 named contacts in test data (phone-only contacts skipped)
500 total samples

Contact distribution:
  Top 15 contacts: ~100 messages (20% of data)
  Top 50 contacts: ~250 messages (50% of data)
  All 186 contacts: 500 messages (100% of data)

Time estimate:
  ~3 seconds per contact × 186 contacts = ~10 minutes
  (just press a key for relationship type)

Recommendation: Profile all 186 named contacts (~10-15 min)
```

---

## Current State Assessment

### What We Have

| Asset | Count | Contents | Status |
|-------|-------|----------|--------|
| `clean_test_data.jsonl` | 500 | conversation, last_message, gold_response, contact, is_group | Available |
| `model_results.jsonl` | 200 | **LLM-labeled** relationship & intent, model outputs | ⚠️ Unverified |
| `contact_clusters.json` | 137 | Contact clustering by message patterns | Unused |
| Embedding classifier | - | Intent classification via sentence-transformers | 44% accurate |

### Data Fields Available

```
clean_test_data.jsonl:
├── contact: "Brian Honea, Gerson" or phone number
├── is_group: true/false
├── conversation: full chat history (them:/me: format)
├── last_message: their most recent message
├── gold_response: YOUR actual reply
└── length_bucket: short/medium/long

model_results.jsonl (200 samples):
├── relationship: family, close_friend, casual_friend, work, romantic
├── intent: statement, open_question, thanks, greeting, logistics...
├── gold_response: your actual reply
└── model_responses: outputs from qwen3-0.6b, lfm2.5-1.2b, lfm2-2.6b-exp
```

---

## The Problem: Unverified Assumptions

### Current Evaluation Chain (Broken)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVERYTHING IS UNVERIFIED                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   "28% intent match" (our reported metric)                          │
│        ↑ compared against                                            │
│   Embedding classifier (44% accurate against heuristic labels)       │
│        ↑ validated against                                           │
│   Heuristic auto-labels (keyword matching)                          │
│        ↑ never validated against                                     │
│   GROUND TRUTH (human judgment) ← WE DON'T HAVE THIS                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### The Relationship/Intent Labels

**How they were created:**
- The `LLMAnalyzer` class used **lfm2.5-1.2b** (a 1.2B parameter model) to guess relationship and intent
- This is the same small model we're testing for generation
- These labels were never validated against human judgment

**Location:** `core/generation/llm_analyzer.py`

```python
class LLMAnalyzer:
    def __init__(self, model_id: str = "lfm2.5-1.2b"):  # Same model we test!
        ...
```

### The Embedding Classifier

**How it works:**
- Uses sentence-transformers (`all-MiniLM-L6-v2`) to embed responses
- Compares to anchor phrases for each intent category
- Returns the intent with highest similarity

**Measured accuracy:** 44% (validated against heuristic labels, not human labels)

**Location:** `scripts/intent_eval.py`, `scripts/validate_classifier.py`

---

## Proper Evaluation Methodology

### The Evaluation Pyramid

```
                    ┌─────────────────────┐
                    │    GROUND TRUTH     │  ← Human labels (expensive, small scale)
                    │   (Human judgment)  │
                    └──────────┬──────────┘
                               │ calibrate against
                    ┌──────────┴──────────┐
                    │   LLM-AS-JUDGE      │  ← GPT-4/Claude rates quality (scalable)
                    │  (GPT-4 or Claude)  │
                    └──────────┬──────────┘
                               │ calibrate against
         ┌─────────────────────┴─────────────────────┐
         │           VALIDATED LLM LABELS             │  ← If >80% accurate
         │   (lfm2.5 labels verified by human spot-check)
         └─────────────────────┬─────────────────────┘
                               │ use as fallback
         ┌─────────────────────┴─────────────────────┐
         │          HEURISTIC METRICS                 │  ← Supplement only
         │  (length, style, embedding classifier)     │
         └───────────────────────────────────────────┘
```

### Step 0: Validate Existing Labels (REQUIRED FIRST)

Before any other evaluation, we must verify the LLM-generated labels are accurate.

**Script:** `scripts/validate_labels.py`

**Process:**
1. Show human 30-50 samples with LLM's guessed relationship/intent
2. Human confirms or corrects each label
3. Calculate LLM label accuracy

**Decision tree:**
```
LLM label accuracy:
├── >80%  → Trust LLM labels, proceed with experiments
├── 60-80% → Use with caution, or re-label with GPT-4/Claude
└── <60%  → Manual labeling required, or rethink approach
```

**Command:**
```bash
python scripts/validate_labels.py --quick 30
```

### Step 1: Establish Ground Truth

Once we know if LLM labels are trustworthy, establish ground truth for generation quality.

**Option A: Human Pairwise Comparison**
- Show gold response + 2-3 model responses (randomized)
- Human ranks them best to worst
- Script: `scripts/human_eval.py`

**Option B: Human Quality Rating**
- For each generated response, human rates 1-5:
  - 1 = completely wrong/inappropriate
  - 3 = acceptable but not ideal
  - 5 = would actually send this

### Step 2: Calibrate LLM-as-Judge

Use a strong LLM (GPT-4, Claude) to scale evaluation.

**Process:**
1. Run LLM-as-judge on the same samples humans rated
2. Compare LLM ratings to human ratings
3. If correlation > 0.8, LLM-as-judge is reliable proxy

**Prompt template:**
```
Rate this text message response on a scale of 1-5.

Context: {conversation}
Their message: {last_message}
Generated reply: {generated}

Criteria:
- Appropriate for casual texting (not formal/assistant-like)
- Matches the tone of the conversation
- Reasonable response to what they said

Rating (1-5):
```

### Step 3: Run Experiments with Validated Metrics

Only after Steps 0-2 can we trust our metrics.

**Metrics to track:**
| Metric | Source | Purpose |
|--------|--------|---------|
| LLM-judge score | GPT-4/Claude | Primary quality metric |
| Human preference | Manual | Ground truth (spot-check) |
| Intent match | Validated labels | Secondary metric |
| Style score | Heuristics | Length, punctuation, etc. |

---

## Experiment Protocol

### Before Running Any Experiment

1. **Verify labels are validated** (Step 0 complete)
2. **Define success metric** (which metric are we optimizing?)
3. **Define baseline** (what are we comparing against?)

### Experiment Structure

```
EXPERIMENT: [Name]
─────────────────────────────────────────
Hypothesis: [What we expect to happen]
Variable:   [What we're changing]
Baseline:   [What we're comparing to]
Metric:     [How we measure success]
Samples:    [N=50 minimum for statistical significance]

RESULTS:
Baseline:    [X]
Experiment:  [Y]
Delta:       [+/- Z%]
Significant: [Yes/No, p < 0.05]
```

### Variables to Test (In Order)

| Priority | Variable | Why |
|----------|----------|-----|
| 1 | Full conversation context | We were only using last_message |
| 2 | Relationship in prompt | "texting a friend" vs "texting mom" |
| 3 | Contact-specific examples (RAG) | Use their actual past responses |
| 4 | Multi-suggestion (3 options) | Let user pick instead of guessing |
| 5 | Fine-tuning | Train on user's actual style |

---

## File Reference

### Data Files
```
results/test_set/
├── clean_test_data.jsonl    # 500 samples, raw
├── model_results.jsonl      # 200 samples, LLM-labeled (UNVERIFIED)
├── labeled_test_data.jsonl  # Test data with human-verified labels
└── test_data.jsonl          # Original test data

results/contacts/
├── contact_profiles.json    # Your contact profiles (relationship, notes)
└── contact_labels.json      # Simple contact → relationship mapping

results/validation/
└── human_labels.json        # Human validation results

results/experiments/
├── exp1_structured_*.json   # Experiment 1 results
├── exp2_embedding_*.json    # Experiment 2 results
├── exp3_full_context_*.json # Experiment 3 results
└── classifier_validation.json
```

### Scripts
```
scripts/
├── profile_my_contacts.py   # STEP 1: Create detailed contact profiles
├── label_contacts.py        # Simple contact labeling (alternative)
├── validate_labels.py       # Validate LLM-generated labels
├── human_eval.py            # Human pairwise comparison
├── proper_eval.py           # Audit what data we have
├── intent_eval.py           # Embedding-based intent classification
├── validate_classifier.py   # Validate the classifier accuracy
├── exp1_structured_generation.py
├── exp2_embedding_classifier.py
├── exp3_full_context.py
└── prompt_ablation.py       # A/B test prompts
```

---

## Success Criteria

| Level | LLM-Judge Score | Description |
|-------|-----------------|-------------|
| Baseline | ~2.5/5 | Current state (unvalidated) |
| Acceptable | 3.5/5 | Usable with user review |
| Good | 4.0/5 | Useful suggestions |
| Excellent | 4.5+/5 | Could auto-send |

---

## Complete Methodology

### Phase 1: Build Ground Truth (One-Time Setup)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1A: PROFILE CONTACTS (~30 min)                                     │
├─────────────────────────────────────────────────────────────────────┤
│  python scripts/profile_my_contacts.py                              │
│                                                                      │
│  For each contact:                                                   │
│  - Specific relationship (dad, brother, close_friend, etc.)         │
│  - Optional notes ("jokes around", "keep it brief")                 │
│                                                                      │
│  Goal: Profile top 100 contacts for 77% sample coverage             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  1B: INFER GROUP RELATIONSHIPS                                       │
├─────────────────────────────────────────────────────────────────────┤
│  python scripts/profile_my_contacts.py --infer-groups               │
│                                                                      │
│  System automatically infers:                                        │
│  - "Dad, Mom, Brother" → family_group                               │
│  - "Het, Meethre, Rishi" → friend_group (if all profiled as friend) │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  1C: VERIFY COVERAGE                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  python scripts/profile_my_contacts.py --stats                      │
│                                                                      │
│  Check: Do profiles cover 80%+ of test samples?                     │
│  If not: Profile more contacts                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Establish Eval Metrics

```
┌─────────────────────────────────────────────────────────────────────┐
│  2A: HUMAN BASELINE (30-50 samples)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  python scripts/human_eval.py --samples 30                          │
│                                                                      │
│  For each sample, rate generated response:                          │
│  1 = completely wrong/inappropriate                                  │
│  3 = acceptable but not ideal                                        │
│  5 = would actually send this                                        │
│                                                                      │
│  This is GROUND TRUTH for eval calibration                          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2B: CALIBRATE LLM-AS-JUDGE                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Run GPT-4/Claude on same 30-50 samples                             │
│  Compare LLM ratings to human ratings                               │
│                                                                      │
│  If correlation > 0.8: LLM-as-judge is reliable                     │
│  If correlation < 0.8: Adjust prompts or use human eval             │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Run Experiments

```
┌─────────────────────────────────────────────────────────────────────┐
│  NOW we can trust our metrics and iterate:                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BASELINE: Generate with current best prompt                         │
│    → Measure with LLM-as-judge                                      │
│    → Record: X/5 average score                                       │
│                                                                      │
│  EXPERIMENT A: Add relationship to prompt                            │
│    "You are texting your brother Mihir..."                          │
│    → Measure: Y/5 (compare to baseline)                             │
│                                                                      │
│  EXPERIMENT B: Add full conversation context                         │
│    → Measure: Z/5 (compare to baseline)                             │
│                                                                      │
│  EXPERIMENT C: RAG with similar past conversations                   │
│    → Measure: W/5 (compare to baseline)                             │
│                                                                      │
│  COMBINE: Best techniques together                                   │
│    → Final score                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Fine-Tuning (Optional)

```
┌─────────────────────────────────────────────────────────────────────┐
│  If experiments plateau below target:                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRAINING DATA:                                                      │
│  - Use profiled contacts as system prompt context                   │
│  - (conversation, your_actual_reply) pairs                          │
│  - 400 train / 100 eval split                                        │
│                                                                      │
│  FINE-TUNE:                                                          │
│  - LoRA on lfm2.5-1.2b or qwen2.5-1.5b                              │
│  - Small learning rate, few epochs                                   │
│                                                                      │
│  EVAL:                                                               │
│  - LLM-as-judge on held-out 100 samples                             │
│  - Compare to pre-fine-tune baseline                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps Checklist

- [ ] **Step 1**: Profile your contacts
  ```bash
  python scripts/profile_my_contacts.py
  ```
  - Profile top 100 contacts for 77% coverage
  - Takes ~30 minutes

- [ ] **Step 2**: Infer group relationships
  ```bash
  python scripts/profile_my_contacts.py --infer-groups
  ```

- [ ] **Step 3**: Verify coverage
  ```bash
  python scripts/profile_my_contacts.py --stats
  ```
  - Goal: 80%+ sample coverage

- [ ] **Step 4**: Human eval baseline
  ```bash
  python scripts/human_eval.py --samples 30
  ```
  - Rate 30 generated responses
  - Establishes ground truth

- [ ] **Step 5**: Run experiments with validated metrics
  - Test relationship prompting
  - Test full context
  - Test RAG

- [ ] **Step 6**: Fine-tune (if needed)
  - Use validated data for training
  - Eval on held-out set

---

## Key Insight

> **We cannot improve what we cannot accurately measure.**
>
> All optimization work is meaningless if our evaluation metrics are unreliable.
> The first priority is establishing trustworthy ground truth, not generating better responses.
