# Model Evaluation & Smart Prompt Findings

This document summarizes the findings from evaluating local LLM models for iMessage reply generation, including prompt engineering experiments and few-shot retrieval improvements.

## Executive Summary

- **Best Model**: `lfm2-2.6b-exp` (LiquidAI's RL-tuned LFM2)
- **Best Approach**: Smart prompts with hybrid few-shot retrieval (V2)
- **Win Rate**: V2 beats V1 by 43% vs 35% on 100 samples
- **Not Recommended**: `qwen3-0.6b` (broken thinking mode)

---

## Models Evaluated

| Model | Size | Status | Notes |
|-------|------|--------|-------|
| `qwen3-0.6b` | 0.5GB | ❌ Not Recommended | Built-in thinking mode can't be disabled |
| `lfm2.5-1.2b` | 0.5GB | ⚠️ Marginal | Produces meta-commentary ("Sure! Here's...") |
| `lfm2-2.6b-exp` | 1.5GB | ✅ Best | Natural responses, follows style well |

### Qwen3-0.6b Issues

Qwen3 models have a built-in "thinking" mode where they output `<think>` tags before responding. This is baked into the model weights and cannot be reliably disabled:

- `/no_think` prefix doesn't work consistently
- Model gets stuck in infinite thinking loop
- When `<think>` is added to stop sequences, returns empty string
- **Conclusion**: Not suitable for reply generation

### LFM2.5-1.2b Issues

LFM2.5 is an instruction-following model that tends to interpret prompts as instructions to explain:

```
Input: "Haha"
Output: "Sure! Here's a casual version under 31 characters:"
```

This meta-commentary behavior is hard to suppress even with explicit "just output the reply" instructions.

### LFM2-2.6b-exp Performance

The RL-tuned LFM2 performs best:
- Follows few-shot examples well
- Generates natural casual text
- Matches user's texting style

---

## Prompt Engineering Evolution

### V1: Original Smart Prompter

The original approach combined:
1. Style instructions based on contact cluster
2. Few-shot examples retrieved by input similarity

**Format:**
```
[~28 chars, casual, no period at end, lol ok]

Examples:
them: Want to come over for dinner tomorrow?
me: Sure

them: wanna grab dinner later
me:
```

**Results on 100 samples:**
- Win rate vs baseline: 75%
- Avg semantic similarity: 0.606

### V2: Improved Smart Prompter

Key improvements:
1. **Hybrid retrieval**: Combines input similarity (60%) + response style matching (40%)
2. **Message type detection**: Identifies question, statement, reaction, greeting
3. **Response embeddings**: Embeds both inputs AND responses for better matching

**Results on 100 samples:**
- V2 win rate: 43%
- V1 win rate: 35%
- Avg V2 similarity: 0.622 (+2.6%)

---

## Few-Shot Retrieval Improvements

### The Problem

Original retrieval matched by INPUT similarity:
- Query: "haha"
- Retrieved: "They dont have access they removed them from everything" (0.75 sim)
- This is NOT a good example of how to respond to "haha"!

### The Solution: Hybrid Retrieval

New approach considers both:
1. **Input similarity**: What they said matches
2. **Response style**: What WE said matches expected style

```python
# Query: "haha" with expected_style="lol ok"

# Regular retrieval:
→ "Costco has niners hoodie can you get one" (0.79) ❌

# Response-style retrieval:
→ "lol ok" (1.00) ✅
→ "Ok" (0.85) ✅
→ "Haha" (0.77) ✅

# Hybrid retrieval:
→ "lol ok" (0.73) ✅
```

### Response Hints by Message Type

| Message Type | Detection | Response Hint |
|--------------|-----------|---------------|
| Reaction | Short, starts with "haha", "lol", etc. | "lol ok" |
| Question | Ends with "?", starts with "what", "when", etc. | "sure" |
| Greeting | Starts with "hey", "hi", "yo" | "hey" |
| Statement | Default | (none - use input matching) |

---

## Key Examples

### Perfect Match Case

**Input**: "Haha"
**Gold**: "Haha"

| Version | Response | Similarity |
|---------|----------|------------|
| V1 | "lol yeah" | 0.82 |
| V2 | "Haha" | **1.00** |

### Near-Perfect Paraphrase

**Input**: "Yea my start date was August 2nd but I went in July 29th"
**Gold**: "Yea my start date was August 2nd but I went in July 29th"

| Version | Response | Similarity |
|---------|----------|------------|
| V1 | "Yea my start date was Aug 2nd but I came July 29th" | **0.96** |
| V2 | "Yea" | 0.45 |

Note: V1 occasionally produces near-perfect matches when retrieval finds very similar examples.

---

## Technical Implementation

### Files Created

| File | Purpose |
|------|---------|
| `core/generation/smart_prompter.py` | V1 prompt builder |
| `core/generation/smart_prompter_v2.py` | V2 prompt builder with hybrid retrieval |
| `core/generation/fewshot_retriever.py` | V1 few-shot retrieval |
| `core/generation/fewshot_retriever_v2.py` | V2 with response embeddings |
| `core/generation/style_prompter.py` | Contact cluster style mapping |
| `core/generation/model_prompts.py` | Model-specific prompt templates |
| `scripts/evaluate_*.py` | Evaluation scripts |
| `scripts/cluster_contacts.py` | Contact clustering by conversation style |

### Evaluation Metrics

1. **Semantic Similarity**: Cosine similarity of embeddings (BGE-base-en-v1.5)
2. **Style Score**: Punctuation, length, abbreviations, casual markers (lol/haha)
3. **Combined Score**: 60% semantic + 40% style
4. **Win Rate**: % of samples where method A beats method B by >0.02

---

## Recommendations

### For Reply Generation

1. **Use `lfm2-2.6b-exp`** as the primary model
2. **Use V2 smart prompter** with hybrid retrieval
3. **Avoid `qwen3-0.6b`** - thinking mode is broken
4. **`lfm2.5-1.2b`** is usable but produces meta-commentary

### For Future Improvements

1. **Expand test set**: Current 200 examples limits retrieval quality
2. **Fine-tune retrieval weights**: 60/40 split could be optimized
3. **Add more response hints**: Cover more message types
4. **Consider conversation context**: Multi-turn context may help

---

## Test Set

The test set contains 200 real iMessage conversations with actual user replies as gold standard:
- Stratified by relationship type (family, friends, work, etc.)
- Clustered into 6 style groups based on conversation patterns
- Location: `results/test_set/test_data.jsonl`

### Cluster Characteristics

| Cluster | Name | Avg Length | LOL Rate |
|---------|------|------------|----------|
| 0 | casual_friends | 29 chars | 2% |
| 1 | formal_contacts | 38 chars | 1% |
| 2 | family | 45 chars | 0% |
| 3 | playful_friends | 26 chars | 4% |
| 4 | close_friends | 31 chars | 3% |
| 5 | group_chats | 24 chars | 2% |

---

## Appendix: Evaluation Results

### Full Model Comparison (200 samples)

| Model | Smart Win% | Baseline Win% | Avg Smart Sim | Avg Baseline Sim |
|-------|------------|---------------|---------------|------------------|
| qwen3-0.6b | N/A | N/A | N/A | N/A |
| lfm2.5-1.2b | 65.5% | 23.0% | 0.524 | 0.508 |
| lfm2-2.6b-exp | **72.5%** | 17.5% | **0.646** | 0.528 |

### V1 vs V2 Comparison (100 samples)

| Metric | V1 | V2 |
|--------|----|----|
| Win Rate | 35% | **43%** |
| Avg Similarity | 0.606 | **0.622** |
| Avg Style Score | 0.868 | 0.866 |
