# Reply Generation Improvement Plan

## Current Status (Updated)

```
BASELINE: 28% intent match (qwen2.5 with roleplay prompt)

EXPERIMENTS RUN:
├── Exp 1: Structured LLM Classification → 22% ❌ (LLM classification too unreliable)
├── Exp 2: Embedding-based Classification → 18% ❌ (Mapping incoming→response too simplistic)
└── Both FAILED because: predicting "what response type is needed" doesn't work
```

### Key Insight

**The problem isn't classification accuracy - it's that response type is UNPREDICTABLE.**

Even humans don't always respond the "expected" way:
- "Fractured my foot" → Could be: sympathy, question ("omg what happened"), reaction ("damn"), or info ("that sucks, happened to me too")
- "what about you" → Could be: info, question back, or reaction

**Trying to predict the response type before generating is fundamentally flawed.**

---

## Revised Strategy

Instead of: `classify incoming → map to response type → generate with constraint`

We should: **Let the model generate freely, but steer it with better prompts/examples**

### The Real Problem

| What We Tried | Why It Failed |
|---------------|---------------|
| LLM classification (Exp 1) | LLM is only 20% accurate at classifying |
| Embedding classification (Exp 2) | Mapping is too rigid (26% expectation accuracy) |
| Both assume | Response type is predictable from incoming message |

### What Actually Works (from ablation)

| Prompt Type | Intent Match | Notes |
|-------------|--------------|-------|
| Roleplay framing | **28%** | Best so far |
| Heavy few-shot | 24% | Helps with style |
| Anti-assistant | 22% | Prevents verbose output |
| Pure completion | 20% | Too unconstrained |

---

## New Plan: Focus on What Works

### Phase 1: Improve Few-Shot Quality ⬅️ **NEXT**
**Hypothesis**: The few-shot examples in prompts are biased. Balance them across intents.

Current few-shot bias:
```
Most examples show: accept ("yeah down", "sure")
Missing: decline, questions, reactions, etc.
```

**Action**: Create balanced few-shot with 2 examples per intent type.

### Phase 2: Retrieval-Augmented Few-Shot (RAG)
**Hypothesis**: Generic examples don't capture user's style. Use their ACTUAL past responses.

```
1. Embed incoming message
2. Find 3 similar past conversations where user actually replied
3. Use those real responses as few-shot examples
```

**Why this is different**: Instead of predicting response TYPE, we show similar SITUATIONS.

### Phase 3: Try Larger/Different Models
**Options**:
- qwen2.5-3b (bigger = better reasoning)
- OpenHermes-2.5-Mistral-7B (roleplay-focused)
- Dolphin-Mistral (uncensored, persona-focused)

### Phase 4: DSPy for Prompt Optimization
**What DSPy does**: Automatically optimizes prompts using a training signal.

**The catch**: DSPy needs a good eval metric to optimize against.

Current eval problem:
```
gold: "yeah sure"
gen:  "sounds good"
```
These have the SAME intent but our intent classifier might score them differently.

**DSPy would work IF** we had:
1. A reliable intent classifier (we're at ~70-80% on clear cases)
2. OR human labels for "is this response appropriate?"

**Recommendation**: Try DSPy after we have better eval, OR use it with human-in-loop.

---

## Eval System Problem

### Current Eval

```python
# We classify both gold and generated, check if same intent
gold_intent = classify(gold_response)      # e.g., "accept"
gen_intent = classify(generated_response)  # e.g., "accept"
match = (gold_intent == gen_intent)        # True/False
```

**Problems**:
1. Classifier is imperfect (~80% on clear cases)
2. Multiple intents can be correct (accept OR question could both work)
3. Intent match ≠ response quality

### Better Eval Options

| Eval Type | Pros | Cons |
|-----------|------|------|
| Intent match (current) | Automated, fast | Too rigid, classifier errors |
| LLM-as-judge | Can assess nuance | Slow, expensive, biased |
| Human eval | Ground truth | Slow, expensive |
| Pairwise preference | Easier for humans | Still need humans |
| Multi-intent acceptable | More lenient | Harder to implement |

**Recommendation for DSPy**: Use LLM-as-judge (e.g., GPT-4) as the reward signal:
```
"Is this response appropriate for a casual text conversation?
Incoming: {incoming}
Response: {generated}
Score 1-5"
```

---

## Immediate Next Steps

```
1. [NEXT] Exp 3: Balanced few-shot prompts
   - 2 examples per intent type
   - Test if distribution improves

2. [THEN] RAG with user's past messages
   - Find similar conversations
   - Use real responses as examples

3. [LATER] DSPy optimization
   - Need better eval first (LLM-as-judge)
   - Or collect human preference data
```

---

## Success Criteria (Unchanged)

| Level | Intent Match | Description |
|-------|--------------|-------------|
| Baseline | 28% | Current best |
| Acceptable | 50% | Can be used with user review |
| Good | 65% | Useful for suggestions |
| Excellent | 80%+ | Can auto-send with confidence |

---

## Summary: Why Classification Failed

```
Our Assumption:     incoming message type → predictable response type
Reality:            Response depends on context, relationship, user's mood, etc.

"wanna hang?"
├── User might accept: "yeah down"
├── User might decline: "nah busy"
├── User might ask: "when?"
├── User might react: "lol maybe"
└── ALL ARE VALID - we can't predict which

Better Approach:    Show the model similar past situations, let IT decide
```
