# JARVIS v3 Improvements Summary

**Date:** 2026-01-29
**Status:** Implemented and tested

---

## Overview

This document summarizes the improvements made to the v3 reply generation system, including features ported from root and v2 versions, and the results of evaluation testing.

---

## Version Comparison

| Feature | Root | V2 | V3 (Before) | V3 (After) |
|---------|------|----|----|-----|
| Intent Classification | ✅ 820 lines, semantic | ❌ None | ❌ None | ✅ Ported + enhanced |
| RAG System | ✅ Basic | ✅ FAISS | ✅ FAISS | ✅ FAISS + used as suggestions |
| Prompts | ✅ 1600 lines, RAG_REPLY | ✅ 600 lines | ✅ 630 lines | ✅ RAG_REPLY + threaded |
| Text Truncation | ? | 200 chars | 200 chars | ✅ Full text |
| Tone Detection | ✅ Yes | ❌ No | ❌ No | ✅ Ported |
| Group Chat Support | ✅ Threaded | ❌ Basic | ❌ Basic | ✅ Threaded |
| Context Awareness | ❌ No | ❌ No | ❌ No | ✅ NEW |

---

## Improvements Implemented

### 1. Removed Text Truncation
**File:** `core/embeddings/store.py:443`

**Before:** Messages truncated to 200 characters
```python
msg["text"][:200],
```

**After:** Full message text stored
```python
msg["text"],  # Store full text, not truncated
```

**Impact:** Long messages (like Aditya's work request) now fully indexed and searchable.

---

### 2. Ported Intent Classifier from Root
**File:** `core/intent.py` (new file, ~500 lines)

**Features:**
- Semantic embedding-based classification using all-MiniLM-L6-v2
- 8 query intents: REPLY, SUMMARIZE, SEARCH, QUICK_REPLY, GENERAL, GROUP_*
- 12 message intents: YES_NO_QUESTION, OPEN_QUESTION, CHOICE_QUESTION, STATEMENT, EMOTIONAL, GREETING, THANKS, FAREWELL, REQUEST, LOGISTICS, SHARING, INFORMATION_SEEKING

**Key Functions:**
- `classify_message(text)` - Classify incoming messages
- `classify_query(text)` - Classify user queries to JARVIS
- Thread-safe singleton pattern

---

### 3. RAG Suggestions as Actual Replies
**File:** `core/generation/reply_generator.py`

**Before:** RAG past_replies used only for few-shot prompting, remaining slots filled with generic fallbacks ("got it", "cool")

**After:** RAG past_replies used as actual reply suggestions

```python
def _get_rag_suggestions(self, past_replies, count):
    """Convert RAG past_replies into reply suggestions."""
    # Your actual past replies are better than "got it", "cool"
```

**Impact:**
| Metric | Before | After |
|--------|--------|-------|
| Fallback replies | 63% | 12% |
| RAG suggestions | 0% | 58% |

---

### 4. Ported RAG_REPLY_TEMPLATE from Root
**File:** `core/generation/prompts.py`

**Template Structure:**
```
### Communication Style with {contact_name}:
{relationship_context}

### Similar Past Exchanges:
{similar_exchanges}

### Current Conversation:
{context}

### Message Type: {intent_hint}

### Instructions:
Generate a natural reply that:
- Matches how you typically communicate with {contact_name} ({tone})
- Is consistent with your past response patterns
- Sounds authentic to your voice
{response_guidance}

### Last message to reply to:
{last_message}

### Your reply:
```

**Features:**
- Relationship context (tone, message length patterns)
- Similar past exchanges formatted as examples
- Intent-specific response guidance
- Tone detection (casual/professional/mixed)

---

### 5. Ported Threaded Reply Template for Groups
**File:** `core/generation/prompts.py`

```python
THREADED_REPLY_TEMPLATE = """### Thread Context:
Topic: {thread_topic}
State: {thread_state}
Your role: {user_role}
{participants_info}
...
"""
```

**Used when:** Multiple senders detected in conversation (group chat)

---

### 6. Tone Detection
**File:** `core/generation/prompts.py`

```python
CASUAL_INDICATORS = {"lol", "haha", "btw", "gonna", "u", "ur", ...}
PROFESSIONAL_INDICATORS = {"regarding", "please", "kindly", "deadline", ...}

def detect_tone(messages) -> "casual" | "professional" | "mixed"
```

---

### 7. Intent-Guided Response Generation
**File:** `core/generation/prompts.py`

```python
INTENT_RESPONSE_GUIDANCE = {
    "yes_no_question": "give a direct answer (yes/no/maybe)",
    "open_question": "provide a thoughtful answer",
    "emotional": "respond with empathy and support",
    "information_seeking": "answer if you know, or ask for clarification",
    ...
}
```

**Impact:** Model now knows what TYPE of response is expected based on the incoming message.

---

### 8. Context-Awareness System (NEW)
**Files:** `core/intent.py`, `core/generation/reply_generator.py`

**New IntentResult fields:**
```python
@dataclass
class IntentResult:
    intent: MessageIntent
    confidence: float
    needs_context: bool = False      # Requires specific info to answer
    is_specific_question: bool = False  # Asking about facts/details
```

**Detection patterns:**
- "What was the address again?" → needs_context=True
- "When did we plan to meet?" → needs_context=True
- "Can you remind me?" → needs_context=True

**Clarification responses:**
When RAG similarity < 0.35 AND message needs context:
- "hmm not sure, when was that?"
- "can you give me more context?"
- "remind me what we were talking about?"

---

### 9. Improved Question Detection
**File:** `core/generation/context_analyzer.py`

**Before:** Missed questions with "u" instead of "you" or without "?"

**After:** Added patterns:
- "were u", "did u", "are u", "can u" variants
- Questions without "?" detected by starter words
- "any progress", "how's it looking" patterns

---

## Evaluation Results

### Before Improvements (baseline_v3.json)
```
Reply Type Distribution:
  fallback: 36 (63%)
  general: 17 (30%)
  personal_template: 4 (7%)

Average warm generation time: 528ms
```

### After Improvements (baseline_v3_improved.json)
```
Reply Type Distribution:
  fallback: 7 (12%)      ↓ 81%
  general: 17 (30%)
  rag_suggestion: 33 (58%)  NEW!

Average warm generation time: ~530ms (no regression)
```

### Example Improvements

| Contact | Message | Before | After |
|---------|---------|--------|-------|
| Rachit Mom | "I am here at your place" | "got it", "cool" | "I'm right outside", "sounds good" |
| Abhinav | "That sounds miserable" | "got it", "cool" | "i mean it's not that bad" |
| Ethan | "Were u in vanshs league" | "got it", "cool" | (RAG suggestions from history) |

---

## Files Changed

| File | Change |
|------|--------|
| `core/intent.py` | NEW - Intent classifier ported from root |
| `core/generation/prompts.py` | Added RAG_REPLY_TEMPLATE, THREADED_REPLY_TEMPLATE, tone detection |
| `core/generation/reply_generator.py` | RAG suggestions, clarification responses, intent-guided prompts |
| `core/generation/context_analyzer.py` | Improved question detection |
| `core/embeddings/store.py` | Removed 200 char truncation |
| `scripts/reindex_full_text.py` | NEW - Re-index script for full text |
| `tests/test_intent.py` | NEW - 49 tests for intent classification |
| `tests/test_context_analyzer.py` | NEW - 22 tests for context analysis |

---

## Test Coverage

```
tests/test_intent.py: 49 tests
tests/test_context_analyzer.py: 22 tests
Total new tests: 71
All passing ✅
```

---

## Next Steps (Potential)

1. **Confidence scoring** - Show confidence level for each suggestion
2. **Search fallback** - When information_seeking detected, offer to search history
3. **Learning from feedback** - Track which suggestions users actually send
4. **A/B testing** - Compare RAG vs legacy prompt strategies

---

## Running Evaluation

```bash
# Re-index with full text (one-time)
uv run python scripts/reindex_full_text.py

# Run evaluation
uv run python scripts/evaluate_replies.py --samples 20 --contacts-only --one-to-one -o results/baseline_v3_final.json

# Compare results
uv run python -c "
import json
with open('results/baseline_v3.json') as f: old = json.load(f)
with open('results/baseline_v3_final.json') as f: new = json.load(f)
# ... comparison logic
"
```
