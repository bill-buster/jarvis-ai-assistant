# Version Comparison Findings

**Date:** 2026-01-29
**Versions Analyzed:** Root, V2, V3

## Executive Summary

We analyzed all three versions of JARVIS reply generation and consolidated the best features into V3. The key improvement was using RAG past replies as actual suggestions instead of generic fallbacks, reducing fallback usage from 63% to 12%.

---

## Feature Comparison Matrix

| Feature | Root | V2 | V3 (Current) |
|---------|------|-----|--------------|
| **Intent Detection** | 820 lines, semantic embeddings | None | Ported from root + improved |
| **RAG System** | Basic embeddings | FAISS indexed | FAISS + used as suggestions |
| **Prompts** | 1600 lines, RAG_REPLY_TEMPLATE | 600 lines, basic | 630 lines, style_header |
| **Relationship Context** | Full profile | Basic contact | Basic contact_profile |
| **Fallback Strategy** | Generic only | Generic + personal | RAG suggestions first |
| **Text Storage** | Unknown | 200 char truncation | Full text (fixed) |
| **Question Detection** | N/A | Basic patterns | Improved ("u" variants, no "?") |

---

## Improvements Made (2026-01-29)

### 1. Removed Text Truncation
- **File:** `core/embeddings/store.py:443`
- **Before:** Messages truncated to 200 characters
- **After:** Full message text stored
- **Impact:** Long messages like "Please work with cooper and make sure we are erring on the side of caution" now fully indexed

### 2. Ported Intent Classifier from Root
- **New file:** `core/intent.py`
- **Features:**
  - Semantic embedding-based classification (all-MiniLM-L6-v2)
  - 8 intent types for user queries (REPLY, SUMMARIZE, SEARCH, etc.)
  - 11 intent types for incoming messages (YES_NO_QUESTION, OPEN_QUESTION, etc.)
  - Thread-safe singleton pattern
  - Parameter extraction (person_name, time_range)

### 3. RAG Past Replies as Suggestions
- **File:** `core/generation/reply_generator.py`
- **Before:** RAG results only used for few-shot prompting, fallbacks were generic
- **After:** RAG results used as actual reply suggestions
- **New method:** `_get_rag_suggestions()` converts past replies to GeneratedReply objects

### 4. Improved Question Detection
- **File:** `core/generation/context_analyzer.py`
- **Added patterns:**
  - "u" variants: "were u", "are u", "did u", etc.
  - Questions without "?" mark
  - "any progress", "how's it looking" patterns

### 5. New Tests
- **Files:** `tests/test_intent.py`, `tests/test_context_analyzer.py`
- **Coverage:** 44 new tests passing

---

## Quality Metrics

### Reply Type Distribution

| Type | Before | After | Change |
|------|--------|-------|--------|
| fallback | 36 (63%) | 7 (12%) | **-81%** |
| general | 17 (30%) | 17 (30%) | — |
| personal_template | 4 (7%) | 0 (0%) | -4 |
| rag_suggestion | 0 (0%) | 33 (58%) | **+33** |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Cold start (model load) | ~27 seconds |
| Warm generation (avg) | ~530ms |
| RAG lookup time | ~10-30ms |
| Past replies found (avg) | 3.7 per sample |

---

## Sample Improvements

### Example 1: Logistics Response
**Message:** "I am here at your place"

| Version | Replies |
|---------|---------|
| Before | `["cool, see you soon!", "sounds good", "got it"]` |
| After | `["cool, see you soon!", "I'm right outside", "sounds good"]` |

### Example 2: Question Response
**Message:** "Were u in vanshs league last year"

| Version | Replies |
|---------|---------|
| Before | `["got it", "cool", "nice"]` (didn't answer!) |
| After | RAG suggestions from actual conversation history |

### Example 3: Statement Response
**Message:** "That sounds miserable"

| Version | Replies |
|---------|---------|
| Before | `["Yeah, it's rough", "got it", "cool"]` |
| After | `["Yeah, it's rough", "i mean it's not that bad", "like when its just 3-4 its aight"]` |

---

## Features Still Available from Root

### 1. RAG_REPLY_TEMPLATE
**Location:** `jarvis/prompts.py:1091-1116`

A sophisticated prompt template with:
```
### Communication Style with {contact_name}:
{relationship_context}

### Similar Past Exchanges:
{similar_exchanges}

### Current Conversation:
{context}

### Instructions:
Generate a natural reply that:
- Matches how you typically communicate with {contact_name}
- Is consistent with your past response patterns
- Sounds authentic to your voice
```

### 2. Relationship Context Formatting
**Location:** `jarvis/prompts.py:1139-1181`

Formats relationship insights:
- Tone description (casual/professional/mixed)
- Message length guidance (short/moderate/long)
- Response time patterns (quick responder vs takes time)

### 3. Similar Exchanges Formatting
**Location:** `jarvis/prompts.py:1119-1136`

Formats past exchanges as clear examples:
```
Example 1:
Context: [past conversation]
Your reply: [what you said]
```

### 4. Threaded Reply Support
**Location:** `jarvis/prompts.py:417-476`

For group chat thread context with:
- Thread topic detection
- Thread state tracking
- User role in thread
- Participant info

### 5. Tone-Aware Examples
**Location:** `jarvis/prompts.py:65-120`

Separate example sets for:
- Casual tone (friends, family)
- Professional tone (work, clients)

---

## Known Issues

### 1. RAG Context Mismatch
Some RAG suggestions don't match the current context well. The similarity search finds your past replies, but they may be from unrelated conversations.

**Example:**
- Message: "It's so bad bc Arrowhead is crazy expensive now"
- RAG suggestion: "bro I fuckign hope so" (from different context)

**Potential fix:** Add topic/context filtering to RAG search

### 2. LLM First Reply Still Generic
The model-generated first reply is often generic ("yeah", "yep", "lol"). The RAG suggestions are more personal but come second.

**Potential fix:** Use RAG_REPLY_TEMPLATE from root to give model better context

### 3. Questions Not Directly Answered
Yes/no questions often get acknowledgments instead of actual yes/no answers.

**Potential fix:** Detect question type and force appropriate response format

---

## Recommended Next Steps

### High Priority
1. **Re-index messages** - Run `index_messages.py` to store full text (not truncated)
2. **Port RAG_REPLY_TEMPLATE** - Better prompt structure with relationship context

### Medium Priority
3. **Add topic filtering to RAG** - Ensure suggestions match current conversation topic
4. **Implement question answering** - Detect and answer yes/no questions directly

### Low Priority
5. **Port threaded reply support** - For better group chat handling
6. **Add tone detection** - Casual vs professional based on contact

---

## File Changes Summary

| File | Change |
|------|--------|
| `core/embeddings/store.py` | Removed 200 char truncation |
| `core/intent.py` | NEW - Semantic intent classifier |
| `core/generation/reply_generator.py` | Added `_get_rag_suggestions()` |
| `core/generation/context_analyzer.py` | Improved question detection patterns |
| `tests/test_intent.py` | NEW - 22 intent tests |
| `tests/test_context_analyzer.py` | NEW - 22 context tests |

---

## Appendix: Data Statistics

| Metric | Value |
|--------|-------|
| Total indexed messages | 336,835 |
| Unique conversations | 796 |
| Your messages | 167,311 |
| Embedding model | all-MiniLM-L6-v2 |
| LLM model | LFM2.5-1.2B |
