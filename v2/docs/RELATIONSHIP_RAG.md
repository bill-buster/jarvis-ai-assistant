# Relationship-Aware RAG Pipeline

## What We Built

A system that finds **few-shot examples from contacts with similar relationships** to improve reply generation. Instead of predicting response type (which failed at 22-44% accuracy), we show the model **similar SITUATIONS from similar people**.

### Example Flow

When generating a reply for **Ishani** (best friend):

```
Incoming: "wanna hang out this weekend?"
    │
    ▼
RelationshipRegistry
    → Ishani is category="friend"
    → Find 323 other friends
    │
    ▼
Phone-to-ChatID Resolution (cached)
    → Resolve friend phones to 77 real chat_ids
    │
    ▼
Reply-Pairs FAISS Index (620K pairs, 27ms search)
    → Find similar incoming messages from friends
    → Return YOUR actual replies to those messages
    │
    ▼
Few-shot examples for LLM:
    - "ok if we were to hangout now..." → "actually its only 50 damn"
    - "Wanna hang" → "I can't Im playing basketball"
    - "We should hang out" → "Yes we shud hang out"
```

## Architecture

### 1. RelationshipRegistry (`core/embeddings/relationship_registry.py`)
- Loads `results/contacts/contact_profiles.json` (562 contacts)
- Indexes by category: 324 friends, 24 family, 3 work, 211 other
- Maps phone numbers → contact names (505 phones)
- `get_similar_contacts(identifier)` → list of contacts in same category

### 2. Phone-to-ChatID Mapping (cached in `EmbeddingStore`)
- Queries DB once to find actual chat_ids for each phone
- Cached to `~/.jarvis/faiss_indices/phone_to_chatids.json`
- `resolve_phones_to_chatids(["+1555..."])` → real chat_ids

### 3. Reply-Pairs Index (`EmbeddingStore`)
- Pre-computed (their_message → your_reply) pairs
- **620,443 pairs** indexed with HNSW
- Cached to `~/.jarvis/faiss_indices/reply_pairs_index.faiss`
- **27ms search** (after warm-up)

### 4. Integration (`ReplyGenerator._find_past_replies`)
- Same-conversation search (boosted +0.05)
- Cross-conversation search (filtered by relationship)
- Results merged, deduplicated, passed as few-shot examples

## Performance

| Metric | Value |
|--------|-------|
| Reply pairs indexed | 620,443 |
| Search time (cold) | ~9 seconds (loads index) |
| Search time (warm) | **27ms** |
| Phone-to-chatid cache | ~1 second to load |
| Relationship registry | <100ms to load |

## How to Test

```bash
# From repo root
cd jarvis-ai-assistant
uv run python v2/scripts/test_relationship_rag.py
```

Or quick test:
```python
from core.embeddings import get_embedding_store
store = get_embedding_store()

# Global search (all conversations)
results = store.find_your_past_replies_cross_conversation(
    "wanna hang out?",
    target_chat_ids=None,  # None = global
    limit=5,
    min_similarity=0.5
)
for msg, reply, score, chat_id in results:
    print(f"[{score:.2f}] \"{msg}\" -> \"{reply}\"")
```

## How to Evaluate Improvement

### Baseline (what we had before)
- Intent classification accuracy: 22-44%
- Few-shot examples from same conversation only
- No cross-conversation learning

### New System Metrics

1. **Few-shot coverage**: % of generations that include past reply examples
   ```python
   result = generator.generate_replies(messages, chat_id=chat_id)
   has_examples = len(result.past_replies) > 0
   ```

2. **Semantic similarity of generated replies to past replies**:
   - Compare generated reply embedding to past reply embeddings
   - Higher similarity = more consistent with user's actual style

3. **Manual evaluation**:
   - Generate replies for 50 test messages
   - Rate: Does it sound like something you'd actually say? (1-5)
   - Compare with/without relationship-aware RAG

### Evaluation Script

```python
# Run this to compare with baseline
from core.embeddings import get_embedding_store
from core.generation import ReplyGenerator

# Test messages
test_cases = [
    {"text": "wanna hang?", "relationship": "friend"},
    {"text": "dinner tonight?", "relationship": "family"},
    {"text": "meeting at 3?", "relationship": "work"},
]

for case in test_cases:
    # With relationship filtering
    results_filtered = store.find_your_past_replies_cross_conversation(
        case["text"],
        target_chat_ids=get_chat_ids_for_category(case["relationship"]),
        limit=5
    )

    # Without filtering (global)
    results_global = store.find_your_past_replies_cross_conversation(
        case["text"],
        target_chat_ids=None,
        limit=5
    )

    print(f"Query: {case['text']}")
    print(f"  Filtered ({case['relationship']}): {len(results_filtered)} results")
    print(f"  Global: {len(results_global)} results")
```

## What Success Looks Like

1. **Coverage**: >80% of generations include at least 1 past reply example
2. **Relevance**: Past replies are semantically similar to the query (>0.5 score)
3. **Quality**: Generated replies match user's actual texting style
4. **Speed**: <100ms total for RAG lookup (achieved: 27ms)

## Files Changed

| File | Change |
|------|--------|
| `core/embeddings/relationship_registry.py` | NEW - Contact relationship lookup |
| `core/embeddings/store.py` | Phone-to-chatid mapping, reply-pairs index |
| `core/generation/reply_generator.py` | Cross-conversation search integration |
| `core/embeddings/__init__.py` | Export new classes |
| `tests/test_relationship_registry.py` | NEW - 25 unit tests |
| `scripts/test_relationship_rag.py` | NEW - Integration test script |

## Caches

All caches stored in `~/.jarvis/faiss_indices/`:
- `reply_pairs_index.faiss` - FAISS index (620K vectors)
- `reply_pairs_index.meta.json` - Metadata with replies
- `phone_to_chatids.json` - Phone → chat_id mapping
- `global_index.faiss` - Full message index (optional)
