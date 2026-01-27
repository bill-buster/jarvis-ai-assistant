# Template System Workflow

## Overview
The template system provides a fast path for common responses using semantic similarity matching, bypassing the need to load the full MLX model.

---

## 🔄 Complete Text Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│                  "thanks for the update"                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INTENT CLASSIFICATION                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ IntentClassifier (jarvis/intent.py)                      │  │
│  │ • Uses sentence transformer embeddings                   │  │
│  │ • Routes to: REPLY, SEARCH, SUMMARIZE, etc.             │  │
│  │ • Extracts params: person_name, search_query, etc.      │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPLATE MATCHING                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TemplateMatcher (models/templates.py)                    │  │
│  │                                                            │  │
│  │ Step 1: Load sentence transformer                        │  │
│  │   Model: all-MiniLM-L6-v2 (~80MB)                        │  │
│  │                                                            │  │
│  │ Step 2: Encode query                                      │  │
│  │   query → embedding vector (384 dims)                     │  │
│  │   ✓ Uses LRU cache (500 queries)                         │  │
│  │                                                            │  │
│  │ Step 3: Compute similarity                                │  │
│  │   cosine_sim = dot(query_emb, pattern_emb) / norms       │  │
│  │   ✓ Pre-normalized embeddings (computed once)            │  │
│  │   ✓ Batch computation for all 1000+ patterns             │  │
│  │                                                            │  │
│  │ Step 4: Check threshold                                   │  │
│  │   if similarity >= 0.7:                                   │  │
│  │     return template response                              │  │
│  │   else:                                                    │  │
│  │     fall through to MLX model                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    ┌───────┴────────┐
                    │                │
            ✓ Match Found      ✗ No Match
            (similarity≥0.7)   (similarity<0.7)
                    │                │
                    ▼                ▼
         ┌────────────────┐   ┌──────────────────┐
         │ FAST PATH      │   │ SLOW PATH        │
         │ Return Template│   │ Load MLX Model   │
         │ Response       │   │ Generate Text    │
         │ (~10ms)        │   │ (~2-5s)          │
         └────────────────┘   └──────────────────┘
```

---

## 🎯 Template Matching Details

### Input Query Processing

```
┌────────────────────────────────────────────────────────────────┐
│                    match_with_context()                        │
│                                                                │
│  Input:                                                        │
│    • query: "thanks for the update"                           │
│    • group_size: 5 (optional)                                 │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 1. Check cache                                            ││
│  │    • Hash query → MD5                                     ││
│  │    • Lookup in LRU cache (500 entries)                   ││
│  │    • Cache hit rate: ~70% for repeated queries           ││
│  └──────────────────────────────────────────────────────────┘│
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 2. Encode query (if cache miss)                          ││
│  │    • Load sentence transformer                            ││
│  │    • Encode: query → vector[384]                         ││
│  │    • Store in cache                                       ││
│  └──────────────────────────────────────────────────────────┘│
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 3. Compute similarities                                   ││
│  │    • For each pattern embedding (1000+ patterns):        ││
│  │      similarity = dot(query_emb, pattern_emb) /          ││
│  │                  (||query_emb|| * ||pattern_emb||)       ││
│  │    • Uses pre-normalized embeddings (O(n) not O(n*d))   ││
│  └──────────────────────────────────────────────────────────┘│
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 4. Filter by group size (if provided)                    ││
│  │    • Skip templates where:                                ││
│  │      - template.min_group_size > group_size              ││
│  │      - template.max_group_size < group_size              ││
│  │    • Boost group templates by +0.05 in groups (≥3)      ││
│  └──────────────────────────────────────────────────────────┘│
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 5. Return best match                                      ││
│  │    • Find max similarity                                  ││
│  │    • If ≥ 0.7: return TemplateMatch                      ││
│  │    • If < 0.7: return None → fall back to MLX           ││
│  └──────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

### Example Matching Process

```
Query: "thanks for the update"
↓
Encode → [0.12, -0.34, 0.56, ..., 0.23]  (384 dims)
↓
Compare against all patterns:
  Pattern: "Thanks for sending the report"     → similarity: 0.89 ✓
  Pattern: "Thank you for the information"     → similarity: 0.85 ✓
  Pattern: "Confirming our meeting tomorrow"   → similarity: 0.32 ✗
  Pattern: "I vote for option A"               → similarity: 0.11 ✗
  ...
↓
Best match: "Thank you for the information" (0.89 ≥ 0.7)
↓
Return: "You're welcome! Let me know if you need anything else."
```

---

## 📊 How to Measure Template Coverage

### What is Template Coverage?

**Coverage** = Percentage of user queries that match templates (similarity ≥ 0.7) vs. queries that fall through to the MLX model.

### Metrics to Track

```
┌──────────────────────────────────────────────────────────────┐
│                   Template Analytics                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Hit Rate                                                 │
│     • % of queries that matched templates (≥0.7)           │
│     • Formula: hits / (hits + misses)                       │
│     • Target: >60% for casual iMessage use                  │
│                                                              │
│  2. Similarity Distribution                                  │
│     • Histogram of similarity scores                         │
│     • Identify "near misses" (0.6-0.69)                     │
│     • Candidates for new templates                          │
│                                                              │
│  3. Cache Efficiency                                         │
│     • Query embedding cache hit rate                         │
│     • Should be >70% for repeated queries                   │
│                                                              │
│  4. Per-Template Usage                                       │
│     • Which templates are used most?                        │
│     • Which are never used? (candidates for removal)        │
│                                                              │
│  5. Group vs 1:1 Coverage                                    │
│     • Do group chats have higher hit rates?                 │
│     • Are group templates being matched correctly?          │
│                                                              │
│  6. Missed Queries                                           │
│     • Track queries that didn't match any template          │
│     • Cluster similar misses → new template opportunities   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Current Analytics Implementation

Your codebase already has `jarvis/metrics.py` with `TemplateAnalytics` class:

```python
from jarvis.metrics import get_template_analytics

analytics = get_template_analytics()

# After matching
if match:
    analytics.record_hit(template_name, similarity)
else:
    analytics.record_miss(query, best_similarity)

# Get stats
stats = analytics.get_stats()
# {
#   "total_queries": 1000,
#   "hits": 650,
#   "misses": 350,
#   "hit_rate": 0.65,
#   "top_templates": [("quick_ok", 120), ("quick_thanks", 98), ...],
#   "missed_queries": [("what's for dinner tonight", 0.62), ...]
# }
```

### How to Evaluate Coverage on Real Data

```
┌─────────────────────────────────────────────────────────────┐
│              Coverage Evaluation Process                    │
└─────────────────────────────────────────────────────────────┘

Step 1: Collect Real Queries
  • Export recent iMessage conversations
  • Extract individual messages (exclude images/links)
  • Sample: 1000-5000 messages

Step 2: Simulate Template Matching
  for each message in sample:
      match = template_matcher.match(message)
      if match and match.similarity >= 0.7:
          record_hit(match.template.name, match.similarity)
      else:
          record_miss(message, best_similarity)

Step 3: Analyze Results
  • Calculate hit rate
  • Identify top templates
  • Find "near miss" clusters (0.6-0.69)
  • Detect query types with low coverage

Step 4: Improve Templates
  • Add templates for high-frequency misses
  • Remove unused templates
  • Tune similarity threshold if needed
```

---

## 🧪 Example Coverage Test

```python
# benchmarks/templates/run.py (to be created)

import json
from collections import Counter, defaultdict
from pathlib import Path

from models.templates import TemplateMatcher, _load_templates
from integrations.imessage.reader import ChatDBReader

def evaluate_coverage():
    """Evaluate template coverage on real iMessage data."""

    # Load templates
    matcher = TemplateMatcher()

    # Load recent messages (last 1000)
    reader = ChatDBReader()
    messages = reader.get_recent_messages(limit=1000)

    # Track results
    hits = []
    misses = []
    similarity_scores = []
    template_usage = Counter()

    for msg in messages:
        text = msg.text
        if not text or len(text) < 3:
            continue

        match = matcher.match(text)
        similarity_scores.append(match.similarity if match else 0.0)

        if match and match.similarity >= 0.7:
            hits.append({
                "query": text,
                "template": match.template.name,
                "similarity": match.similarity
            })
            template_usage[match.template.name] += 1
        else:
            misses.append({
                "query": text,
                "best_similarity": match.similarity if match else 0.0
            })

    # Calculate metrics
    total = len(hits) + len(misses)
    hit_rate = len(hits) / total if total > 0 else 0

    # Find near misses (0.6-0.69)
    near_misses = [m for m in misses if 0.6 <= m["best_similarity"] < 0.7]

    # Unused templates
    all_templates = {t.name for t in matcher.templates}
    used_templates = set(template_usage.keys())
    unused_templates = all_templates - used_templates

    return {
        "total_queries": total,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": hit_rate,
        "near_misses": len(near_misses),
        "near_miss_queries": [m["query"] for m in near_misses[:20]],
        "top_templates": template_usage.most_common(10),
        "unused_templates": list(unused_templates),
        "similarity_distribution": {
            "mean": sum(similarity_scores) / len(similarity_scores),
            "median": sorted(similarity_scores)[len(similarity_scores) // 2],
            "min": min(similarity_scores),
            "max": max(similarity_scores),
        }
    }
```

---

## 🎯 Key Insights

### Fast Path Advantages
- **Speed**: 10-50ms vs 2-5s for MLX generation
- **Memory**: 80MB (sentence transformer) vs 2-4GB (MLX model)
- **Consistency**: Deterministic responses
- **Cost**: No compute cost for inference

### When Templates Work Best
- Short, common phrases ("thanks", "ok", "on my way")
- Social coordination ("what time?", "where are we meeting?")
- Emotional responses ("lol", "congrats", "happy birthday")
- Assistant queries ("summarize my messages", "find texts from X")

### When Templates Don't Work
- Complex, context-specific questions
- Novel phrasing not seen in patterns
- Queries requiring reasoning or calculation
- Messages with names/dates/specifics that need personalization

### Coverage Targets by Intent Type

```
Intent Type          Target Hit Rate    Notes
──────────────────   ───────────────    ─────────────────────────
QUICK_REPLY          80-90%             "ok", "thanks", "lol"
GROUP_COORDINATION   60-70%             Scheduling, RSVP, polls
GENERAL_CHAT         40-50%             Conversational messages
REPLY                30-40%             Context-dependent
SEARCH               5-10%              Mostly falls to model
SUMMARIZE            5-10%              Mostly falls to model
```

---

## 🚀 Next Steps

1. **Create `benchmarks/templates/run.py`** to measure coverage on real data
2. **Collect baseline metrics** using your actual iMessage history
3. **Identify high-frequency misses** and add new templates
4. **Monitor cache efficiency** to ensure repeated queries are fast
5. **Track hit rate over time** to validate template additions

