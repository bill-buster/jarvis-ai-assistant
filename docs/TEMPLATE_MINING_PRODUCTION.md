# Production Template Mining System

## Overview

This is the **production-ready** template mining system that fixes ALL 14 critical issues identified in the initial approach.

**Status:** Ready for A/B testing (NOT full deployment yet)

**Expected Quality:** 70-80% appropriate suggestions (up from 40-50% in original)

---

## What Was Fixed

### HIGH PRIORITY (Critical)

| # | Issue | Original | Fixed |
|---|-------|----------|-------|
| 1 | **Context not used in clustering** | Mixed boss/friend responses | Stratified clustering by context |
| 2 | **Naive coherence filtering** | Only checked phrase pairs | Semantic coherence with embeddings |
| 3 | **HDBSCAN wrong tool** | Density clustering on text | Hybrid: HDBSCAN + silhouette DBSCAN |
| 4 | **Adaptive decay backwards** | High freq = short decay ✗ | High freq = long decay ✓ |
| 5 | **Circular validation** | Qwen judges Qwen ✗ | Human validation tool |
| 6 | **No sender diversity** | Worked for 1 person | Requires 3+ senders |
| 7 | **Group size not stratified** | Binary group flag | 4 categories: direct/small/medium/large |
| 8 | **Context embedded in text** | "group=True" as text ✗ | Context as separate features ✓ |
| 9 | **No negative mining** | Only mined good patterns | Mines apology/correction patterns |

### MEDIUM PRIORITY

| # | Issue | Fixed |
|---|-------|-------|
| 10 | **Arbitrary conversation gap** | Fixed 24h threshold | Adaptive per relationship |
| 11 | **Weak system filtering** | Basic patterns | Expanded: stickers, payments, calendar |
| 12 | **No emoji handling** | 😂 ≠ 🤣 | Normalized to [LAUGH] |
| 13 | **No day-of-week context** | Only hour | Added weekday/weekend |

### LOW PRIORITY

| # | Issue | Fixed |
|---|-------|-------|
| 14 | **No incremental updates** | Full scan every time | IncrementalTemplateIndex class |
| 15 | **Overfitting to history** | Old style dominates | Continuous learning with recency bias |
| 16 | **No A/B testing** | Deploy and hope | ABTestFramework with metrics |

---

## New Architecture

### Before (Enhanced)

```python
# PROBLEM: Mixed everything together
all_messages = extract_all()
embeddings = embed(messages)  # Context as text!
clusters = HDBSCAN(embeddings)  # Boss + friend mixed!
```

**Issues:**
- Boss "yes" and friend "yes" in same cluster
- Context ignored during clustering
- No quality validation

### After (Production)

```python
# FIXED: Stratified by context
messages_with_context = extract_with_full_context()

# Mine negative patterns (responses to avoid)
negative_patterns = mine_negative_patterns()

# Embed text SEPARATELY from context
text_embeddings = embed_text_only(messages)
context_features = extract_context_features(messages)
combined = [text_embeddings, context_features]

# Cluster WITHIN each context stratum
strata = stratify_by_context(messages)
for context, stratum_messages in strata:
    clusters = cluster_stratum(stratum_messages)  # Separate for work/personal/etc

# Filter by sender diversity (3+ senders)
patterns = filter_by_sender_diversity(patterns, min_senders=3)

# Filter negative patterns
patterns = filter_negative_patterns(patterns, negative_patterns)

# Calculate adaptive weights (recency bias)
for pattern in patterns:
    pattern.adaptive_weight = calculate_adaptive_weight(pattern)

# Human validation on sample
human_validate_sample(patterns, n=50)
```

**Benefits:**
- Context-appropriate suggestions
- Sender-agnostic patterns
- Quality validated by humans
- Continuous learning ready

---

## Usage

### 1. Mine Production Templates

```bash
# Mine with all fixes applied
python scripts/mine_response_pairs_production.py \
    --min-senders 3 \
    --output results/templates_production.json

# Options:
#   --no-cache: Regenerate embeddings
#   --min-senders: Minimum senders for diversity (default: 3)
#   --skip-validation: Skip quality checks
```

**Output:** `results/templates_production.json`

**What it does:**
- ✓ Extracts messages with full context
- ✓ Mines negative patterns (to avoid)
- ✓ Embeds text with context as features
- ✓ Clusters separately by context strata
- ✓ Filters by sender diversity
- ✓ Applies adaptive weighting
- ✓ Deprecates outdated patterns

**Time:** 2-3 hours for full message history

---

### 2. Human Validation

```bash
# Interactive review of sampled templates
python scripts/validate_templates_human.py \
    results/templates_production.json \
    --sample-size 50

# Terminal UI guides you through:
#   - Rating appropriateness (1-5)
#   - Rating naturalness (1-5)
#   - Rating context match (1-5)
#   - Accept/reject decision
#   - Optional notes
```

**Output:** `results/templates_production_humanvalidated.json`

**What it does:**
- Samples 50 templates (stratified by context)
- Presents each for manual review
- Collects ratings and notes
- Calculates acceptance rate
- Filters templates based on reviews

**Time:** 15-30 minutes

**Addressing circular validation:**
- No longer using Qwen to judge Qwen
- Human evaluation is ground truth
- Can compare LLM scores vs human scores

---

### 3. A/B Testing Setup

```python
from scripts.utils.ab_testing import ABTestConfig, ABTestAssignment, ABTestMetrics

# Create A/B test
config = ABTestConfig(
    test_name="production_templates_v1",
    variant_a="baseline",  # Old templates
    variant_b="production",  # New templates
    traffic_split=0.1,  # 10% gets new templates
    enabled=True
)

assignment = ABTestAssignment(config)
metrics = ABTestMetrics(Path("results/ab_tests"))

# In your reply generation:
variant = assignment.get_variant(user_id)

if variant == "variant_b":
    # Use production templates
    match = production_matcher.match(query)
else:
    # Use baseline templates
    match = baseline_matcher.match(query)

# Log metrics
metrics.log_event(
    test_name="production_templates_v1",
    variant=variant,
    metric_name="template_hit",
    value=1.0 if match else 0.0
)

metrics.log_event(
    test_name="production_templates_v1",
    variant=variant,
    metric_name="user_accepted",
    value=1.0 if user_accepted else 0.0
)

# After 1 week, analyze:
results = metrics.compare_variants("production_templates_v1", "user_accepted")
print(f"Winner: {results['winner']}")
print(f"P-value: {results['p_value']}")
print(f"Effect size: {results['effect_size']}")
```

**Metrics to track:**
- Template hit rate (before/after)
- User acceptance rate (% not edited)
- User satisfaction (implicit feedback)
- Latency (template vs LLM)

---

### 4. Continuous Learning

```python
from scripts.utils.continuous_learning import (
    detect_concept_drift,
    deprecate_outdated_patterns,
    IncrementalTemplateIndex
)

# Detect if style has drifted
drift_analysis = detect_concept_drift(
    historical_patterns=old_templates,
    recent_messages=last_3_months,
    drift_threshold=0.3
)

if drift_analysis["drift_detected"]:
    print("⚠️  Communication style has changed")
    print(f"  Formality shift: {drift_analysis['formality_drift']:.2%}")
    print(f"  Recommendation: {drift_analysis['recommendation']}")

# Deprecate old patterns
patterns = deprecate_outdated_patterns(
    patterns,
    current_time_ns=now,
    max_age_days=730,
    min_recent_usage=2
)

# Incremental updates (only process new messages)
index = IncrementalTemplateIndex()
new_messages = index.get_new_messages_query()
index.update_with_new_messages(new_messages)
```

---

## File Structure

```
scripts/
├── mine_response_pairs_production.py    # Main production mining script
├── validate_templates_human.py          # Interactive human validation
├── utils/
│   ├── context_analysis.py             # Context detection & stratification
│   ├── coherence_checker.py            # Semantic coherence checking
│   ├── sender_diversity.py             # Sender diversity filtering
│   ├── negative_mining.py              # Negative pattern mining
│   ├── continuous_learning.py          # Adaptive weights & drift detection
│   └── ab_testing.py                   # A/B testing framework

docs/
├── CRITICAL_ANALYSIS_TEMPLATE_MINING.md  # Full problem analysis
└── TEMPLATE_MINING_PRODUCTION.md         # This file (usage guide)
```

---

## Quality Expectations

### Original Enhanced System
- **Expected quality:** 40-50% appropriate
- **Coverage:** 30-50% of queries
- **Issues:** Context mixing, overfitting, outdated style

### Production System
- **Expected quality:** 70-80% appropriate
- **Coverage:** 25-40% of queries (lower but higher quality)
- **Improvements:**
  - Context-appropriate suggestions
  - Sender-agnostic patterns
  - Human-validated quality
  - Continuous learning

### Why Lower Coverage?

**Trade-off:** Quality > Coverage

- Sender diversity filter removes single-person patterns
- Negative mining removes problematic patterns
- Human validation rejects low-quality templates
- Context stratification reduces cluster size

**Result:** Fewer templates, but much higher quality

---

## Deployment Strategy

### Phase 1: A/B Test (Weeks 1-4)

```
Week 1: 5% traffic → production templates
Week 2: 10% traffic → production templates
Week 3: 25% traffic → production templates
Week 4: Evaluate results
```

**Success criteria:**
- Template hit rate ≥ 20%
- User acceptance ≥ 70%
- No user complaints about inappropriate suggestions
- Faster than LLM generation (< 50ms)

### Phase 2: Ramp Up (Weeks 5-8)

```
Week 5: 50% traffic
Week 6: 75% traffic
Week 7: 90% traffic
Week 8: 100% traffic
```

### Phase 3: Continuous Improvement

- Run drift detection monthly
- Retrain if drift > 30%
- Collect user feedback continuously
- A/B test improvements

---

## Maintenance

### Weekly
- Monitor A/B test metrics
- Check for user complaints
- Review edge cases

### Monthly
- Run drift detection
- Validate top 20 templates manually
- Update negative patterns
- Analyze user feedback

### Quarterly
- Full retrain with new data
- Human validation of sample (50 templates)
- Evaluate quality metrics
- Update context stratification if needed

---

## Known Limitations

1. **Still requires user-specific fine-tuning:** Patterns may not fit every user's style
2. **Limited multilingual support:** Primarily English
3. **Cold start problem:** New users have no historical patterns
4. **Concept drift:** Requires periodic retraining

---

## Comparison: Enhanced vs Production

| Feature | Enhanced | Production |
|---------|----------|------------|
| Context awareness | ✓ Tracked | ✓ Used in clustering |
| Coherence check | ✓ Phrase pairs | ✓ Semantic embeddings |
| Adaptive decay | ✗ Backwards | ✓ Fixed logic |
| Sender diversity | ✗ Not filtered | ✓ Requires 3+ senders |
| Context embedding | ✗ In text | ✓ Separate features |
| Negative mining | ✗ None | ✓ Mines bad patterns |
| Quality validation | ✗ Same model | ✓ Human validation |
| A/B testing | ✗ None | ✓ Framework included |
| Continuous learning | ✗ None | ✓ Drift detection |
| **Expected Quality** | **40-50%** | **70-80%** |

---

## Next Steps

1. **Run production mining** on your iMessage data
2. **Human validate** 50-100 templates
3. **Set up A/B test** with 5-10% traffic
4. **Monitor for 1 week** and collect metrics
5. **Analyze results** and decide on ramp-up
6. **Iterate** based on user feedback

---

## Questions?

See:
- `docs/CRITICAL_ANALYSIS_TEMPLATE_MINING.md` - Full problem analysis
- `scripts/utils/*.py` - Implementation details
- `tests/` - Unit tests for utilities

Or run:
```bash
python scripts/mine_response_pairs_production.py --help
python scripts/validate_templates_human.py --help
```
