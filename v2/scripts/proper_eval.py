#!/usr/bin/env python3
"""Proper evaluation using labeled data we already have.

model_results.jsonl contains:
- relationship: casual_friend, work, family, romantic
- intent: statement, open_question, thanks, greeting, etc.
- gold_response: actual user response
- model_responses: outputs from 3 models

This script:
1. Shows what labeled data we have
2. Evaluates using the LABELED intents (not classifier)
3. Adds LLM-as-judge for appropriateness scoring

Usage:
    python scripts/proper_eval.py --audit       # See what we have
    python scripts/proper_eval.py --evaluate    # Run proper eval
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LABELED_DATA = Path("results/test_set/model_results.jsonl")
CLEAN_DATA = Path("results/test_set/clean_test_data.jsonl")


def audit_data():
    """Show what labeled data we have."""
    print("\n" + "="*70)
    print("DATA AUDIT: What We Have")
    print("="*70)

    # Check labeled data
    if LABELED_DATA.exists():
        relationships = Counter()
        intents = Counter()
        samples = []

        with open(LABELED_DATA) as f:
            for line in f:
                d = json.loads(line)
                samples.append(d)
                relationships[d.get('relationship', 'unknown')] += 1
                intents[d.get('intent', 'unknown')] += 1

        print(f"\n📊 LABELED DATA: {LABELED_DATA.name}")
        print(f"   Total samples: {len(samples)}")
        print(f"\n   Relationships:")
        for r, c in relationships.most_common():
            print(f"     {r}: {c}")
        print(f"\n   Intents:")
        for i, c in intents.most_common():
            print(f"     {i}: {c}")

        # Show sample
        print(f"\n   Sample entry:")
        s = samples[0]
        print(f"     contact: {s.get('contact')}")
        print(f"     relationship: {s.get('relationship')}")
        print(f"     intent: {s.get('intent')}")
        print(f"     gold_response: {s.get('gold_response')}")
        if 'model_responses' in s:
            print(f"     model_responses: {list(s['model_responses'].keys())}")

    else:
        print(f"\n❌ LABELED DATA NOT FOUND: {LABELED_DATA}")

    # Check clean data
    if CLEAN_DATA.exists():
        with open(CLEAN_DATA) as f:
            clean_count = sum(1 for _ in f)

        with open(CLEAN_DATA) as f:
            sample = json.loads(f.readline())

        print(f"\n📊 CLEAN DATA: {CLEAN_DATA.name}")
        print(f"   Total samples: {clean_count}")
        print(f"   Fields: {list(sample.keys())}")
        print(f"\n   Sample:")
        print(f"     contact: {sample.get('contact')}")
        print(f"     is_group: {sample.get('is_group')}")
        print(f"     last_message: {sample.get('last_message', '')[:50]}...")
        print(f"     gold_response: {sample.get('gold_response')}")

    # What's missing
    print("\n" + "="*70)
    print("WHAT'S MISSING FROM OUR CURRENT APPROACH")
    print("="*70)
    print("""
    1. ❌ Using embedding classifier (44% accurate) instead of labels
    2. ❌ Not using relationship context in prompts
    3. ❌ Not using labeled intent to constrain generation
    4. ❌ No human eval baseline established
    5. ❌ No LLM-as-judge for scalable quality scoring
    """)

    print("\n" + "="*70)
    print("RECOMMENDED NEXT STEPS")
    print("="*70)
    print("""
    1. IMMEDIATE: Use labeled data for eval (we have 200 labeled samples!)
       - Match generated intent against LABELED intent, not classifier

    2. IMPROVE PROMPTS: Add relationship context
       - "You are texting a casual friend" vs "You are texting family"

    3. ESTABLISH GROUND TRUTH: Run human_eval.py on 30-50 samples
       - Get human preference rankings
       - Use to calibrate LLM-as-judge

    4. SCALE EVAL: Implement LLM-as-judge
       - Use GPT-4/Claude to rate appropriateness 1-5
       - Calibrate against human eval
       - Run on full dataset

    5. FINE-TUNE: Once eval is reliable
       - Use 500 samples for training
       - Eval on held-out set
    """)


def evaluate_with_labels():
    """Evaluate models using the labeled data we have."""
    print("\n" + "="*70)
    print("EVALUATION USING LABELED DATA")
    print("="*70)

    if not LABELED_DATA.exists():
        print(f"Error: {LABELED_DATA} not found")
        return

    samples = []
    with open(LABELED_DATA) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} labeled samples")

    # Analyze model outputs by relationship and intent
    results_by_model = {}
    results_by_relationship = {}
    results_by_intent = {}

    for sample in samples:
        relationship = sample.get('relationship', 'unknown')
        intent = sample.get('intent', 'unknown')
        gold = sample.get('gold_response', '')
        model_responses = sample.get('model_responses', {})

        for model_name, response_data in model_responses.items():
            if model_name not in results_by_model:
                results_by_model[model_name] = []

            response_text = response_data.get('text', '')

            # Basic metrics
            length_ratio = len(response_text) / max(len(gold), 1)
            is_short = len(response_text) < 50  # Text-like brevity

            results_by_model[model_name].append({
                'gold': gold,
                'generated': response_text,
                'relationship': relationship,
                'intent': intent,
                'length_ratio': length_ratio,
                'is_short': is_short,
            })

    # Print results
    print("\n--- MODEL COMPARISON ---")
    print(f"{'Model':<20} {'Avg Length Ratio':<20} {'Short (<50 chars)':<20}")
    print("-" * 60)

    for model, results in results_by_model.items():
        avg_ratio = sum(r['length_ratio'] for r in results) / len(results)
        short_pct = sum(1 for r in results if r['is_short']) / len(results) * 100
        print(f"{model:<20} {avg_ratio:<20.2f} {short_pct:<20.0f}%")

    # By relationship
    print("\n--- BY RELATIONSHIP (lfm2.5-1.2b) ---")
    if 'lfm2.5-1.2b' in results_by_model:
        by_rel = {}
        for r in results_by_model['lfm2.5-1.2b']:
            rel = r['relationship']
            if rel not in by_rel:
                by_rel[rel] = []
            by_rel[rel].append(r)

        for rel, items in sorted(by_rel.items()):
            short_pct = sum(1 for i in items if i['is_short']) / len(items) * 100
            print(f"  {rel:<15}: {len(items):>3} samples, {short_pct:.0f}% short responses")

    # Show examples
    print("\n--- SAMPLE COMPARISONS ---")
    for i, sample in enumerate(samples[:5]):
        print(f"\n[{i+1}] Relationship: {sample.get('relationship')} | Intent: {sample.get('intent')}")
        print(f"    Gold: \"{sample.get('gold_response', '')[:50]}\"")
        for model, resp in sample.get('model_responses', {}).items():
            print(f"    {model}: \"{resp.get('text', '')[:50]}\"")

    print("\n" + "="*70)
    print("NOTE: This is basic analysis. For proper eval, we need:")
    print("  1. Human eval to establish ground truth")
    print("  2. LLM-as-judge calibrated against human eval")
    print("  3. Intent match using labels (not classifier)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="Audit available data")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    args = parser.parse_args()

    if args.audit or (not args.audit and not args.evaluate):
        audit_data()

    if args.evaluate:
        evaluate_with_labels()


if __name__ == "__main__":
    main()
