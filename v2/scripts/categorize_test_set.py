#!/usr/bin/env python3
"""Categorize test set samples by achievability.

Some gold responses require specific knowledge the model can't have.
This script categorizes samples so we can measure the right things.

Categories:
- STYLE: Generic responses where matching style is the goal (lol, ok, sounds good)
- OPINION: Expressing preference/opinion (I like X, that's cool)
- QUESTION: Asking a question back
- KNOWLEDGE: Requires specific facts not in context (dates, names, addresses)
- CONTINUATION: Continuing a thought that requires memory

Usage:
    python scripts/categorize_test_set.py           # Categorize and show stats
    python scripts/categorize_test_set.py --save    # Save categorized version
    python scripts/categorize_test_set.py --examples STYLE  # Show examples of category
"""

import argparse
import json
import re
from pathlib import Path

TEST_SET_FILE = Path("results/test_set/test_data.jsonl")
CATEGORIZED_FILE = Path("results/test_set/test_data_categorized.jsonl")


def categorize_response(gold: str, context: str) -> tuple[str, str]:
    """Categorize a gold response by what's required to generate it.

    Returns: (category, reason)
    """
    gold_lower = gold.lower().strip()
    gold_words = set(gold_lower.split())
    context_lower = context.lower()

    # === STYLE responses (short, generic, achievable) ===

    # Very short reactions
    if len(gold) < 15:
        short_patterns = {
            "lol", "haha", "hahaha", "ok", "okay", "k", "bet", "yea", "yeah",
            "yes", "no", "nah", "nice", "cool", "true", "fr", "facts", "same",
            "damn", "wow", "oof", "rip", "valid", "gotcha", "got it", "sure",
            "sounds good", "im down", "i'm down", "down", "word", "aight",
        }
        if gold_lower in short_patterns or gold_lower.rstrip("!?.") in short_patterns:
            return "STYLE", "short_reaction"

    # Laughter variations
    if re.match(r'^(ha)+$', gold_lower) or re.match(r'^lo+l+$', gold_lower):
        return "STYLE", "laughter"

    # Simple acknowledgments (under 25 chars, no proper nouns)
    if len(gold) < 25 and not re.search(r'[A-Z][a-z]+', gold[1:]):  # Skip first char
        ack_starters = ["ok", "got", "sounds", "cool", "nice", "yea", "yeah", "sure", "bet"]
        if any(gold_lower.startswith(s) for s in ack_starters):
            return "STYLE", "acknowledgment"

    # === QUESTION responses ===
    if gold.rstrip().endswith("?"):
        # Check if the question words are inferable from context
        # vs requiring specific knowledge
        question_words = {"what", "when", "where", "who", "why", "how", "do", "does", "did", "can", "will", "would"}
        if gold_words & question_words:
            # Check if question contains specific entities not in context
            # Extract potential entities (capitalized words, numbers)
            entities = re.findall(r'\b[A-Z][a-z]+\b', gold)
            numbers = re.findall(r'\b\d+\b', gold)

            # If entities/numbers are in context, it's achievable
            entities_in_context = all(e.lower() in context_lower for e in entities)
            if entities_in_context and not numbers:
                return "QUESTION", "generic_question"
            else:
                return "KNOWLEDGE", "question_with_specifics"

    # === KNOWLEDGE responses (contain specific facts) ===

    # Contains specific times/dates
    if re.search(r'\b\d{1,2}(:\d{2})?\s*(am|pm|AM|PM)\b', gold):
        return "KNOWLEDGE", "specific_time"
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+', gold_lower):
        return "KNOWLEDGE", "specific_date"
    if re.search(r'\b\d{1,2}(st|nd|rd|th)\b', gold_lower):
        return "KNOWLEDGE", "specific_date"

    # Contains addresses or locations
    if re.search(r'\b\d+\s+[A-Za-z]+\s+(street|st|ave|avenue|road|rd|blvd|drive|dr|lane|ln)\b', gold_lower):
        return "KNOWLEDGE", "address"

    # Contains proper nouns not in context
    proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', gold)
    # Filter out sentence starters
    if proper_nouns:
        # Check if first word - might just be sentence start
        first_word = gold.split()[0] if gold.split() else ""
        proper_nouns_not_first = [p for p in proper_nouns if p != first_word]

        if proper_nouns_not_first:
            # Check if these nouns are in the context
            nouns_not_in_context = [p for p in proper_nouns_not_first if p.lower() not in context_lower]
            if nouns_not_in_context:
                return "KNOWLEDGE", f"proper_nouns: {', '.join(nouns_not_in_context[:3])}"

    # Contains numbers that seem like specific data
    numbers = re.findall(r'\b\d+\b', gold)
    if numbers:
        # Check if these numbers are in context
        numbers_not_in_context = [n for n in numbers if n not in context]
        if numbers_not_in_context and any(int(n) > 10 for n in numbers_not_in_context if n.isdigit()):
            return "KNOWLEDGE", f"specific_numbers: {', '.join(numbers_not_in_context[:3])}"

    # === OPINION responses ===
    opinion_markers = ["i think", "i like", "i love", "i hate", "i want", "i need",
                       "i feel", "i guess", "i mean", "imo", "tbh"]
    if any(m in gold_lower for m in opinion_markers):
        # Opinions are somewhat achievable if they match your patterns
        return "OPINION", "subjective"

    # === CONTINUATION (continuing a thought) ===
    if gold_lower.startswith(("and ", "but ", "also ", "plus ", "like ")):
        return "CONTINUATION", "continues_thought"

    # === Default: Check length and complexity ===
    if len(gold) > 60:
        return "KNOWLEDGE", "long_response_likely_specific"

    # Medium length, no obvious knowledge requirement
    if len(gold) < 40:
        return "STYLE", "medium_generic"

    return "OPINION", "unclear"


def categorize_test_set():
    """Categorize all samples in the test set."""
    if not TEST_SET_FILE.exists():
        print(f"Test set not found: {TEST_SET_FILE}")
        return []

    samples = []
    with open(TEST_SET_FILE) as f:
        for line in f:
            samples.append(json.loads(line))

    categorized = []
    for s in samples:
        gold = s["gold_response"]
        context = s.get("conversation", s.get("prompt", ""))

        category, reason = categorize_response(gold, context)

        categorized.append({
            **s,
            "category": category,
            "category_reason": reason,
        })

    return categorized


def show_stats(samples: list[dict]):
    """Show categorization stats."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("TEST SET CATEGORIZATION")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")

    # By category
    by_cat = Counter(s["category"] for s in samples)
    print("\nBy category:")
    for cat, count in by_cat.most_common():
        pct = count / len(samples) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {cat:12} {bar} {count:3d} ({pct:5.1f}%)")

    # By reason within each category
    print("\nDetailed breakdown:")
    for cat in ["STYLE", "OPINION", "QUESTION", "KNOWLEDGE", "CONTINUATION"]:
        cat_samples = [s for s in samples if s["category"] == cat]
        if not cat_samples:
            continue
        print(f"\n  {cat} ({len(cat_samples)} samples):")
        reasons = Counter(s["category_reason"] for s in cat_samples)
        for reason, count in reasons.most_common(5):
            print(f"    - {reason}: {count}")

    # Achievable vs not
    achievable = sum(1 for s in samples if s["category"] in ["STYLE", "OPINION", "QUESTION"])
    print(f"\n{'='*60}")
    print(f"ACHIEVABLE (STYLE + OPINION + QUESTION): {achievable}/{len(samples)} ({achievable/len(samples)*100:.1f}%)")
    print(f"KNOWLEDGE-REQUIRED: {len(samples) - achievable}/{len(samples)} ({(len(samples)-achievable)/len(samples)*100:.1f}%)")
    print(f"{'='*60}")

    # Length distribution by category
    print("\nAvg gold response length by category:")
    for cat in ["STYLE", "OPINION", "QUESTION", "KNOWLEDGE"]:
        cat_samples = [s for s in samples if s["category"] == cat]
        if cat_samples:
            avg_len = sum(len(s["gold_response"]) for s in cat_samples) / len(cat_samples)
            print(f"  {cat:12}: {avg_len:5.1f} chars")


def show_examples(samples: list[dict], category: str, n: int = 5):
    """Show examples from a category."""
    cat_samples = [s for s in samples if s["category"] == category]

    print(f"\n{'='*60}")
    print(f"EXAMPLES: {category} ({len(cat_samples)} total)")
    print("=" * 60)

    import random
    random.shuffle(cat_samples)

    for s in cat_samples[:n]:
        print(f"\n[{s['contact']}] - {s['category_reason']}")
        print("-" * 40)

        # Show last 4 lines of conversation
        conv_lines = s.get("conversation", s.get("prompt", "")).split("\n")
        conv_lines = [l for l in conv_lines if l.strip() and not l.startswith("[")]
        for line in conv_lines[-4:]:
            print(f"  {line}")

        print(f"\n  → GOLD: \"{s['gold_response']}\"")


def save_categorized(samples: list[dict]):
    """Save categorized test set."""
    CATEGORIZED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATEGORIZED_FILE, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved: {CATEGORIZED_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save categorized version")
    parser.add_argument("--examples", type=str, metavar="CATEGORY",
                        help="Show examples (STYLE, OPINION, QUESTION, KNOWLEDGE)")
    parser.add_argument("-n", type=int, default=5, help="Number of examples")
    args = parser.parse_args()

    samples = categorize_test_set()
    if not samples:
        return

    if args.examples:
        show_examples(samples, args.examples.upper(), args.n)
    else:
        show_stats(samples)

    if args.save:
        save_categorized(samples)


if __name__ == "__main__":
    main()
