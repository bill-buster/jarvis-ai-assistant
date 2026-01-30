#!/usr/bin/env python3
"""Experiment 3: Full conversation context + multiple suggestions.

Key insight: We were only using last_message, ignoring conversation context.

This experiment:
1. Provides full conversation context to the model
2. Optionally generates multiple suggestions (like Smart Reply)

Usage:
    python scripts/exp3_full_context.py --samples 50
    python scripts/exp3_full_context.py --samples 50 --multi-suggest
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CLEAN_TEST_SET = Path("results/test_set/clean_test_data.jsonl")
RESULTS_DIR = Path("results/experiments")


# =============================================================================
# PROMPTS WITH FULL CONTEXT
# =============================================================================

# Single response prompt with full context
FULL_CONTEXT_PROMPT = """You are texting as me. Reply naturally based on the conversation.
Casual texting style: brief, lowercase, no formal punctuation.

{conversation}
me:"""

# Few-shot version with context
FEWSHOT_CONTEXT_PROMPT = """You are texting as me. Here are examples of how I text:

Example 1:
them: wanna hang tonight?
me: what time
them: like 8?
me: yeah down

Example 2:
them: did you see the game
them: that was insane
me: fr that ending was wild

Example 3:
them: can you pick up milk
me: yeah ill grab some
them: thanks!
me: np

Now continue this conversation as me (brief, casual):

{conversation}
me:"""

# Roleplay + context
ROLEPLAY_CONTEXT_PROMPT = """You ARE me in this text conversation. Don't be an assistant - just text back naturally.
Style: casual, brief (1-8 words), lowercase, use "lol" "fr" "bet" when appropriate.

{conversation}
me:"""

# Multi-suggest prompt (generate 3 options)
MULTI_SUGGEST_PROMPT = """Generate 3 different reply options for this text conversation.
Each reply should be casual texting style (brief, lowercase).

{conversation}

Option 1 (accept/positive):
Option 2 (question/clarify):
Option 3 (reaction/casual):"""


# =============================================================================
# RESPONSE CLASSIFIER
# =============================================================================

class ResponseClassifier:
    """Classifies what intent a response has."""

    RESPONSE_ANCHORS = {
        "accept": [
            "yes", "yeah", "sure", "sounds good", "down", "i'm in", "definitely",
            "yep", "let's do it", "i'm down", "for sure", "bet", "count me in",
        ],
        "decline": [
            "no", "nah", "can't make it", "sorry can't", "busy", "not going",
            "pass", "maybe later", "i can't", "won't be able to",
        ],
        "question": [
            "what time", "when is it", "where at", "why", "how", "what's the plan",
            "which one", "who's coming", "what happened", "where", "when",
        ],
        "reaction": [
            "lol", "haha", "nice", "wow", "crazy", "damn", "omg", "that's funny",
            "hilarious", "no way", "wild", "bruh", "fr", "congrats", "ayy",
        ],
        "info": [
            "i'll be there at", "at 5pm", "tomorrow morning", "the address is",
            "it's at", "i'm at", "heading there now", "running late", "omw",
        ],
        "acknowledge": [
            "ok", "got it", "understood", "cool", "bet", "alright", "noted",
            "aight", "k", "np", "sounds good", "perfect", "works for me",
        ],
    }

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.anchor_embeddings = {}
        for intent, phrases in self.RESPONSE_ANCHORS.items():
            self.anchor_embeddings[intent] = self.model.encode(
                phrases, normalize_embeddings=True
            )

    def classify(self, message: str) -> tuple[str, float]:
        """Classify the intent of a response."""
        if not message.strip():
            return "unknown", 0.0

        msg_emb = self.model.encode([message], normalize_embeddings=True)[0]

        best_intent, best_score = "unknown", -1
        for intent, anchor_embs in self.anchor_embeddings.items():
            max_sim = float(np.max(np.dot(anchor_embs, msg_emb)))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        return best_intent, best_score


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # Remove common prefixes
    for prefix in ["me:", "Me:", "REPLY:", "Reply:", "Option 1:", "Option 2:", "Option 3:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Take first line only
    text = text.split("\n")[0].strip()
    # Remove quotes
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1]
    return text.strip()


def parse_multi_suggest(text: str) -> list[str]:
    """Parse multiple suggestions from model output."""
    suggestions = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        # Look for "Option N:" or numbered lines
        for prefix in ["Option 1:", "Option 2:", "Option 3:", "1.", "2.", "3.", "1)", "2)", "3)"]:
            if line.lower().startswith(prefix.lower()):
                suggestion = line[len(prefix):].strip()
                if suggestion:
                    suggestions.append(clean_response(suggestion))
                break
        else:
            # If line doesn't start with option marker but isn't empty
            if line and not any(line.startswith(p) for p in ["Option", "Generate", "Each"]):
                suggestions.append(clean_response(line))

    return suggestions[:3]  # Max 3


def load_test_set(limit: int) -> list[dict]:
    samples = []
    with open(CLEAN_TEST_SET) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


@dataclass
class ExpResult:
    sample_id: int
    conversation: str
    last_message: str
    gold: str
    gold_intent: str
    generated: str
    gen_intent: str
    intent_match: bool
    # For multi-suggest
    all_suggestions: list[str] = None
    any_match: bool = False


def run_experiment(
    model_name: str,
    n_samples: int,
    prompt_type: str,
    multi_suggest: bool,
    verbose: bool,
):
    from core.models.loader import ModelLoader

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Select prompt
    prompts = {
        "full_context": FULL_CONTEXT_PROMPT,
        "fewshot": FEWSHOT_CONTEXT_PROMPT,
        "roleplay": ROLEPLAY_CONTEXT_PROMPT,
    }
    base_prompt = prompts.get(prompt_type, FULL_CONTEXT_PROMPT)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 3: Full Context Generation")
    print(f"  Model: {model_name}")
    print(f"  Samples: {n_samples}")
    print(f"  Prompt: {prompt_type}")
    print(f"  Multi-suggest: {multi_suggest}")
    print(f"{'='*70}")

    # Load components
    print("\nLoading test set...")
    samples = load_test_set(n_samples)

    print("Loading response classifier...")
    classifier = ResponseClassifier()

    print(f"Loading LLM ({model_name})...")
    loader = ModelLoader(model_name)
    loader.preload()

    # Run experiment
    results = []
    stop_seqs = ["\nthem:", "\nme:", "<|im_end|>", "<|eot_id|>"]

    print(f"\nRunning generation with full context...")
    start_time = time.time()

    intent_matches = 0
    any_matches = 0  # For multi-suggest: at least one matches

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            pct = intent_matches / i * 100 if i > 0 else 0
            extra = f" | any_match: {any_matches}/{i}" if multi_suggest else ""
            print(f"  [{i+1}/{n_samples}] Intent match: {intent_matches}/{i} ({pct:.0f}%){extra}")

        conv = sample.get("conversation", "")
        last_msg = sample.get("last_message", "")
        gold = sample["gold_response"]
        gold_intent, _ = classifier.classify(gold)

        if multi_suggest:
            # Generate multiple suggestions
            prompt = MULTI_SUGGEST_PROMPT.format(conversation=conv)
            gen_result = loader.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,  # Higher for diversity
                stop=["<|im_end|>", "<|eot_id|>"],
            )
            suggestions = parse_multi_suggest(gen_result.text)

            # Check if any suggestion matches gold intent
            suggestion_intents = [classifier.classify(s)[0] for s in suggestions]
            any_match = gold_intent in suggestion_intents
            if any_match:
                any_matches += 1

            # Use first suggestion as "generated"
            generated = suggestions[0] if suggestions else ""
            gen_intent = suggestion_intents[0] if suggestion_intents else "unknown"

        else:
            # Single response
            prompt = base_prompt.format(conversation=conv)
            gen_result = loader.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.3,
                stop=stop_seqs,
            )
            generated = clean_response(gen_result.text)
            gen_intent, _ = classifier.classify(generated)
            suggestions = None
            any_match = False

        # Check intent match
        match = (gold_intent == gen_intent)
        if match:
            intent_matches += 1

        result = ExpResult(
            sample_id=i,
            conversation=conv[-200:],  # Truncate for storage
            last_message=last_msg,
            gold=gold,
            gold_intent=gold_intent,
            generated=generated,
            gen_intent=gen_intent,
            intent_match=match,
            all_suggestions=suggestions,
            any_match=any_match,
        )
        results.append(result)

        if verbose:
            marker = "✓" if match else "✗"
            print(f"    {marker} [{gold_intent}→{gen_intent}] \"{gold[:25]}\" → \"{generated[:25]}\"")
            if multi_suggest and suggestions:
                print(f"       All: {suggestions}")

    elapsed = time.time() - start_time

    # Print results
    n = len(results)
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

    print(f"\n--- METRICS ---")
    print(f"  Intent Match (first/only): {intent_matches}/{n} ({intent_matches/n*100:.1f}%)")
    if multi_suggest:
        print(f"  Any Match (multi-suggest): {any_matches}/{n} ({any_matches/n*100:.1f}%)")

    # By gold intent
    print(f"\n--- BY GOLD INTENT ---")
    for intent in ["accept", "decline", "question", "reaction", "info", "acknowledge"]:
        intent_results = [r for r in results if r.gold_intent == intent]
        if intent_results:
            matches = sum(1 for r in intent_results if r.intent_match)
            print(f"  {intent:12}: {matches}/{len(intent_results)} ({matches/len(intent_results)*100:3.0f}%)")

    # Sample outputs
    print(f"\n--- SAMPLE MATCHES ---")
    for r in [r for r in results if r.intent_match][:5]:
        print(f"  [{r.gold_intent}] \"{r.gold[:30]}\" → \"{r.generated[:30]}\"")

    print(f"\n--- SAMPLE MISMATCHES ---")
    for r in [r for r in results if not r.intent_match][:5]:
        print(f"  [{r.gold_intent}→{r.gen_intent}] \"{r.gold[:25]}\" → \"{r.generated[:25]}\"")
        if verbose:
            print(f"     Context: ...{r.conversation[-100:]}")

    if multi_suggest:
        print(f"\n--- MULTI-SUGGEST EXAMPLES ---")
        for r in results[:5]:
            match_marker = "✓" if r.any_match else "✗"
            print(f"  {match_marker} Gold [{r.gold_intent}]: \"{r.gold[:30]}\"")
            if r.all_suggestions:
                for s in r.all_suggestions:
                    s_intent, _ = classifier.classify(s)
                    m = "✓" if s_intent == r.gold_intent else " "
                    print(f"    {m} [{s_intent}] \"{s[:40]}\"")

    # Save
    suffix = "_multi" if multi_suggest else ""
    save_path = RESULTS_DIR / f"exp3_{prompt_type}{suffix}_{model_name}_{n_samples}.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": model_name,
            "prompt_type": prompt_type,
            "multi_suggest": multi_suggest,
            "n_samples": n,
            "elapsed": elapsed,
            "intent_match_rate": intent_matches / n,
            "any_match_rate": any_matches / n if multi_suggest else None,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nSaved: {save_path}")

    return intent_matches / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", type=str, default="lfm2.5-1.2b")
    parser.add_argument("--prompt", type=str, default="roleplay",
                        choices=["full_context", "fewshot", "roleplay"])
    parser.add_argument("--multi-suggest", action="store_true",
                        help="Generate 3 suggestions like Smart Reply")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_experiment(
        args.model,
        args.samples,
        args.prompt,
        args.multi_suggest,
        args.verbose,
    )


if __name__ == "__main__":
    main()
