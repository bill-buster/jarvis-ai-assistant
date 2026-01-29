#!/usr/bin/env python3
"""Test model capabilities for complex instructions.

Before optimizing generation quality, we need to know which models can:
1. Follow structured output instructions (ASK: vs REPLY:)
2. Recognize when they need more information
3. Generate diverse options
4. Handle multi-step reasoning

This is the REAL bottleneck - not generation quality.

Usage:
    python scripts/test_model_capabilities.py                    # Test all models
    python scripts/test_model_capabilities.py --model llama-3.2-3b  # Single model
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/capability_tests")

# Models to test (2026 comprehensive list)
MODELS = [
    # Small (< 1.5B) - fastest
    "lfm2.5-1.2b",      # Latest Liquid AI (Jan 2026), optimized for agents
    "llama-3.2-1b",     # Meta's edge model
    "qwen2.5-1.5b",     # Good for casual text
    "gemma3-1b",        # Google's edge model
    # Medium (1.5-2B)
    "smollm2-1.7b",     # HuggingFace's best small
    "qwen3-1.7b",       # Newer Qwen, better reasoning
    # Larger (2-3B) - best quality
    "lfm2-2.6b",        # LFM2 base
    "lfm2-2.6b-exp",    # LFM2 RL-tuned (should be best)
    "llama-3.2-3b",     # Meta 3B
    "smollm3-3b",       # HF's newest, beats Llama-3.2-3B
    "ministral-3b",     # Mistral's edge model
]

# Number of iterations per test for statistical reliability
N_ITERATIONS = 5


@dataclass
class CapabilityResult:
    """Result for a single capability test."""
    model: str
    test_name: str
    iteration: int
    passed: bool
    score: float  # 0-1 score instead of just pass/fail
    output: str
    evaluation_reason: str


# =============================================================================
# SEMANTIC EVALUATION FUNCTIONS
# =============================================================================

def evaluate_asks_vs_replies(output: str, should_ask: bool) -> tuple[bool, float, str]:
    """Evaluate if model correctly asks vs replies.

    More robust than string matching - looks for semantic intent.
    """
    output_lower = output.lower().strip()

    # Indicators that model is asking/uncertain
    ask_indicators = [
        "ask:", "need_info:", "?",  # Explicit markers
        "do you want", "would you like", "should i",  # Questions to user
        "accept or decline", "yes or no",  # Offering choices
        "i need to know", "i'm not sure", "uncertain",  # Expressing uncertainty
        "what is your", "what's your", "what do you",  # Asking for info
    ]

    # Indicators that model is providing a direct reply
    reply_indicators = [
        "reply:", "response:",  # Explicit markers
        "yeah", "yea", "sure", "ok", "nah", "no",  # Direct answers
        "sounds good", "i'm down", "count me in",  # Acceptances
        "can't", "busy", "sorry",  # Declines
        "lol", "haha", "nice", "cool",  # Reactions
    ]

    has_ask = any(ind in output_lower for ind in ask_indicators)
    has_reply = any(ind in output_lower for ind in reply_indicators)

    if should_ask:
        if has_ask and not has_reply:
            return True, 1.0, "Correctly asks for info"
        elif has_ask and has_reply:
            return True, 0.7, "Asks but also includes reply"
        elif "?" in output_lower:
            return True, 0.6, "Contains question (implicit ask)"
        else:
            return False, 0.2, "Should ask but gave direct reply"
    else:  # should reply
        if has_reply and not has_ask:
            return True, 1.0, "Correctly gives direct reply"
        elif has_reply and has_ask:
            return True, 0.7, "Replies but also asks"
        elif not has_ask:
            return True, 0.6, "Gives response without explicit marker"
        else:
            return False, 0.2, "Should reply but asked question"


def evaluate_brevity(output: str, max_words: int = 5) -> tuple[bool, float, str]:
    """Evaluate if model stays brief."""
    # Clean up output
    clean = output.strip()
    if clean.lower().startswith(("reply:", "me:")):
        clean = clean.split(":", 1)[1].strip()

    words = len(clean.split())
    chars = len(clean)

    if words <= max_words and chars <= 40:
        return True, 1.0, f"Brief: {words} words, {chars} chars"
    elif words <= max_words + 2:
        return True, 0.7, f"Slightly long: {words} words"
    elif words <= max_words * 2:
        return False, 0.4, f"Too long: {words} words"
    else:
        return False, 0.1, f"Way too long: {words} words"


def evaluate_style_match(output: str, expected_style: str = "casual") -> tuple[bool, float, str]:
    """Evaluate if output matches expected style."""
    output_lower = output.lower()

    # Casual style indicators
    casual_good = ["yeah", "yea", "lol", "haha", "nah", "ok", "u ", "ur ", "rn", "gonna", "wanna"]
    casual_bad = ["I would", "I am", "Thank you", "Certainly", "Of course", "I apologize"]

    has_casual_good = sum(1 for c in casual_good if c in output_lower)
    has_casual_bad = sum(1 for c in casual_bad if c in output_lower)

    # Check for proper capitalization (casual = lowercase)
    starts_lowercase = output[0].islower() if output else True

    score = 0.5  # Base score

    if has_casual_good > 0:
        score += 0.2 * min(has_casual_good, 2)
    if has_casual_bad > 0:
        score -= 0.3 * has_casual_bad
    if starts_lowercase:
        score += 0.1

    score = max(0, min(1, score))
    passed = score >= 0.5

    return passed, score, f"Style score: {score:.2f}"


def evaluate_diverse_options(output: str, n_options: int = 3) -> tuple[bool, float, str]:
    """Evaluate if model generates diverse options."""
    lines = [l.strip() for l in output.split("\n") if l.strip()]

    # Try to extract options
    options = []
    for line in lines:
        # Remove common prefixes
        for prefix in ["option 1:", "option 2:", "option 3:", "1.", "2.", "3.", "1)", "2)", "3)", "-"]:
            if line.lower().startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and len(line) > 2:
            options.append(line.lower())

    if len(options) < n_options:
        return False, 0.2, f"Only {len(options)} options found"

    # Check diversity - options should be different
    unique_options = set(options[:n_options])
    diversity = len(unique_options) / n_options

    if diversity >= 0.9:
        return True, 1.0, f"{len(unique_options)} unique options"
    elif diversity >= 0.6:
        return True, 0.7, f"Some duplicate options"
    else:
        return False, 0.3, "Options too similar"


def evaluate_intent_classification(output: str, expected: str = "A") -> tuple[bool, float, str]:
    """Evaluate intent classification."""
    output_clean = output.strip().upper()

    if expected in output_clean[:5]:  # Check first few chars
        return True, 1.0, f"Correct: {expected}"
    elif expected in output_clean:
        return True, 0.8, f"Contains correct answer: {expected}"
    else:
        return False, 0.0, f"Wrong answer, expected {expected}"


# =============================================================================
# CAPABILITY TESTS (with semantic evaluation)
# =============================================================================

CAPABILITY_TESTS = {
    # Test 1: Can it output structured format (ASK vs REPLY)?
    "structured_format": {
        "prompt": """Help me reply to this text.

Output ONLY one of these (nothing else):
- ASK: <question> - if you need to know my intent first
- REPLY: <message> - if you can reply directly

Example:
them: want to come to my party?
ASK: Do you want to go?

them: wanna grab dinner tonight?
""",
        "evaluator": lambda output: evaluate_asks_vs_replies(output, should_ask=True),
        "description": "Uses structured ASK:/REPLY: format",
    },

    # Test 2: Does it know to ASK for invitations?
    "ask_on_invitation": {
        "prompt": """Help me reply to this text.

Rules:
- If they're inviting me somewhere, ASK me first whether I want to go
- Don't assume my answer - I need to decide

Example:
them: wanna get coffee tomorrow?
Output: ASK: Do you want to go get coffee with them?

Now your turn:
them: wanna play basketball tomorrow?

Output:""",
        "evaluator": lambda output: evaluate_asks_vs_replies(output, should_ask=True),
        "description": "Asks for intent on invitations",
    },

    # Test 3: Does it REPLY for simple reactions?
    "reply_on_reaction": {
        "prompt": """Help me reply to this text.

Rules:
- For casual reactions/banter, just give a direct reply
- No need to ask me anything - just respond naturally

Example:
them: haha nice one
Output: REPLY: lol thanks

Now your turn:
them: lol that's hilarious

Output:""",
        "evaluator": lambda output: evaluate_asks_vs_replies(output, should_ask=False),
        "description": "Replies directly to reactions",
    },

    # Test 4: Does it stay brief?
    "brevity": {
        "prompt": """[MAX 5 words. Lowercase only.]

them: u coming tonight?

me:""",
        "evaluator": lambda output: evaluate_brevity(output, max_words=5),
        "description": "Stays brief when instructed",
    },

    # Test 5: Few-shot style following
    "few_shot_style": {
        "prompt": """Reply like a friend texting. Rules: lowercase, 2-4 words max, no emoji.

them: wanna hang? → yeah down
them: nice job! → lol thanks
them: you coming? → yea omw
them: got plans? → nah not really

them: you free later?
→""",
        "evaluator": lambda output: evaluate_style_match(output, "casual"),
        "description": "Follows few-shot casual style",
    },

    # Test 6: Uncertainty detection
    "uncertainty_detection": {
        "prompt": """Help me reply. You don't know my schedule - ask me first.

Output format: ASK: <short question>

them: what time works for you?
ASK: What time should I say?

them: where should we meet?
ASK: Where do you want to meet?

them: when should I pick you up?
ASK:""",
        "evaluator": lambda output: evaluate_asks_vs_replies(output, should_ask=True),
        "description": "Recognizes when it needs info",
    },

    # Test 7: Generate diverse options
    "diverse_options": {
        "prompt": """Generate exactly 3 different reply options for this message.

Example:
them: want to grab lunch?
1) yeah I'm down
2) can't today, maybe tomorrow?
3) where were you thinking?

Now your turn:
them: hey are you free this weekend?

1)
2)
3)""",
        "evaluator": lambda output: evaluate_diverse_options(output, n_options=3),
        "description": "Generates 3 diverse options",
    },

    # Test 8: Intent classification
    "intent_classification": {
        "prompt": """What response type is needed?

them: can you help me move saturday?

A) ACCEPT/DECLINE decision
B) PROVIDE INFORMATION
C) SIMPLE REACTION

Answer (A/B/C):""",
        "evaluator": lambda output: evaluate_intent_classification(output, expected="A"),
        "description": "Classifies response intent",
    },
}


def run_capability_test(
    model_name: str,
    test_name: str,
    test_config: dict,
    loader,
    iteration: int = 0,
) -> CapabilityResult:
    """Run a single capability test with semantic evaluation."""
    prompt = test_config["prompt"]

    # Generate with slight temperature variation for diversity
    temp = 0.2 + (iteration * 0.05)  # 0.2, 0.25, 0.3, etc.

    result = loader.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=min(temp, 0.5),
        stop=["\n\n", "<|im_end|>", "<|eot_id|>"],
    )
    output = result.text.strip()

    # Use semantic evaluator
    evaluator = test_config["evaluator"]
    passed, score, reason = evaluator(output)

    return CapabilityResult(
        model=model_name,
        test_name=test_name,
        iteration=iteration,
        passed=passed,
        score=score,
        output=output[:200],
        evaluation_reason=reason,
    )


def run_test_iterations(
    model_name: str,
    test_name: str,
    test_config: dict,
    loader,
    n_iterations: int = 10,
) -> list[CapabilityResult]:
    """Run multiple iterations of a test."""
    results = []
    for i in range(n_iterations):
        result = run_capability_test(model_name, test_name, test_config, loader, iteration=i)
        results.append(result)
    return results


def test_model(model_name: str, n_iterations: int = 10, verbose: bool = False) -> dict:
    """Test all capabilities for a model with multiple iterations.

    Returns dict with aggregated results per test.
    """
    from core.models.loader import ModelLoader

    print(f"\n{'='*70}")
    print(f"Testing: {model_name} ({n_iterations} iterations per test)")
    print(f"{'='*70}")

    loader = ModelLoader(model_name)
    loader.preload()

    all_results = []
    test_summaries = {}

    for test_name, test_config in CAPABILITY_TESTS.items():
        print(f"\n  Testing: {test_config['description']}...", end=" ", flush=True)

        # Run multiple iterations
        results = run_test_iterations(
            model_name, test_name, test_config, loader, n_iterations
        )
        all_results.extend(results)

        # Aggregate results
        pass_count = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / len(results)
        pass_rate = pass_count / len(results)

        test_summaries[test_name] = {
            "pass_count": pass_count,
            "total": len(results),
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "results": results,
        }

        # Status indicator
        if pass_rate >= 0.9:
            status = "✓✓"
            status_text = f"{pass_count}/{n_iterations}"
        elif pass_rate >= 0.7:
            status = "✓ "
            status_text = f"{pass_count}/{n_iterations}"
        elif pass_rate >= 0.5:
            status = "~ "
            status_text = f"{pass_count}/{n_iterations}"
        else:
            status = "✗ "
            status_text = f"{pass_count}/{n_iterations}"

        print(f"{status} {status_text} (avg score: {avg_score:.2f})")

        if verbose:
            for r in results[:3]:  # Show first 3 examples
                print(f"         [{r.iteration}] {r.evaluation_reason}: {r.output[:40]}...")

    # Overall summary
    total_tests = len(CAPABILITY_TESTS)
    reliable_tests = sum(1 for s in test_summaries.values() if s["pass_rate"] >= 0.8)
    avg_overall_score = sum(s["avg_score"] for s in test_summaries.values()) / total_tests

    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {reliable_tests}/{total_tests} tests reliable (≥80% pass rate)")
    print(f"  Average score: {avg_overall_score:.2f}")

    # Cleanup
    del loader
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except:
        pass

    return {
        "model": model_name,
        "n_iterations": n_iterations,
        "test_summaries": test_summaries,
        "reliable_tests": reliable_tests,
        "total_tests": total_tests,
        "avg_score": avg_overall_score,
        "all_results": all_results,
    }


def run_all_tests(models: list[str], n_iterations: int = 10, verbose: bool = False):
    """Test all models and create leaderboard."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_results = {}

    for model in models:
        try:
            result = test_model(model, n_iterations=n_iterations, verbose=verbose)
            model_results[model] = result
        except Exception as e:
            print(f"\nERROR testing {model}: {e}")
            import traceback
            traceback.print_exc()
            model_results[model] = {
                "model": model,
                "error": str(e),
                "reliable_tests": 0,
                "avg_score": 0,
            }

    # Print leaderboard
    print("\n" + "=" * 80)
    print("CAPABILITY LEADERBOARD (sorted by average score)")
    print("=" * 80)

    # Sort by avg_score
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1].get("avg_score", 0),
        reverse=True
    )

    print(f"\n{'Rank':<5} {'Model':<20} {'Reliable':<12} {'Avg Score':<12} {'Status'}")
    print("-" * 65)

    for i, (model, data) in enumerate(sorted_models):
        if "error" in data:
            print(f"{i+1:<5} {model:<20} {'ERROR':<12} {'-':<12} ✗ Failed to load")
            continue

        reliable = data["reliable_tests"]
        total = data["total_tests"]
        avg_score = data["avg_score"]

        if reliable >= total - 1 and avg_score >= 0.7:
            status = "✓✓ Excellent"
        elif reliable >= total * 0.7 and avg_score >= 0.6:
            status = "✓  Good"
        elif reliable >= total * 0.5:
            status = "~  Limited"
        else:
            status = "✗  Not suitable"

        print(f"{i+1:<5} {model:<20} {reliable}/{total:<10} {avg_score:<12.3f} {status}")

    # Detailed breakdown by test
    print("\n" + "-" * 80)
    print("DETAILED BREAKDOWN BY TEST (pass rate across all iterations)")
    print("-" * 80)

    for test_name, test_config in CAPABILITY_TESTS.items():
        print(f"\n{test_config['description']}:")
        for model, data in sorted_models:
            if "test_summaries" not in data:
                continue
            summary = data["test_summaries"].get(test_name, {})
            pass_rate = summary.get("pass_rate", 0)
            avg_score = summary.get("avg_score", 0)

            if pass_rate >= 0.9:
                bar = "██████████"
            elif pass_rate >= 0.7:
                bar = "███████░░░"
            elif pass_rate >= 0.5:
                bar = "█████░░░░░"
            else:
                bar = "██░░░░░░░░"

            print(f"  {model:<20} [{bar}] {pass_rate*100:5.1f}% (score: {avg_score:.2f})")

    # Save results
    results_file = RESULTS_DIR / f"capability_results_{n_iterations}iter.json"

    # Convert results for JSON
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_iterations": n_iterations,
        "models": {},
    }

    for model, data in model_results.items():
        if "error" in data:
            save_data["models"][model] = {"error": data["error"]}
        else:
            save_data["models"][model] = {
                "reliable_tests": data["reliable_tests"],
                "total_tests": data["total_tests"],
                "avg_score": data["avg_score"],
                "test_summaries": {
                    name: {
                        "pass_rate": s["pass_rate"],
                        "avg_score": s["avg_score"],
                    }
                    for name, s in data["test_summaries"].items()
                },
            }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSaved: {results_file}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_model, best_data = sorted_models[0]
    if "error" not in best_data and best_data["avg_score"] >= 0.7:
        print(f"\n✓ BEST: {best_model}")
        print(f"  Score: {best_data['avg_score']:.3f}")
        print(f"  Reliable tests: {best_data['reliable_tests']}/{best_data['total_tests']}")
        print(f"\n  This model can handle complex instructions. Proceed with generation testing.")
    else:
        print(f"\n⚠ No model achieved excellent results.")
        print(f"  Best: {best_model} (score: {best_data.get('avg_score', 0):.3f})")
        print(f"\n  Consider: simplifying prompts or accepting limitations.")

    return model_results


def main():
    parser = argparse.ArgumentParser(description="Test model capabilities for complex instructions")
    parser.add_argument("--model", type=str, help="Test single model")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Iterations per test (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    run_all_tests(models, n_iterations=args.iterations, verbose=args.verbose)


if __name__ == "__main__":
    main()
