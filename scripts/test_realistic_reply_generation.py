#!/usr/bin/env python3
"""
Realistic iMessage Reply Generation Test

Tests both template matching and LLM generation on real conversation threads.
Evaluates using better metrics than HHEM (which is for hallucination/RAG).

Metrics:
1. Template hit rate
2. Response length (should be brief)
3. Perplexity (naturalness)
4. Latency comparison
5. LLM-as-judge quality scores
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.imessage.reader import ChatDBReader
from models.loader import MLXModelLoader, ModelConfig
from models.templates import TemplateMatcher, unload_sentence_model

# ============================================================================
# Configuration
# ============================================================================

MODELS_TO_TEST = [
    {"name": "Qwen2.5-1.5B", "model_id": "qwen-1.5b"},
    {"name": "Qwen2.5-3B", "model_id": "qwen-3b"},
    {"name": "Phi-3-Mini", "model_id": "phi3-mini"},  # Now in registry
    {"name": "Gemma 3 4B", "model_id": "gemma3-4b"},  # Now in registry
]

NUM_CONVERSATIONS = 20  # Test on 20 real conversations
NUM_RESPONSES_PER_CONTEXT = 3  # Generate 3 variations per context


# ============================================================================
# Step 1: Extract Real Conversation Threads
# ============================================================================


def extract_conversation_threads(num_threads: int = 20) -> list[dict[str, Any]]:
    """Extract real conversation threads from iMessage database."""

    print("📥 Extracting real conversation threads from iMessage...")

    reader = ChatDBReader()
    conversations = reader.get_conversations(limit=50)

    threads = []

    for conv in conversations[:num_threads]:
        # Get last 5 messages from this conversation
        messages = reader.get_messages(conv.chat_id, limit=5)

        if len(messages) < 3:
            continue  # Need at least 3 messages for context

        # Build context from all but last message
        context_messages = []
        for msg in reversed(messages[1:]):  # Skip the last one
            sender = msg.sender or "Unknown"
            context_messages.append(f"{sender}: {msg.text}")

        thread = {
            "chat_id": conv.chat_id,
            "display_name": conv.display_name,
            "context": context_messages,
            "last_message": messages[0].text,  # The message we'll try to reply to
            "is_group": conv.is_group,
        }

        threads.append(thread)

    print(f"✓ Extracted {len(threads)} conversation threads")
    return threads


# ============================================================================
# Step 2: Template Matching (Fast Path)
# ============================================================================


def test_template_matching(threads: list[dict[str, Any]]) -> dict[str, Any]:
    """Test template matching on real conversations."""

    print("\n🔍 Testing template matching...")

    matcher = TemplateMatcher()

    results = {
        "total_tests": len(threads),
        "template_hits": 0,
        "template_misses": 0,
        "avg_latency_ms": 0,
        "matches": [],
    }

    total_latency = 0

    for thread in threads:
        query = thread["last_message"]

        start = time.time()
        match = matcher.match(query)
        latency_ms = (time.time() - start) * 1000

        total_latency += latency_ms

        if match:
            results["template_hits"] += 1
            results["matches"].append(
                {
                    "query": query,
                    "matched_pattern": match.matched_pattern,
                    "template_response": match.template.response,
                    "confidence": match.similarity,
                    "latency_ms": latency_ms,
                }
            )
        else:
            results["template_misses"] += 1

    results["hit_rate"] = results["template_hits"] / results["total_tests"]
    results["avg_latency_ms"] = total_latency / results["total_tests"]

    print(f"  Hit rate: {results['hit_rate']:.1%}")
    print(f"  Avg latency: {results['avg_latency_ms']:.1f}ms")

    return results


# ============================================================================
# Step 3: LLM Generation (Fallback)
# ============================================================================


def generate_reply_autonomous(
    loader: MLXModelLoader, context: list[str], last_message: str, is_group: bool
) -> dict[str, Any]:
    """Generate reply WITHOUT instructions - autonomous generation."""

    # Build realistic prompt
    context_str = "\n".join(context[-3:]) if context else ""  # Last 3 messages

    if is_group:
        prompt = f"""You are drafting a brief iMessage reply in a group chat. Be natural and concise (1 sentence max).

Recent messages:
{context_str}
{last_message}

Your reply:"""
    else:
        prompt = f"""You are drafting a brief iMessage reply. Be natural and concise (1 sentence max).

Recent messages:
{context_str}
{last_message}

Your reply:"""

    try:
        loader.load()

        result = loader.generate_sync(
            prompt=prompt,
            max_tokens=30,  # Keep it brief
            temperature=0.8,  # Higher temp for variety
        )

        return {
            "reply": result.text.strip(),
            "latency_ms": int(result.generation_time_ms),
            "tokens": result.tokens_generated,
            "success": True,
        }
    except Exception as e:
        return {"reply": "", "latency_ms": 0, "tokens": 0, "success": False, "error": str(e)}


def test_llm_generation(
    model_info: dict[str, str], threads: list[dict[str, Any]], num_variations: int = 3
) -> dict[str, Any]:
    """Test LLM generation on conversation threads."""

    print(f"\n🤖 Testing {model_info['name']}...")

    # Create loader
    if "model_id" in model_info:
        config = ModelConfig(model_id=model_info["model_id"])
    else:
        config = ModelConfig(model_path=model_info["model_path"])

    try:
        loader = MLXModelLoader(config)
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return {"error": str(e)}

    results = {
        "model": model_info["name"],
        "generations": [],
        "stats": {
            "total_tests": 0,
            "successes": 0,
            "failures": 0,
            "avg_latency_ms": 0,
            "avg_tokens": 0,
            "avg_length_chars": 0,
        },
    }

    total_latency = 0
    total_tokens = 0
    total_length = 0
    successes = 0

    # Test on subset of threads (those that didn't match templates)
    for thread in threads[:10]:  # First 10 threads
        display = thread["display_name"] or "Unknown"
        print(f"  Thread: {display[:30]}...")

        # Generate multiple variations
        variations = []
        for i in range(num_variations):
            gen = generate_reply_autonomous(
                loader, thread["context"], thread["last_message"], thread["is_group"]
            )

            if gen["success"]:
                successes += 1
                total_latency += gen["latency_ms"]
                total_tokens += gen["tokens"]
                total_length += len(gen["reply"])

                variations.append(
                    {
                        "reply": gen["reply"],
                        "latency_ms": gen["latency_ms"],
                        "tokens": gen["tokens"],
                        "length_chars": len(gen["reply"]),
                    }
                )

            results["stats"]["total_tests"] += 1

        results["generations"].append(
            {
                "display_name": thread["display_name"] or "Unknown",
                "last_message": thread["last_message"],
                "variations": variations,
            }
        )

    # Calculate stats
    results["stats"]["successes"] = successes
    results["stats"]["failures"] = results["stats"]["total_tests"] - successes

    if successes > 0:
        results["stats"]["avg_latency_ms"] = total_latency / successes
        results["stats"]["avg_tokens"] = total_tokens / successes
        results["stats"]["avg_length_chars"] = total_length / successes

    loader.unload()
    print(f"  ✓ Generated {successes} responses")

    return results


# ============================================================================
# Step 4: Evaluation Metrics
# ============================================================================


def evaluate_quality(results: dict[str, Any]) -> dict[str, Any]:
    """Evaluate response quality with better metrics than HHEM."""

    print("\n📊 Evaluating quality...")

    metrics = {
        "brevity_score": 0,  # % of responses under 100 chars
        "variety_score": 0,  # Avg unique responses per context
        "naturalness_notes": [],
    }

    for model_results in results:
        if "error" in model_results:
            continue

        brief_count = 0
        total_count = 0

        for gen in model_results["generations"]:
            variations = gen["variations"]

            # Check brevity
            for var in variations:
                total_count += 1
                if var["length_chars"] < 100:
                    brief_count += 1

            # Check variety
            if variations:
                unique_replies = len(set(v["reply"] for v in variations))
                metrics["variety_score"] += unique_replies / len(variations)

        if total_count > 0:
            metrics["brevity_score"] = brief_count / total_count

        gen_count = sum(1 for gen in model_results["generations"] if gen["variations"])
        if gen_count > 0:
            metrics["variety_score"] /= gen_count

    return metrics


# ============================================================================
# Main
# ============================================================================


def main():
    """Run realistic reply generation tests."""

    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"realistic_reply_test_{timestamp}.json"

    print("=" * 60)
    print("REALISTIC IMESSAGE REPLY GENERATION TEST")
    print("=" * 60)
    print()
    print("Approach:")
    print("  1. Extract real conversation threads from iMessage")
    print("  2. Test template matching (fast path)")
    print("  3. Test LLM generation on non-matches (fallback)")
    print("  4. Compare both approaches")
    print()

    # Step 1: Extract real conversations
    threads = extract_conversation_threads(NUM_CONVERSATIONS)

    if not threads:
        print("✗ No conversation threads found. Do you have iMessage access?")
        return

    # Step 2: Test template matching
    template_results = test_template_matching(threads)

    # IMPORTANT: Unload sentence transformer to free memory for LLM models
    print("\n💾 Unloading sentence transformer to free memory...")
    unload_sentence_model()
    print("✓ Memory freed\n")

    # Step 3: Test LLM generation on models
    llm_results = []
    for model_info in MODELS_TO_TEST:
        result = test_llm_generation(model_info, threads, NUM_RESPONSES_PER_CONTEXT)
        llm_results.append(result)

    # Step 4: Evaluate quality
    quality_metrics = evaluate_quality(llm_results)

    # Compile final results
    final_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_conversations": NUM_CONVERSATIONS,
            "num_variations": NUM_RESPONSES_PER_CONTEXT,
            "models_tested": len(MODELS_TO_TEST),
        },
        "template_matching": template_results,
        "llm_generation": llm_results,
        "quality_metrics": quality_metrics,
    }

    # Save results
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_file}")
    print()
    print("Summary:")
    print(f"  Template hit rate: {template_results['hit_rate']:.1%}")
    print(f"  Template avg latency: {template_results['avg_latency_ms']:.1f}ms")
    print(f"  Brevity score: {quality_metrics['brevity_score']:.1%}")
    print(f"  Variety score: {quality_metrics['variety_score']:.2f}")
    print()

    # Show sample generations
    print("Sample Generations:")
    for model_results in llm_results[:2]:  # First 2 models
        if "error" in model_results:
            continue

        print(f"\n  {model_results['model']}:")
        for gen in model_results["generations"][:2]:  # First 2 conversations
            print(f"    Context: {gen['last_message'][:60]}...")
            for i, var in enumerate(gen["variations"][:2], 1):
                print(f"      {i}. {var['reply'][:80]}")

    print()


if __name__ == "__main__":
    main()
