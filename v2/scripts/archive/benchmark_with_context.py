#!/usr/bin/env python3
"""Enhanced benchmark with conversation context and style analysis.

Tests how well models respond when given richer context about:
- Relationship type (close friend, family, work, etc.)
- Message intent (question, emotional, logistics, etc.)
- User's texting style (emoji, abbreviations, length)
- Conversation mood and topic

Usage:
    python scripts/benchmark_with_context.py --models lfm2-2.6b-exp lfm2.5-1.2b qwen3-0.6b
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.registry import MODELS as MODEL_REGISTRY
from core.generation.context_analyzer import ContextAnalyzer, ConversationContext
from core.generation.style_analyzer import StyleAnalyzer, UserStyle


def get_real_samples_with_context(num_samples: int = 10, context_size: int = 25) -> list[dict]:
    """Pull real conversation samples with full context analysis.

    Returns samples enriched with:
    - Conversation context (relationship, intent, mood, topic)
    - User style analysis (from your messages in the conversation)
    """
    try:
        from core.imessage.reader import MessageReader
    except ImportError:
        print("Warning: MessageReader not available")
        return []

    try:
        reader = MessageReader()
        conversations = reader.get_conversations(limit=100)
    except Exception as e:
        print(f"Warning: Could not read iMessages ({e})")
        return []

    context_analyzer = ContextAnalyzer()
    style_analyzer = StyleAnalyzer()

    samples = []
    seen_patterns = set()
    skipped = 0

    for conv in conversations:
        if len(samples) >= num_samples:
            break

        try:
            messages = reader.get_messages(conv.chat_id, limit=context_size + 10)
            if not messages or len(messages) < 5:  # Need enough for analysis
                continue

            messages = list(reversed(messages))

            # Skip spam (simple check)
            contact_name = conv.display_name or ""
            participants = conv.participants or []
            participant_str = participants[0] if participants else ""

            # Skip short codes
            if participant_str.isdigit() and 5 <= len(participant_str) <= 6:
                skipped += 1
                continue

            # Check message content for spam
            all_text = " ".join((m.text or "").lower() for m in messages[-5:])
            spam_keywords = [
                "reward points", "expire", "your order", "tracking",
                "legal representation", "law firm", "utm_source",
            ]
            if sum(1 for kw in spam_keywords if kw in all_text) >= 2:
                skipped += 1
                continue

            # Last message should be from them
            last_msg = messages[-1]
            if last_msg.is_from_me:
                continue

            last_text = (last_msg.text or "").strip()
            if len(last_text) < 3 or len(last_text) > 200:
                continue

            # Skip reactions
            skip_patterns = ["loved", "liked", "emphasized", "laughed at"]
            if any(p in last_text.lower() for p in skip_patterns):
                continue

            # Avoid duplicates
            pattern = last_text[:20].lower()
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)

            # Convert to dict format for analyzers
            msg_dicts = [
                {
                    "text": (m.text or "").strip(),
                    "is_from_me": m.is_from_me,
                    "sender": "them" if not m.is_from_me else "me",
                }
                for m in messages[-context_size:]
                if (m.text or "").strip()
            ]

            # Analyze context
            context = context_analyzer.analyze(msg_dicts)

            # Analyze user's style (from their own messages)
            my_messages = [m for m in msg_dicts if m.get("is_from_me")]
            user_style = style_analyzer.analyze(my_messages)

            # Build style instructions
            style_instructions = style_analyzer.build_style_instructions(user_style)

            # Get contact name
            short_name = contact_name.split()[0][:10] if contact_name else participant_str[:10]

            sample = {
                "name": f"{short_name}_{len(samples)+1}",
                "contact": contact_name or participant_str,
                "messages": msg_dicts,
                "context": context,
                "user_style": user_style,
                "style_instructions": style_instructions,
            }

            if msg_dicts and not msg_dicts[-1]["is_from_me"]:
                samples.append(sample)
                print(f"  [{context.relationship.value}] {short_name}: \"{last_text[:35]}...\"")
                print(f"      Intent: {context.intent.value}, Mood: {context.mood}, Topic: {context.topic}")

        except Exception as e:
            continue

    if skipped > 0:
        print(f"  (Skipped {skipped} spam conversations)")

    return samples


def build_rich_prompt(sample: dict) -> str:
    """Build a prompt with rich context information.

    Includes:
    - Relationship context
    - Style instructions
    - Conversation history
    """
    context: ConversationContext = sample["context"]
    style: UserStyle = sample["user_style"]
    messages = sample["messages"]

    # Build style hint from analysis
    style_parts = []

    # Length
    if style.avg_word_count < 6:
        style_parts.append("very brief (under 6 words)")
    elif style.avg_word_count < 12:
        style_parts.append("brief")

    # Capitalization
    if style.capitalization == "lowercase":
        style_parts.append("lowercase")

    # Emoji
    if style.uses_emoji and style.emoji_frequency > 0.2:
        style_parts.append("can use emoji")

    # Abbreviations
    if style.uses_abbreviations:
        style_parts.append("casual abbreviations ok")

    # Relationship context
    rel = context.relationship.value.replace("_", " ")
    if rel != "unknown":
        style_parts.append(f"tone: {rel}")

    # Mood match
    if context.mood == "positive":
        style_parts.append("upbeat")
    elif context.mood == "negative":
        style_parts.append("supportive")

    style_hint = ", ".join(style_parts) if style_parts else "casual, brief"

    # Format conversation
    lines = []
    for msg in messages[-15:]:  # Last 15 messages
        text = msg.get("text", "")
        if not text:
            continue
        prefix = "me:" if msg.get("is_from_me") else "them:"
        lines.append(f"{prefix} {text}")

    conversation = "\n".join(lines)

    # Build prompt
    prompt = f"[{style_hint}]\n\n{conversation}\nme:"
    return prompt


def benchmark_with_context(model_name: str, samples: list[dict]) -> dict:
    """Benchmark a model with rich context."""
    from core.models.loader import ModelLoader, reset_model_loader
    from core.models.registry import get_model_spec

    reset_model_loader()
    gc.collect()

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    spec = get_model_spec(model_name)

    print(f"\n{'='*60}")
    print(f"Model: {spec.display_name} ({model_name})")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    load_start = time.time()

    try:
        loader = ModelLoader(model_name)
        loader.preload()
        load_time = time.time() - load_start
        print(f"Loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"Failed to load: {e}")
        return {"model": model_name, "error": str(e), "responses": []}

    results = []

    for i, sample in enumerate(samples):
        context: ConversationContext = sample["context"]
        print(f"\n[{i+1}/{len(samples)}] {sample['name']} ({context.relationship.value})")
        print(f"  Intent: {context.intent.value} | Mood: {context.mood}")

        prompt = build_rich_prompt(sample)

        # Show condensed prompt
        prompt_preview = prompt.replace("\n", " | ")[:80]
        print(f"  Prompt: {prompt_preview}...")

        try:
            stop_sequences = ["\n", "them:", "<|im_end|>", "<|eot_id|>", "<end_of_turn>"]

            result = loader.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.3,
                stop=stop_sequences,
                use_chat_template=False,
            )

            output = result.text.strip()
            print(f"  Output: {output}")
            print(f"  Time: {result.generation_time_ms:.0f}ms")

            results.append({
                "name": sample["name"],
                "contact": sample["contact"],
                "relationship": context.relationship.value,
                "intent": context.intent.value,
                "mood": context.mood,
                "topic": context.topic,
                "last_message": context.last_message[:50],
                "output": output,
                "total_ms": result.generation_time_ms,
                "prompt_tokens": result.prompt_tokens,
            })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"name": sample["name"], "error": str(e)})

    loader.unload()
    gc.collect()

    # Calculate averages
    valid = [r for r in results if "total_ms" in r]
    avg_time = sum(r["total_ms"] for r in valid) / len(valid) if valid else 0

    return {
        "model": model_name,
        "load_time_s": load_time,
        "avg_total_ms": avg_time,
        "responses": results,
    }


def print_comparison(all_results: list[dict], samples: list[dict]):
    """Print side-by-side comparison of responses."""
    print("\n" + "=" * 80)
    print("RESPONSE COMPARISON (with context)")
    print("=" * 80)

    for sample in samples[:8]:  # First 8
        context: ConversationContext = sample["context"]
        print(f"\n[{sample['name']}] {context.relationship.value} | {context.intent.value}")
        print(f"  them: \"{context.last_message[:60]}\"")

        for result in all_results:
            if "error" in result:
                continue
            resp = next((r for r in result["responses"] if r.get("name") == sample["name"]), None)
            if resp and "output" in resp:
                model = result["model"]
                output = resp["output"][:60]
                time_ms = resp.get("total_ms", 0)
                print(f"  {model:<15}: {output} ({time_ms:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark with conversation context")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lfm2-2.6b-exp", "lfm2.5-1.2b", "qwen3-0.6b"],
        help="Models to test",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/context_benchmark.json",
        help="Output file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("JARVIS Context-Aware Benchmark")
    print("=" * 80)
    print(f"Models: {args.models}")
    print()

    # Get samples with context
    print("Loading samples with context analysis...")
    samples = get_real_samples_with_context(args.num_samples)

    if not samples:
        print("No samples found!")
        return

    print(f"\nFound {len(samples)} samples")

    # Benchmark each model
    all_results = []
    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}")
            continue
        result = benchmark_with_context(model_name, samples)
        all_results.append(result)

    # Print comparison
    print_comparison(all_results, samples)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Load':<8} {'Avg Time':<10}")
    print("-" * 40)
    for r in sorted(all_results, key=lambda x: x.get("avg_total_ms", 9999)):
        if "error" not in r:
            print(f"{r['model']:<20} {r['load_time_s']:.1f}s     {r['avg_total_ms']:.0f}ms")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": [
                {
                    "name": s["name"],
                    "contact": s["contact"],
                    "relationship": s["context"].relationship.value,
                    "intent": s["context"].intent.value,
                    "mood": s["context"].mood,
                    "topic": s["context"].topic,
                    "style_instructions": s["style_instructions"],
                }
                for s in samples
            ],
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
