#!/usr/bin/env python3
"""Benchmark different MLX models for text reply generation.

Tests speed and quality of various small LLMs for JARVIS text replies.
Uses simple conversation-style prompts with last N messages.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --models llama-3.2-1b gemma3-4b
    python scripts/benchmark_models.py --quick  # Just 3 samples
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


# Fallback test samples (used if --use-real-messages is not specified)
FALLBACK_SAMPLES = [
    {
        "name": "greeting",
        "messages": [
            {"text": "hey!", "is_from_me": False},
        ],
    },
    {
        "name": "dinner_invite",
        "messages": [
            {"text": "wanna grab dinner tonight?", "is_from_me": False},
        ],
    },
    {
        "name": "running_late",
        "messages": [
            {"text": "hey im gonna be like 10 min late", "is_from_me": False},
        ],
    },
    {
        "name": "question",
        "messages": [
            {"text": "what time does the movie start?", "is_from_me": False},
        ],
    },
    {
        "name": "multi_turn",
        "messages": [
            {"text": "hey you free this weekend?", "is_from_me": False},
            {"text": "yeah whats up", "is_from_me": True},
            {"text": "thinking of checking out that new ramen place", "is_from_me": False},
            {"text": "oh nice ive been wanting to try that", "is_from_me": True},
            {"text": "saturday work?", "is_from_me": False},
        ],
    },
    {
        "name": "emotional",
        "messages": [
            {"text": "ugh today was so stressful", "is_from_me": False},
        ],
    },
    {
        "name": "good_news",
        "messages": [
            {"text": "omg i got the job!!", "is_from_me": False},
        ],
    },
    {
        "name": "follow_up",
        "messages": [
            {"text": "how was the concert?", "is_from_me": False},
            {"text": "it was amazing", "is_from_me": True},
            {"text": "did they play your favorite song?", "is_from_me": False},
        ],
    },
]


def is_spam_or_promotional(conv, messages: list) -> bool:
    """Check if a conversation looks like spam or promotional messages.

    Args:
        conv: Conversation object
        messages: List of messages in the conversation

    Returns:
        True if this looks like spam/promotional, False if real conversation
    """
    contact_name = conv.display_name or ""
    participants = conv.participants or []
    participant_str = participants[0] if participants else ""

    # Check for short codes (5-6 digit numbers) - usually promotional
    if participant_str.isdigit() and 5 <= len(participant_str) <= 6:
        return True

    # Check for phone numbers without names (likely unknown/spam)
    if participant_str.startswith("+") and not contact_name:
        # Allow if there's back-and-forth conversation (real contact)
        my_messages = sum(1 for m in messages if m.is_from_me)
        their_messages = sum(1 for m in messages if not m.is_from_me)
        # If mostly one-sided from them, likely spam
        if my_messages < 2 and their_messages > 3:
            return True

    # Check message content for spam indicators
    all_text = " ".join((m.text or "").lower() for m in messages[-5:])

    spam_keywords = [
        # Promotional
        "reward points", "expire", "click here", "sign up", "subscribe",
        "unsubscribe", "opt out", "text stop", "reply stop",
        # Automated notifications
        "your order", "tracking", "delivery", "shipped",
        "appointment reminder", "confirmation code", "verification code",
        # Marketing
        "limited time", "special offer", "discount", "sale ends",
        "act now", "don't miss", "exclusive",
        # Legal spam
        "legal representation", "law firm", "free consultation",
        "motor vehicle accident", "personal injury",
        # URL spam patterns
        "utm_source", "utm_campaign", "click_id", ".cc/", "bit.ly",
    ]

    spam_count = sum(1 for kw in spam_keywords if kw in all_text)
    if spam_count >= 2:
        return True

    # Check for automated message patterns
    automated_patterns = [
        "thank you for your .* order",
        "this is .* from",
        "important notice",
        "requesting your signature",
    ]
    import re
    for pattern in automated_patterns:
        if re.search(pattern, all_text):
            return True

    return False


def get_real_samples(num_samples: int = 15, context_size: int = 25) -> list[dict]:
    """Pull real conversation samples from iMessage database.

    Finds conversations where they sent the last message (so you need to reply),
    and extracts the conversation context. Filters out spam and promotional messages.

    Args:
        num_samples: Number of samples to extract (default 15 for statistical significance)
        context_size: Number of messages to include as context (default 25 for rich context)

    Returns:
        List of sample dicts with 'name' and 'messages' keys
    """
    try:
        from core.imessage.reader import MessageReader
    except ImportError:
        print("Warning: MessageReader not available, using fallback samples")
        return FALLBACK_SAMPLES[:num_samples]

    try:
        reader = MessageReader()
        conversations = reader.get_conversations(limit=100)  # Check more convos to find good ones
    except Exception as e:
        print(f"Warning: Could not read iMessages ({e}), using fallback samples")
        return FALLBACK_SAMPLES[:num_samples]

    samples = []
    seen_patterns = set()  # Avoid duplicate message types
    skipped_spam = 0

    for conv in conversations:
        if len(samples) >= num_samples:
            break

        try:
            # Get recent messages from this conversation (extra buffer for filtering)
            messages = reader.get_messages(conv.chat_id, limit=context_size + 10)

            if not messages or len(messages) < 2:
                continue

            # Messages are returned newest first, reverse for chronological order
            messages = list(reversed(messages))

            # Skip spam and promotional messages
            if is_spam_or_promotional(conv, messages):
                skipped_spam += 1
                continue

            # Find a good sample: last message should be from them (needs reply)
            # and have some substance
            last_msg = messages[-1]
            if last_msg.is_from_me:
                continue

            last_text = (last_msg.text or "").strip()
            if len(last_text) < 3 or len(last_text) > 200:
                continue

            # Skip reactions and system messages
            skip_patterns = ["loved", "liked", "emphasized", "laughed at", "questioned"]
            if any(p in last_text.lower() for p in skip_patterns):
                continue

            # Create a pattern fingerprint to avoid similar samples
            pattern = last_text[:20].lower()
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)

            # Extract context (last N messages)
            context_msgs = messages[-context_size:]

            # Get contact name for sample name
            contact_name = conv.display_name or conv.participants[0] if conv.participants else "unknown"
            # Anonymize: just use first name or initial
            short_name = contact_name.split()[0][:10] if contact_name else "contact"

            sample = {
                "name": f"{short_name}_{len(samples)+1}",
                "chat_id": conv.chat_id,
                "contact": contact_name,
                "messages": [
                    {
                        "text": (m.text or "").strip(),
                        "is_from_me": m.is_from_me,
                    }
                    for m in context_msgs
                    if (m.text or "").strip()
                ],
            }

            if sample["messages"] and not sample["messages"][-1]["is_from_me"]:
                samples.append(sample)
                print(f"  Found sample from {short_name}: \"{last_text[:40]}...\"")

        except Exception as e:
            continue

    if skipped_spam > 0:
        print(f"  (Skipped {skipped_spam} spam/promotional conversations)")

    if not samples:
        print("Warning: No suitable real samples found, using fallback")
        return FALLBACK_SAMPLES[:num_samples]

    return samples


def format_conversation_prompt(messages: list[dict], style_hint: str = "") -> str:
    """Format conversation as simple text for completion.

    Just shows the conversation and lets the model continue naturally.
    """
    lines = []
    if style_hint:
        lines.append(f"[{style_hint}]")
        lines.append("")

    for msg in messages:
        prefix = "me:" if msg["is_from_me"] else "them:"
        lines.append(f"{prefix} {msg['text']}")

    # End with "me:" for completion
    if not messages[-1]["is_from_me"]:
        lines.append("me:")

    return "\n".join(lines)


@dataclass
class BenchmarkResult:
    model_name: str
    model_path: str
    load_time_s: float
    avg_prefill_ms: float
    avg_generation_ms: float
    avg_total_ms: float
    avg_prefill_tok_s: float
    avg_generation_tok_s: float
    responses: list[dict]
    memory_gb: float = 0.0


def get_memory_usage() -> float:
    """Get current memory usage in GB (MLX Metal memory)."""
    try:
        import mlx.core as mx
        # Use active memory, not total device memory
        return mx.metal.get_active_memory() / (1024 ** 3)
    except Exception:
        pass

    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        return 0.0


def benchmark_model(model_name: str, samples: list[dict], style_hint: str = "brief, casual") -> BenchmarkResult:
    """Benchmark a single model using our ModelLoader with detailed timing."""
    from core.models.loader import ModelLoader, reset_model_loader
    from core.models.registry import get_model_spec

    # Clean up any existing model before loading new one
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
    print(f"Path: {spec.path}")
    print(f"Prompt format: {spec.prompt_format}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    load_start = time.time()

    try:
        loader = ModelLoader(model_name)
        loader.preload()
        load_time = time.time() - load_start
        print(f"Loaded in {load_time:.1f}s")

        memory_after_load = get_memory_usage()
        print(f"Memory: {memory_after_load:.2f} GB")

    except Exception as e:
        print(f"Failed to load: {e}")
        return BenchmarkResult(
            model_name=model_name,
            model_path=spec.path,
            load_time_s=0,
            avg_prefill_ms=0,
            avg_generation_ms=0,
            avg_total_ms=0,
            avg_prefill_tok_s=0,
            avg_generation_tok_s=0,
            responses=[{"error": str(e)}],
        )

    # Run benchmarks
    results = []

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample['name']}...")

        # Build prompt using conversation format
        prompt = format_conversation_prompt(sample["messages"], style_hint)
        print(f"  Prompt: {prompt.replace(chr(10), ' | ')[:60]}...")

        try:
            # Stop sequences for different model formats
            stop_sequences = [
                "\n",           # Newline
                "them:",        # Next turn
                "Them:",
                "<|im_end|>",   # ChatML (Qwen, LFM)
                "<|eot_id|>",   # Llama 3
                "<end_of_turn>", # Gemma
                "</s>",         # Mistral
            ]

            result = loader.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.3,
                stop=stop_sequences,
                use_chat_template=False,  # Use raw completion for few-shot style
            )

            output = result.text.strip()

            # Calculate tok/s rates
            prefill_tok_s = result.prompt_tokens / (result.prefill_time_ms / 1000) if result.prefill_time_ms > 0 else 0
            gen_tok_s = result.tokens_generated / (result.generation_only_ms / 1000) if result.generation_only_ms > 0 else 0

            print(f"  Output: {output}")
            print(f"  Prefill: {result.prefill_time_ms:.0f}ms ({result.prompt_tokens} tok, {prefill_tok_s:.0f} tok/s)")
            print(f"  Generate: {result.generation_only_ms:.0f}ms ({result.tokens_generated} tok, {gen_tok_s:.1f} tok/s)")

            results.append({
                "test": sample["name"],
                "prompt": prompt,
                "output": output,
                "prefill_ms": result.prefill_time_ms,
                "generation_ms": result.generation_only_ms,
                "total_ms": result.generation_time_ms,
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.tokens_generated,
                "prefill_tok_s": prefill_tok_s,
                "generation_tok_s": gen_tok_s,
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test": sample["name"],
                "error": str(e),
            })

    # Cleanup
    loader.unload()
    gc.collect()

    # Calculate averages
    valid = [r for r in results if "prefill_ms" in r]
    if valid:
        avg_prefill = sum(r["prefill_ms"] for r in valid) / len(valid)
        avg_gen = sum(r["generation_ms"] for r in valid) / len(valid)
        avg_total = sum(r["total_ms"] for r in valid) / len(valid)
        avg_prefill_tok_s = sum(r["prefill_tok_s"] for r in valid) / len(valid)
        avg_gen_tok_s = sum(r["generation_tok_s"] for r in valid) / len(valid)
    else:
        avg_prefill = avg_gen = avg_total = avg_prefill_tok_s = avg_gen_tok_s = 0

    print(f"\n>>> {spec.display_name} Summary:")
    print(f"    Avg Prefill: {avg_prefill:.0f}ms ({avg_prefill_tok_s:.0f} tok/s)")
    print(f"    Avg Generate: {avg_gen:.0f}ms ({avg_gen_tok_s:.1f} tok/s)")
    print(f"    Avg Total: {avg_total:.0f}ms")

    return BenchmarkResult(
        model_name=model_name,
        model_path=spec.path,
        load_time_s=load_time,
        avg_prefill_ms=avg_prefill,
        avg_generation_ms=avg_gen,
        avg_total_ms=avg_total,
        avg_prefill_tok_s=avg_prefill_tok_s,
        avg_generation_tok_s=avg_gen_tok_s,
        responses=results,
        memory_gb=memory_after_load,
    )


def print_summary(results: list[BenchmarkResult], samples: list[dict]):
    """Print summary table and outputs for comparison."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Sort by total time (fastest first)
    sorted_results = sorted(results, key=lambda r: r.avg_total_ms)

    print(f"\n{'Model':<18} {'Load(s)':<8} {'Prefill':<10} {'Generate':<10} {'Total':<8} {'Gen tok/s':<10} {'Memory':<8}")
    print("-" * 82)

    for r in sorted_results:
        print(f"{r.model_name:<18} {r.load_time_s:>5.1f}s   {r.avg_prefill_ms:>6.0f}ms   {r.avg_generation_ms:>6.0f}ms   {r.avg_total_ms:>5.0f}ms   {r.avg_generation_tok_s:>8.1f}   {r.memory_gb:>5.2f}GB")

    print("-" * 82)

    # Print outputs for manual comparison
    print("\n" + "=" * 80)
    print("OUTPUTS FOR COMPARISON")
    print("=" * 80)

    for sample in samples[:5]:  # First 5 samples
        last_msg = sample["messages"][-1]["text"]
        print(f"\n[{sample['name']}] them: \"{last_msg}\"")
        for r in sorted_results:
            resp = next((x for x in r.responses if x.get("test") == sample["name"]), None)
            if resp and "output" in resp:
                print(f"  {r.model_name:<18}: {resp['output']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX models for text replies")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to test (default: representative set)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - only 3 samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/model_benchmark.json",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="brief, casual, lowercase",
        help="Style hint to include in prompt",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real messages from iMessage (instead of fake samples)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of conversation samples to test (default: 15)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=25,
        help="Number of messages to include as context per sample (default: 25)",
    )
    args = parser.parse_args()

    # Determine which models to test
    if args.models:
        model_ids = args.models
    else:
        # Default representative set
        model_ids = [
            "lfm2.5-1.2b",
            "llama-3.2-1b",
            "llama-3.2-3b",
            "gemma3-1b",
            "gemma3-4b",
            "ministral-3b",
            "qwen3-1.7b",
        ]

    # Filter to available models
    available = [m for m in model_ids if m in MODEL_REGISTRY]
    missing = [m for m in model_ids if m not in MODEL_REGISTRY]
    if missing:
        print(f"Warning: Unknown models skipped: {missing}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")

    # Determine samples
    num_samples = 3 if args.quick else args.num_samples
    context_size = 6 if args.quick else args.context_size
    if args.real:
        print(f"\nLoading {num_samples} real samples from iMessage (context: {context_size} messages each)...")
        samples = get_real_samples(num_samples, context_size)
    else:
        samples = FALLBACK_SAMPLES[:num_samples]

    print("=" * 80)
    print("JARVIS Model Benchmark - Text Reply Generation")
    print("=" * 80)
    print(f"Models: {available}")
    print(f"Samples: {len(samples)}")
    print(f"Context: {context_size} messages per sample")
    print(f"Style hint: {args.style}")
    print("=" * 80)

    results = []
    for model_name in available:
        try:
            result = benchmark_model(model_name, samples, args.style)
            results.append(result)
        except Exception as e:
            print(f"\nERROR: Failed to benchmark {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results, samples)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "style_hint": args.style,
                    "models": [
                        {
                            "model": r.model_name,
                            "path": r.model_path,
                            "load_time_s": r.load_time_s,
                            "avg_prefill_ms": r.avg_prefill_ms,
                            "avg_generation_ms": r.avg_generation_ms,
                            "avg_total_ms": r.avg_total_ms,
                            "avg_prefill_tok_s": r.avg_prefill_tok_s,
                            "avg_generation_tok_s": r.avg_generation_tok_s,
                            "memory_gb": r.memory_gb,
                            "responses": r.responses,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {output_path}")

        # Recommendations
        fastest = min(results, key=lambda r: r.avg_total_ms)
        best_gen = max(results, key=lambda r: r.avg_generation_tok_s)
        print(f"\nFastest overall: {fastest.model_name} ({fastest.avg_total_ms:.0f}ms avg)")
        print(f"Fastest generation: {best_gen.model_name} ({best_gen.avg_generation_tok_s:.1f} tok/s)")


if __name__ == "__main__":
    main()
