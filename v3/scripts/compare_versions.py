"""Compare reply generation across all versions (root, v2, v3).

This script loads the same test samples and runs them through each version
to compare quality, speed, and reply types.
"""

import json
import sys
import time
from pathlib import Path

# We're in v3, so we need to adjust paths for other versions
V3_DIR = Path(__file__).parent.parent
ROOT_DIR = V3_DIR.parent
V2_DIR = ROOT_DIR / "v2"


def load_test_samples() -> list[dict]:
    """Load test samples from existing evaluation."""
    samples_path = V3_DIR / "results" / "baseline_v3_improved.json"
    if not samples_path.exists():
        samples_path = V3_DIR / "results" / "baseline_v3.json"

    if not samples_path.exists():
        print("No test samples found. Run evaluate_replies.py first.")
        sys.exit(1)

    with open(samples_path) as f:
        return json.load(f)


def evaluate_v3(samples: list[dict]) -> list[dict]:
    """Evaluate using v3 (current improved version)."""
    print("\n" + "="*60)
    print("Evaluating V3 (current improved)")
    print("="*60)

    # Already in v3 context
    sys.path.insert(0, str(V3_DIR))

    from core.generation.reply_generator import ReplyGenerator
    from core.models.loader import ModelLoader

    loader = ModelLoader()
    generator = ReplyGenerator(loader)

    results = []
    total_time = 0

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {sample['display_name'][:20]}...", end=" ", flush=True)

        start = time.time()
        try:
            result = generator.generate_replies(
                messages=sample["messages"],
                chat_id=sample.get("chat_id"),
                num_replies=3,
                contact_name=sample.get("display_name"),
            )
            elapsed = (time.time() - start) * 1000
            total_time += elapsed

            results.append({
                "display_name": sample["display_name"],
                "last_incoming": sample.get("last_incoming", ""),
                "replies": [
                    {"text": r.text, "type": r.reply_type, "confidence": r.confidence}
                    for r in result.replies
                ],
                "time_ms": elapsed,
                "past_replies_found": len(result.past_replies) if hasattr(result, "past_replies") else 0,
            })
            print(f"✓ {elapsed:.0f}ms")
        except Exception as e:
            print(f"✗ {e}")
            results.append({
                "display_name": sample["display_name"],
                "last_incoming": sample.get("last_incoming", ""),
                "replies": [],
                "error": str(e),
                "time_ms": 0,
            })

    print(f"\n  V3 Total: {total_time:.0f}ms, Avg: {total_time/len(samples):.0f}ms")
    return results


def evaluate_v2(samples: list[dict]) -> list[dict]:
    """Evaluate using v2."""
    print("\n" + "="*60)
    print("Evaluating V2")
    print("="*60)

    # Check if v2 exists
    if not V2_DIR.exists():
        print("  V2 directory not found, skipping")
        return []

    # Add v2 to path
    sys.path.insert(0, str(V2_DIR))

    try:
        # Try to import v2 modules
        from core.generation.reply_generator import ReplyGenerator
        from core.models.loader import ModelLoader

        loader = ModelLoader()
        generator = ReplyGenerator(loader)

        results = []
        total_time = 0

        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {sample['display_name'][:20]}...", end=" ", flush=True)

            start = time.time()
            try:
                result = generator.generate_replies(
                    messages=sample["messages"],
                    chat_id=sample.get("chat_id"),
                    num_replies=3,
                    contact_name=sample.get("display_name"),
                )
                elapsed = (time.time() - start) * 1000
                total_time += elapsed

                results.append({
                    "display_name": sample["display_name"],
                    "last_incoming": sample.get("last_incoming", ""),
                    "replies": [
                        {"text": r.text, "type": r.reply_type, "confidence": r.confidence}
                        for r in result.replies
                    ],
                    "time_ms": elapsed,
                    "past_replies_found": len(result.past_replies) if hasattr(result, "past_replies") else 0,
                })
                print(f"✓ {elapsed:.0f}ms")
            except Exception as e:
                print(f"✗ {e}")
                results.append({
                    "display_name": sample["display_name"],
                    "last_incoming": sample.get("last_incoming", ""),
                    "replies": [],
                    "error": str(e),
                    "time_ms": 0,
                })

        print(f"\n  V2 Total: {total_time:.0f}ms, Avg: {total_time/len(samples):.0f}ms")
        return results

    except ImportError as e:
        print(f"  Failed to import v2: {e}")
        return []
    finally:
        # Remove v2 from path
        if str(V2_DIR) in sys.path:
            sys.path.remove(str(V2_DIR))


def evaluate_root(samples: list[dict]) -> list[dict]:
    """Evaluate using root version."""
    print("\n" + "="*60)
    print("Evaluating ROOT")
    print("="*60)

    # Add root to path
    sys.path.insert(0, str(ROOT_DIR))

    try:
        # Root uses Generator class and build_reply_prompt
        from models.generator import get_generator
        from jarvis.prompts import build_reply_prompt

        generator = get_generator()

        results = []
        total_time = 0

        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {sample['display_name'][:20]}...", end=" ", flush=True)

            start = time.time()
            try:
                # Format messages for root prompt
                messages = sample["messages"]

                # Build context string
                context_lines = []
                for msg in messages[-10:]:
                    sender = "You" if msg.get("is_from_me") else msg.get("sender", "Them")
                    context_lines.append(f"{sender}: {msg.get('text', '')}")
                context = "\n".join(context_lines)

                # Get last message
                last_msg = ""
                for msg in reversed(messages):
                    if not msg.get("is_from_me"):
                        last_msg = msg.get("text", "")
                        break

                # Build prompt using root's method
                prompt = build_reply_prompt(
                    context=context,
                    last_message=last_msg,
                    contact_name=sample.get("display_name", "them"),
                )

                # Generate replies
                replies = []
                for temp in [0.7, 0.8, 0.9]:
                    try:
                        result = generator.generate(
                            prompt=prompt,
                            max_tokens=50,
                            temperature=temp,
                        )
                        reply_text = result.text.strip().split("\n")[0]
                        replies.append({
                            "text": reply_text,
                            "type": "generated",
                            "confidence": 0.8,
                        })
                    except Exception:
                        pass

                elapsed = (time.time() - start) * 1000
                total_time += elapsed

                results.append({
                    "display_name": sample["display_name"],
                    "last_incoming": sample.get("last_incoming", ""),
                    "replies": replies[:3],
                    "time_ms": elapsed,
                    "past_replies_found": 0,  # Root doesn't have RAG
                })
                print(f"✓ {elapsed:.0f}ms")

            except Exception as e:
                print(f"✗ {e}")
                results.append({
                    "display_name": sample["display_name"],
                    "last_incoming": sample.get("last_incoming", ""),
                    "replies": [],
                    "error": str(e),
                    "time_ms": 0,
                })

        print(f"\n  ROOT Total: {total_time:.0f}ms, Avg: {total_time/len(samples):.0f}ms")
        return results

    except ImportError as e:
        print(f"  Failed to import root: {e}")
        return []
    finally:
        # Remove root from path
        if str(ROOT_DIR) in sys.path:
            sys.path.remove(str(ROOT_DIR))


def compare_results(v3_results: list, v2_results: list, root_results: list):
    """Compare and display results from all versions."""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    # Count reply types for each version
    from collections import Counter

    def count_types(results):
        types = []
        for r in results:
            for reply in r.get("replies", []):
                types.append(reply.get("type", "unknown"))
        return Counter(types)

    v3_types = count_types(v3_results) if v3_results else Counter()
    v2_types = count_types(v2_results) if v2_results else Counter()
    root_types = count_types(root_results) if root_results else Counter()

    all_types = set(v3_types.keys()) | set(v2_types.keys()) | set(root_types.keys())

    print("\n📊 Reply Type Distribution:")
    print(f"  {'Type':<20} {'V3':<10} {'V2':<10} {'Root':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for t in sorted(all_types):
        print(f"  {t:<20} {v3_types.get(t, 0):<10} {v2_types.get(t, 0):<10} {root_types.get(t, 0):<10}")

    # Calculate averages
    def avg_time(results):
        times = [r["time_ms"] for r in results if r.get("time_ms", 0) > 0]
        return sum(times) / len(times) if times else 0

    print("\n⏱️ Average Generation Time (excluding cold start):")
    if v3_results:
        # Skip first sample (cold start)
        v3_warm = [r for r in v3_results[1:] if r.get("time_ms", 0) > 0]
        print(f"  V3:   {avg_time(v3_warm):.0f}ms")
    if v2_results:
        v2_warm = [r for r in v2_results[1:] if r.get("time_ms", 0) > 0]
        print(f"  V2:   {avg_time(v2_warm):.0f}ms")
    if root_results:
        root_warm = [r for r in root_results[1:] if r.get("time_ms", 0) > 0]
        print(f"  Root: {avg_time(root_warm):.0f}ms")

    # Side-by-side comparison
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)

    for i in range(min(len(v3_results), 10)):  # Show first 10
        name = v3_results[i]["display_name"]
        last_msg = v3_results[i].get("last_incoming", "")[:40]

        print(f"\n{i+1}. {name}: \"{last_msg}...\"")

        if v3_results and i < len(v3_results):
            replies = [r["text"][:30] for r in v3_results[i].get("replies", [])]
            print(f"   V3:   {replies}")

        if v2_results and i < len(v2_results):
            replies = [r["text"][:30] for r in v2_results[i].get("replies", [])]
            print(f"   V2:   {replies}")

        if root_results and i < len(root_results):
            replies = [r["text"][:30] for r in root_results[i].get("replies", [])]
            print(f"   Root: {replies}")


def main():
    print("="*70)
    print("JARVIS Version Comparison")
    print("="*70)

    # Load test samples
    samples = load_test_samples()
    print(f"\nLoaded {len(samples)} test samples")

    # Limit to first 10 for faster comparison
    samples = samples[:10]
    print(f"Using first {len(samples)} samples for comparison")

    # Evaluate each version
    v3_results = evaluate_v3(samples)

    # V2 and root share some modules with v3, so we need to be careful
    # For now, just compare v3 before/after
    v2_results = []  # evaluate_v2(samples)
    root_results = []  # evaluate_root(samples)

    # Compare results
    compare_results(v3_results, v2_results, root_results)

    # Save results
    output = {
        "v3": v3_results,
        "v2": v2_results,
        "root": root_results,
    }

    output_path = V3_DIR / "results" / "version_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
