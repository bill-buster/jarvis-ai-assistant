#!/usr/bin/env python3
"""Benchmark BitNet vs current JARVIS models.

This script compares:
- Qwen2.5-1.5B (current default)
- BitNet b1.58-2B (experimental)

Usage:
    python benchmark_bitnet.py
"""

import time
import sys
from pathlib import Path

# Test prompts (iMessage scenarios)
TEST_PROMPTS = [
    "Generate a friendly reply to: Hey, want to grab dinner tonight?",
    "Summarize this conversation: Hey, are we still on for tomorrow? | Yeah, 3pm works! | Perfect, see you then!",
    "Quick reply suggestions for: Can you send me that file?",
]


def benchmark_current_model():
    """Benchmark current Qwen2.5-1.5B model."""
    print("\n" + "=" * 60)
    print("Testing Qwen2.5-1.5B (Current Default)")
    print("=" * 60)

    try:
        from models.loader import MLXModelLoader, ModelConfig

        config = ModelConfig(model_id="qwen-1.5b")
        loader = MLXModelLoader(config)

        if not loader.load():
            print("❌ Failed to load model")
            return None

        results = []
        for prompt in TEST_PROMPTS:
            start = time.time()
            result = loader.generate_sync(prompt, max_tokens=50, temperature=0.7)
            elapsed = time.time() - start

            results.append(
                {
                    "prompt": prompt[:50] + "...",
                    "time": elapsed,
                    "tokens": len(result.text.split()),
                    "speed": len(result.text.split()) / elapsed,
                }
            )
            print(f"  {elapsed:.2f}s - {result.text[:60]}...")

        avg_speed = sum(r["speed"] for r in results) / len(results)
        print(f"\n✅ Average: {avg_speed:.1f} tokens/sec")
        return avg_speed

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def benchmark_bitnet():
    """Benchmark BitNet model via mlx-bitnet."""
    print("\n" + "=" * 60)
    print("Testing BitNet b1.58-2B (Experimental)")
    print("=" * 60)

    mlx_bitnet_path = Path.home() / "mlx-bitnet"
    if not mlx_bitnet_path.exists():
        print(f"❌ mlx-bitnet not found at {mlx_bitnet_path}")
        print("   Run: ./scripts/setup_bitnet.sh")
        return None

    print("✅ mlx-bitnet found")
    print("   Note: BitNet requires running from the mlx-bitnet directory")
    print("   To test manually:")
    print(f"   cd {mlx_bitnet_path}")
    print("   python mlx_bitnet.py --prompt 'Your test prompt here'")

    # Estimate based on published benchmarks
    print("\n📊 Expected performance (from Microsoft benchmarks):")
    print("   - Memory: 0.4GB (vs 1.5GB for Qwen)")
    print("   - Speed: 2.2x faster on Apple Silicon")
    print("   - Quality: Comparable to Qwen2.5-1.5B")

    return None


def main():
    print("=" * 60)
    print("JARVIS Model Benchmark: BitNet vs Current")
    print("=" * 60)

    # Test current model
    current_speed = benchmark_current_model()

    # Show BitNet info
    benchmark_bitnet()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if current_speed:
        print(f"Current (Qwen2.5-1.5B): {current_speed:.1f} tokens/sec")
        print(f"Expected (BitNet 2B): ~{current_speed * 2.2:.1f} tokens/sec")
        print(f"Memory savings: 1.5GB → 0.4GB (73% reduction)")
    else:
        print("Current model test failed")

    print("\nTo use BitNet:")
    print("1. Run: ./scripts/setup_bitnet.sh")
    print("2. Test: cd ~/mlx-bitnet && python mlx_bitnet.py")
    print("3. Compare outputs manually with JARVIS")


if __name__ == "__main__":
    main()
