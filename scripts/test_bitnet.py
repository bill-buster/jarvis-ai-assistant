"""Test script for BitNet model comparison."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import get_model_spec


def compare_models():
    """Compare BitNet vs LFM2.5 vs current models."""

    print("=" * 60)
    print("Model Comparison for JARVIS")
    print("=" * 60)

    # Current models
    models_to_compare = [
        "qwen-1.5b",  # Current default
        "lfm2.5-1.2b",  # LFM2.5 (if available)
        "bitnet-2b",  # BitNet (new)
    ]

    print("\nModel Specifications:")
    print("-" * 60)
    print(f"{'Model':<20} {'Size':<10} {'Memory':<12} {'Tier':<10}")
    print("-" * 60)

    for model_id in models_to_compare:
        spec = get_model_spec(model_id)
        if spec:
            print(
                f"{spec.id:<20} {spec.display_name:<10} {spec.size_gb:<12.1f} {spec.quality_tier:<10}"
            )
        else:
            print(f"{model_id:<20} {'NOT FOUND':<10}")

    print("-" * 60)
    print("\nTo test BitNet:")
    print("1. Download: huggingface-cli download microsoft/bitnet-b1.58-2B-4T")
    print("2. Run: jarvis chat --model bitnet-2b")
    print("\nNote: BitNet requires special handling (not standard MLX)")
    print("Consider using: https://github.com/exo-explore/mlx-bitnet")


if __name__ == "__main__":
    compare_models()
