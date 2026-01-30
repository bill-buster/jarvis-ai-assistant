#!/usr/bin/env python3
"""
Template Quality Validation using LLM-as-Judge

Takes mined templates and scores them for:
1. Appropriateness (is response suitable for incoming message?)
2. Naturalness (does it sound human/natural?)
3. Safety (no inappropriate content)
4. Specificity (not too generic like "ok" for everything)

Only templates scoring >= 0.7 overall are kept.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.loader import MLXModelLoader, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def score_template_appropriateness(loader: MLXModelLoader, incoming: str, response: str) -> float:
    """Score template appropriateness using LLM-as-judge (0-1).

    Args:
        loader: Model loader
        incoming: Incoming message
        response: Template response

    Returns:
        Score from 0 (terrible) to 1 (perfect)
    """

    prompt = f"""Rate this iMessage reply on a scale of 0-10 for appropriateness.

Consider:
- Is the response relevant to the incoming message?
- Is it a reasonable thing to say in this context?
- Does the tone match?

Incoming: "{incoming}"
Response: "{response}"

Rating (just the number 0-10):"""

    try:
        loader.load()
        result = loader.generate_sync(prompt=prompt, max_tokens=5, temperature=0.3)

        # Extract number
        text = result.text.strip()
        # Try to parse number
        for char in text:
            if char.isdigit():
                score = int(char) / 10.0
                return score

        logger.warning("Failed to parse score from: %s", text)
        return 0.5  # Default to neutral

    except Exception as e:
        logger.error("Failed to score appropriateness: %s", e)
        return 0.5


def score_template_naturalness(loader: MLXModelLoader, response: str) -> float:
    """Score how natural the response sounds (0-1).

    Args:
        loader: Model loader
        response: Template response

    Returns:
        Score from 0 (robotic/weird) to 1 (natural)
    """

    prompt = f"""Rate this iMessage reply on naturalness (0-10).

Does it sound like something a real person would text?

Reply: "{response}"

Rating (just the number 0-10):"""

    try:
        loader.load()
        result = loader.generate_sync(prompt=prompt, max_tokens=5, temperature=0.3)

        text = result.text.strip()
        for char in text:
            if char.isdigit():
                score = int(char) / 10.0
                return score

        return 0.5

    except Exception as e:
        logger.error("Failed to score naturalness: %s", e)
        return 0.5


def check_template_safety(response: str) -> float:
    """Rule-based safety check (0-1).

    Flags:
    - Profanity (context-dependent, so lenient)
    - Offensive content
    - Personal info patterns

    Returns:
        1.0 if safe, lower if issues detected
    """

    response_lower = response.lower()

    # Very basic profanity check (most casual language is okay)
    severe_profanity = ["fuck you", "screw you", "hate you"]
    if any(phrase in response_lower for phrase in severe_profanity):
        return 0.0

    # Personal info patterns (phone, email, address)
    import re

    if re.search(r"\d{3}-\d{3}-\d{4}", response):  # Phone number
        return 0.3
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", response):  # Email
        return 0.3

    return 1.0


def check_template_specificity(incoming: str, response: str) -> float:
    """Check if response is too generic (0-1).

    Very generic responses like "ok" work for many contexts,
    but we want more specific templates too.

    Returns:
        1.0 if appropriately specific, lower if too generic
    """

    # Single-word generic responses
    ultra_generic = ["ok", "k", "yeah", "yes", "no", "sure"]
    if response.lower().strip() in ultra_generic:
        # These are okay, but score them lower
        return 0.6

    # Very short responses (under 5 chars) are often too generic
    if len(response.strip()) < 5:
        return 0.7

    return 1.0


def validate_template(
    loader: MLXModelLoader, incoming: str, response: str, use_llm: bool = True
) -> dict[str, Any]:
    """Validate a single template.

    Args:
        loader: Model loader for LLM-as-judge
        incoming: Incoming message
        response: Template response
        use_llm: Whether to use LLM scoring (slower but more accurate)

    Returns:
        Dict with scores and overall pass/fail
    """

    scores = {
        "appropriateness": 1.0,
        "naturalness": 1.0,
        "safety": 1.0,
        "specificity": 1.0,
    }

    # Always run rule-based checks
    scores["safety"] = check_template_safety(response)
    scores["specificity"] = check_template_specificity(incoming, response)

    # Optional LLM-based checks
    if use_llm:
        scores["appropriateness"] = score_template_appropriateness(loader, incoming, response)
        scores["naturalness"] = score_template_naturalness(loader, response)

    # Overall score (weighted average)
    overall = (
        scores["appropriateness"] * 0.4
        + scores["naturalness"] * 0.3
        + scores["safety"] * 0.2
        + scores["specificity"] * 0.1
    )

    return {"scores": scores, "overall": overall, "passed": overall >= 0.7}


def validate_templates(
    templates_file: Path, output_file: Path, use_llm: bool = True, sample_size: int | None = None
) -> dict[str, Any]:
    """Validate mined templates.

    Args:
        templates_file: Path to mined templates JSON
        output_file: Path to save validated templates
        use_llm: Whether to use LLM scoring
        sample_size: Only validate first N templates (for testing)

    Returns:
        Validation results summary
    """

    logger.info("Loading templates from: %s", templates_file)

    with open(templates_file) as f:
        data = json.load(f)

    patterns = data.get("patterns", [])

    if sample_size:
        patterns = patterns[:sample_size]
        logger.info("Validating sample of %d templates", sample_size)
    else:
        logger.info("Validating %d templates", len(patterns))

    # Initialize model if using LLM
    loader = None
    if use_llm:
        logger.info("Loading model for LLM-as-judge...")
        config = ModelConfig(model_id="qwen-1.5b")  # Use fast model
        loader = MLXModelLoader(config)

    # Validate each template
    validated_patterns = []
    passed_count = 0
    failed_count = 0

    for i, pattern in enumerate(patterns, 1):
        incoming = pattern.get("representative_incoming", "")
        response = pattern.get("representative_response", "")

        if not incoming or not response:
            continue

        if i % 10 == 0:
            logger.info("Progress: %d / %d templates validated", i, len(patterns))

        validation = validate_template(loader, incoming, response, use_llm)

        # Add validation results to pattern
        pattern["validation"] = validation

        if validation["passed"]:
            passed_count += 1
            validated_patterns.append(pattern)
        else:
            failed_count += 1
            logger.debug(
                "Template failed validation (score=%.2f): '%s' → '%s'",
                validation["overall"],
                incoming[:40],
                response[:40],
            )

    # Unload model
    if loader:
        loader.unload()

    # Sort by combined score (original metric) and validation score
    validated_patterns.sort(
        key=lambda x: x["combined_score"] * x["validation"]["overall"], reverse=True
    )

    # Save results
    results = {
        "total_templates": len(patterns),
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": passed_count / len(patterns) if patterns else 0,
        "patterns": validated_patterns,
        "validation_config": {
            "used_llm": use_llm,
            "threshold": 0.7,
            "weights": {
                "appropriateness": 0.4,
                "naturalness": 0.3,
                "safety": 0.2,
                "specificity": 0.1,
            },
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info("Total templates: %d", len(patterns))
    logger.info("Passed: %d (%.1f%%)", passed_count, results["pass_rate"] * 100)
    logger.info("Failed: %d", failed_count)
    logger.info("\nValidated templates saved to: %s", output_file)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate template quality")
    parser.add_argument("input", type=str, help="Input templates JSON file")
    parser.add_argument(
        "--output", type=str, default=None, help="Output file (default: input_validated.json)"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Use only rule-based validation (faster)"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Only validate first N templates (for testing)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        return

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_validated.json"

    validate_templates(input_file, output_file, use_llm=not args.no_llm, sample_size=args.sample)


if __name__ == "__main__":
    main()
