#!/usr/bin/env python3
"""
Human-in-the-Loop Template Validation

Interactive tool for manually reviewing and rating templates.
Addresses the circular validation problem - no longer using same model to judge itself.

Creates a sample of templates for human review, presents them in terminal,
collects ratings, and generates validation report.
"""

import json
import random
from pathlib import Path
from typing import Any


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}{text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}{text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}{text}{Colors.ENDC}")


def sample_templates(
    templates_file: Path, sample_size: int = 50, stratified: bool = True
) -> list[dict]:
    """Sample templates for human review.

    Args:
        templates_file: Path to templates JSON
        sample_size: Number to sample
        stratified: Whether to stratify by context

    Returns:
        List of sampled templates
    """
    with open(templates_file) as f:
        data = json.load(f)

    patterns = data.get("patterns", [])

    if not patterns:
        print_warning("No patterns found in file")
        return []

    if len(patterns) <= sample_size:
        return patterns

    if stratified:
        # Stratify by context_stratum
        from collections import defaultdict

        strata = defaultdict(list)

        for pattern in patterns:
            stratum = pattern.get("context_stratum", "unknown")
            strata[stratum].append(pattern)

        # Sample proportionally from each stratum
        samples_per_stratum = max(1, sample_size // len(strata))
        sampled = []

        for stratum, stratum_patterns in strata.items():
            n = min(samples_per_stratum, len(stratum_patterns))
            sampled.extend(random.sample(stratum_patterns, n))

        # Fill remaining slots from top patterns
        remaining = sample_size - len(sampled)
        if remaining > 0:
            remaining_patterns = [p for p in patterns if p not in sampled]
            sampled.extend(
                random.sample(remaining_patterns, min(remaining, len(remaining_patterns)))
            )

        return sampled[:sample_size]
    else:
        return random.sample(patterns, sample_size)


def review_template(pattern: dict, index: int, total: int) -> dict[str, Any]:
    """Present template to user for review.

    Args:
        pattern: Pattern dict
        index: Current index
        total: Total templates

    Returns:
        Review results
    """
    print_header(f"Template {index}/{total}")
    print()

    incoming = pattern.get("representative_incoming", "")
    response = pattern.get("representative_response", "")
    context = pattern.get("context_stratum", "unknown")
    frequency = pattern.get("frequency", 0)
    num_senders = pattern.get("num_senders", 0)

    print(f"  Context: {Colors.BOLD}{context}{Colors.ENDC}")
    print(f"  Frequency: {frequency} occurrences across {num_senders} senders")
    print()
    print(f"  {Colors.OKBLUE}Incoming:{Colors.ENDC}")
    print(f'    "{incoming}"')
    print()
    print(f"  {Colors.OKGREEN}Response:{Colors.ENDC}")
    print(f'    "{response}"')
    print()

    # Collect ratings
    print("Rate this template:")
    print()

    # Appropriateness
    print("  1. Is the response appropriate for the incoming message?")
    print("     (1=terrible, 5=perfect)")
    appropriateness = None
    while appropriateness is None:
        try:
            val = input("     Rating [1-5]: ").strip()
            if val == "q":
                return {"quit": True}
            if val == "s":
                return {"skip": True}
            appropriateness = int(val)
            if not 1 <= appropriateness <= 5:
                print("     Please enter 1-5")
                appropriateness = None
        except ValueError:
            print("     Please enter a number 1-5 (or 'q' to quit, 's' to skip)")

    # Naturalness
    print()
    print("  2. Does the response sound natural?")
    print("     (1=robotic/weird, 5=perfectly natural)")
    naturalness = None
    while naturalness is None:
        try:
            val = input("     Rating [1-5]: ").strip()
            if val == "q":
                return {"quit": True}
            if val == "s":
                return {"skip": True}
            naturalness = int(val)
            if not 1 <= naturalness <= 5:
                print("     Please enter 1-5")
                naturalness = None
        except ValueError:
            print("     Please enter a number 1-5 (or 'q' to quit, 's' to skip)")

    # Context match
    print()
    print(f"  3. Does the response match the context ({context})?")
    print("     (1=completely wrong, 5=perfect match)")
    context_match = None
    while context_match is None:
        try:
            val = input("     Rating [1-5]: ").strip()
            if val == "q":
                return {"quit": True}
            if val == "s":
                return {"skip": True}
            context_match = int(val)
            if not 1 <= context_match <= 5:
                print("     Please enter 1-5")
                context_match = None
        except ValueError:
            print("     Please enter a number 1-5 (or 'q' to quit, 's' to skip)")

    # Overall
    print()
    print("  4. Overall, would you use this template?")
    overall_accept = None
    while overall_accept is None:
        val = input("     (y/n): ").strip().lower()
        if val == "q":
            return {"quit": True}
        if val == "s":
            return {"skip": True}
        if val in ["y", "yes"]:
            overall_accept = True
        elif val in ["n", "no"]:
            overall_accept = False
        else:
            print("     Please enter y or n (or 'q' to quit, 's' to skip)")

    # Optional notes
    print()
    notes = input("  5. Any notes/concerns? (press Enter to skip): ").strip()

    return {
        "appropriateness": appropriateness,
        "naturalness": naturalness,
        "context_match": context_match,
        "overall_accept": overall_accept,
        "notes": notes if notes else None,
        "skip": False,
        "quit": False,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Human template validation")
    parser.add_argument("input", type=str, help="Input templates JSON file")
    parser.add_argument(
        "--sample-size", type=int, default=50, help="Number of templates to review (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for reviews (default: input_humanvalidated.json)",
    )
    parser.add_argument(
        "--no-stratify", action="store_true", help="Don't stratify sampling by context"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print_warning(f"Input file not found: {input_file}")
        return

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_humanvalidated.json"

    print_header("HUMAN TEMPLATE VALIDATION")
    print()
    print_info("This tool presents templates for manual review.")
    print_info("You'll rate each template on:")
    print_info("  - Appropriateness (is response suitable?)")
    print_info("  - Naturalness (does it sound human?)")
    print_info("  - Context match (does it fit the context?)")
    print_info("  - Overall acceptance (would you use it?)")
    print()
    print_info("Commands:")
    print_info("  - Enter ratings 1-5 as prompted")
    print_info("  - Type 's' to skip a template")
    print_info("  - Type 'q' to quit and save progress")
    print()
    input("Press Enter to start...")

    # Sample templates
    print_info(f"\nSampling {args.sample_size} templates...")
    templates = sample_templates(input_file, args.sample_size, stratified=not args.no_stratify)

    if not templates:
        print_warning("No templates to review")
        return

    print_success(f"✓ Sampled {len(templates)} templates\n")

    # Review loop
    reviews = []
    reviewed_count = 0

    for i, pattern in enumerate(templates, 1):
        result = review_template(pattern, i, len(templates))

        if result.get("quit"):
            print_info("\nQuitting and saving progress...")
            break

        if result.get("skip"):
            print_info("Skipped\n")
            continue

        # Add pattern info to review
        result["pattern"] = {
            "representative_incoming": pattern.get("representative_incoming", ""),
            "representative_response": pattern.get("representative_response", ""),
            "context_stratum": pattern.get("context_stratum", ""),
            "frequency": pattern.get("frequency", 0),
            "cluster_id": pattern.get("cluster_id", ""),
        }

        reviews.append(result)
        reviewed_count += 1

        print_success(f"✓ Reviewed {reviewed_count}/{len(templates)}\n")

    # Calculate statistics
    if reviews:
        import numpy as np

        appropriateness_scores = [r["appropriateness"] for r in reviews]
        naturalness_scores = [r["naturalness"] for r in reviews]
        context_match_scores = [r["context_match"] for r in reviews]
        acceptance_rate = sum(1 for r in reviews if r["overall_accept"]) / len(reviews)

        print_header("VALIDATION RESULTS")
        print()
        print(f"  Reviewed: {reviewed_count} templates")
        print(f"  Average appropriateness: {np.mean(appropriateness_scores):.2f}/5")
        print(f"  Average naturalness: {np.mean(naturalness_scores):.2f}/5")
        print(f"  Average context match: {np.mean(context_match_scores):.2f}/5")
        print(f"  Acceptance rate: {acceptance_rate:.1%}")
        print()

        # Filter templates based on reviews
        accepted_clusters = set(r["pattern"]["cluster_id"] for r in reviews if r["overall_accept"])

        print_info(f"✓ {len(accepted_clusters)} clusters accepted by human review")

        # Save results
        output_data = {
            "validation_method": "human_review",
            "reviewed_count": reviewed_count,
            "sample_size": len(templates),
            "acceptance_rate": acceptance_rate,
            "statistics": {
                "avg_appropriateness": float(np.mean(appropriateness_scores)),
                "avg_naturalness": float(np.mean(naturalness_scores)),
                "avg_context_match": float(np.mean(context_match_scores)),
            },
            "reviews": reviews,
            "accepted_clusters": list(accepted_clusters),
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print_success(f"\n✓ Saved validation results to: {output_file}")
    else:
        print_warning("\nNo templates reviewed")


if __name__ == "__main__":
    main()
