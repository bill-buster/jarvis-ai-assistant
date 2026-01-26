#!/usr/bin/env python3
"""Check decision gates based on benchmark results.

Gates:
- G1: Memory - Total model stack memory usage
- G2: HHEM - Mean hallucination score
- G3: Warm Latency - p95 warm start latency
- G4: Cold Latency - p95 cold start latency
"""

import json
import sys
from pathlib import Path


def check_gates(results_dir: Path) -> dict[str, tuple[str, str]]:
    """Check all gates and return status."""

    gates: dict[str, tuple[str, str]] = {}

    # G1: Memory
    memory_file = results_dir / "memory.json"
    if memory_file.exists():
        memory = json.loads(memory_file.read_text())
        # Check if benchmark was skipped
        if memory.get("skipped"):
            gates["G1"] = ("SKIP", memory.get("reason", "benchmark skipped"))
        elif "profiles" in memory:
            # Sum up the model stack (LLM + embeddings)
            total_mb = sum(p["rss_mb"] for p in memory["profiles"])
            if total_mb < 5500:
                gates["G1"] = ("PASS", f"total = {total_mb:.0f}MB")
            elif total_mb < 6500:
                gates["G1"] = ("CONDITIONAL", f"total = {total_mb:.0f}MB")
            else:
                gates["G1"] = ("FAIL", f"total = {total_mb:.0f}MB")
        else:
            gates["G1"] = ("SKIP", "invalid memory.json format")
    else:
        gates["G1"] = ("SKIP", "memory.json not found")

    # G2: HHEM
    hhem_file = results_dir / "hhem.json"
    if hhem_file.exists():
        hhem = json.loads(hhem_file.read_text())
        # Check if benchmark was skipped
        if hhem.get("skipped"):
            gates["G2"] = ("SKIP", hhem.get("reason", "benchmark skipped"))
        elif "mean_score" in hhem:
            mean_score = hhem["mean_score"]
            if mean_score >= 0.5:
                gates["G2"] = ("PASS", f"mean HHEM = {mean_score:.3f}")
            elif mean_score >= 0.4:
                gates["G2"] = ("CONDITIONAL", f"mean HHEM = {mean_score:.3f}")
            else:
                gates["G2"] = ("FAIL", f"mean HHEM = {mean_score:.3f}")
        else:
            gates["G2"] = ("SKIP", "invalid hhem.json format")
    else:
        gates["G2"] = ("SKIP", "hhem.json not found")

    # G3: Warm Latency
    latency_file = results_dir / "latency.json"
    if latency_file.exists():
        latency = json.loads(latency_file.read_text())
        # Check if benchmark was skipped
        if latency.get("skipped"):
            skip_reason = latency.get("reason", "benchmark skipped")
            gates["G3"] = ("SKIP", skip_reason)
            gates["G4"] = ("SKIP", skip_reason)
        elif "results" in latency:
            warm_results = [r for r in latency["results"] if r["scenario"] == "warm"]
            if warm_results:
                warm_p95 = warm_results[0]["p95_ms"]
                if warm_p95 < 3000:
                    gates["G3"] = ("PASS", f"warm p95 = {warm_p95:.0f}ms")
                elif warm_p95 < 5000:
                    gates["G3"] = ("CONDITIONAL", f"warm p95 = {warm_p95:.0f}ms")
                else:
                    gates["G3"] = ("FAIL", f"warm p95 = {warm_p95:.0f}ms")
            else:
                gates["G3"] = ("SKIP", "no warm scenario in results")

            # G4: Cold Latency
            cold_results = [r for r in latency["results"] if r["scenario"] == "cold"]
            if cold_results:
                cold_p95 = cold_results[0]["p95_ms"]
                if cold_p95 < 15000:
                    gates["G4"] = ("PASS", f"cold p95 = {cold_p95:.0f}ms")
                elif cold_p95 < 20000:
                    gates["G4"] = ("CONDITIONAL", f"cold p95 = {cold_p95:.0f}ms")
                else:
                    gates["G4"] = ("FAIL", f"cold p95 = {cold_p95:.0f}ms")
            else:
                gates["G4"] = ("SKIP", "no cold scenario in results")
        else:
            gates["G3"] = ("SKIP", "invalid latency.json format")
            gates["G4"] = ("SKIP", "invalid latency.json format")
    else:
        gates["G3"] = ("SKIP", "latency.json not found")
        gates["G4"] = ("SKIP", "latency.json not found")

    return gates


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/latest")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    gates = check_gates(results_dir)

    print("\n" + "=" * 50)
    print("GATE STATUS")
    print("=" * 50)

    for gate, (status, detail) in gates.items():
        emoji = {"PASS": "✅", "CONDITIONAL": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}[status]
        print(f"{emoji} {gate}: {status} ({detail})")

    fails = sum(1 for _, (s, _) in gates.items() if s == "FAIL")
    if fails >= 2:
        print("\n⛔ RECOMMENDATION: Consider project cancellation (2+ failures)")
        sys.exit(2)
    elif fails == 1:
        print("\n🛑 RECOMMENDATION: Stop and reassess (1 failure)")
        sys.exit(1)
    else:
        print("\n🚀 RECOMMENDATION: Proceed with development")
        sys.exit(0)
