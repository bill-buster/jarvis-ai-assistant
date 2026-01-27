#!/usr/bin/env python3
"""Generate markdown report from realistic reply generation test results."""

import json
import sys
from pathlib import Path

def generate_report(results_file: Path) -> str:
    """Generate markdown report from test results."""

    with open(results_file) as f:
        data = json.load(f)

    template = data["template_matching"]
    llm = data["llm_generation"]
    quality = data["quality_metrics"]

    report = f"""# Realistic iMessage Reply Generation Test Results

**Date:** {data['timestamp']}
**Conversations Tested:** {data['config']['num_conversations']}
**Variations per Context:** {data['config']['num_variations']}

---

## Executive Summary

### Template Matching (Fast Path)
- **Hit Rate:** {template['hit_rate']:.1%}
- **Avg Latency:** {template['avg_latency_ms']:.1f}ms
- **Verdict:** {'✅ Good coverage' if template['hit_rate'] > 0.3 else '⚠️ Low coverage - needs more templates'}

### LLM Generation (Fallback)
- **Models Tested:** {data['config']['models_tested']}
- **Brevity Score:** {quality['brevity_score']:.1%} (responses < 100 chars)
- **Variety Score:** {quality['variety_score']:.2f} (unique responses per context)

---

## 1. Template Matching Performance

"""

    # Template stats
    report += f"""
**Total Tests:** {template['total_tests']}
**Hits:** {template['template_hits']} ({template['hit_rate']:.1%})
**Misses:** {template['template_misses']}
**Avg Latency:** {template['avg_latency_ms']:.1f}ms

### Sample Matches

"""

    for match in template['matches'][:5]:
        report += f"""
**Query:** "{match['query'][:80]}"
**Matched Pattern:** "{match['matched_pattern'][:80]}"
**Response:** "{match['template_response'][:80]}"
**Confidence:** {match['confidence']:.2f}
**Latency:** {match['latency_ms']:.1f}ms

"""

    # LLM comparison
    report += """
---

## 2. LLM Model Comparison

| Model | Avg Latency | Avg Tokens | Avg Length | Successes |
|-------|-------------|------------|------------|-----------|
"""

    for model_result in llm:
        if "error" in model_result:
            report += f"| {model_result.get('model', 'Unknown')} | ERROR | - | - | - |\n"
            continue

        stats = model_result["stats"]
        report += f"| {model_result['model']} | {stats['avg_latency_ms']:.0f}ms | {stats['avg_tokens']:.0f} | {stats['avg_length_chars']:.0f} chars | {stats['successes']}/{stats['total_tests']} |\n"

    # Sample generations
    report += """
---

## 3. Sample Generations (Autonomous - No Instructions)

"""

    for model_result in llm:
        if "error" in model_result:
            continue

        report += f"\n### {model_result['model']}\n\n"

        for gen in model_result['generations'][:3]:  # Show first 3
            report += f"**Context:** {gen['last_message'][:100]}...\n\n"

            for i, var in enumerate(gen['variations'], 1):
                report += f"{i}. \"{var['reply']}\" ({var['latency_ms']}ms, {var['tokens']} tokens)\n"

            report += "\n"

    # Recommendations
    report += """
---

## 4. Recommendations

### Template Matching
"""

    if template['hit_rate'] > 0.4:
        report += "- ✅ Template coverage is good (>40% hit rate)\n"
        report += "- Templates handle common queries effectively\n"
        report += f"- Average latency: {template['avg_latency_ms']:.1f}ms (very fast)\n"
    else:
        report += "- ⚠️ Template coverage is low (<40% hit rate)\n"
        report += "- Consider mining more templates from historical messages\n"
        report += "- Or adjust similarity threshold (currently 0.7)\n"

    report += "\n### LLM Generation\n"

    # Find fastest model
    fastest = min(
        (m for m in llm if "error" not in m),
        key=lambda x: x['stats']['avg_latency_ms'],
        default=None
    )

    if fastest:
        report += f"- ⚡ **Fastest Model:** {fastest['model']} ({fastest['stats']['avg_latency_ms']:.0f}ms avg)\n"

    # Find most concise
    most_concise = min(
        (m for m in llm if "error" not in m),
        key=lambda x: x['stats']['avg_length_chars'],
        default=None
    )

    if most_concise:
        report += f"- 📝 **Most Concise:** {most_concise['model']} ({most_concise['stats']['avg_length_chars']:.0f} chars avg)\n"

    report += f"\n### Quality Metrics\n"
    report += f"- **Brevity:** {quality['brevity_score']:.1%} of responses are brief (<100 chars)\n"
    report += f"- **Variety:** {quality['variety_score']:.2f} unique responses per context (higher = more creative)\n"

    report += """

### Hybrid Approach (Recommended)

```
User Query
    ↓
Template Match (threshold=0.7)
    ↓
Hit? → Return template (10ms)
    ↓
Miss? → LLM Generation (400-1000ms)
    ↓
Return response
```

**Expected Performance:**
"""

    if template['hit_rate'] > 0:
        template_time = template['hit_rate'] * template['avg_latency_ms']
        llm_time = (1 - template['hit_rate']) * (fastest['stats']['avg_latency_ms'] if fastest else 500)
        hybrid_avg = template_time + llm_time

        report += f"- Template hits: {template['hit_rate']:.1%} × {template['avg_latency_ms']:.0f}ms = {template_time:.0f}ms\n"
        report += f"- LLM fallback: {1-template['hit_rate']:.1%} × {fastest['stats']['avg_latency_ms'] if fastest else 500:.0f}ms = {llm_time:.0f}ms\n"
        report += f"- **Hybrid Average: {hybrid_avg:.0f}ms**\n"

    report += """
---

## 5. Next Steps

1. **Review sample generations** - Are they natural and appropriate?
2. **Pick winning model** based on speed/quality tradeoff
3. **Update registry** if switching from current baseline
4. **Implement hybrid pipeline** in production
5. **Monitor hit rate** in production and mine new templates as needed

"""

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_realistic_report.py <results.json>")
        print("\nOr use the latest results:")
        results_dir = Path(__file__).parent.parent / "results"
        latest = max(results_dir.glob("realistic_reply_test_*.json"), default=None)
        if latest:
            print(f"  python {sys.argv[0]} {latest}")
        sys.exit(1)

    results_file = Path(sys.argv[1])

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        sys.exit(1)

    report = generate_report(results_file)

    # Save report
    report_file = results_file.with_suffix('.md')
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Report generated: {report_file}")
    print()
    print(report)


if __name__ == "__main__":
    main()
