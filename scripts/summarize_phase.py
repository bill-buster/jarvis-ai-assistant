#!/usr/bin/env python3
"""Generate phase summaries for context management.

This script helps reduce context usage by creating concise summaries of completed work.
It can be used manually or integrated into git hooks.

Usage:
    python scripts/summarize_phase.py --since HEAD~5 --output docs/PHASE_1_2_SUMMARY.md
    python scripts/summarize_phase.py --commits "abc123..def456"
    python scripts/summarize_phase.py --last-n 10
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_git_command(cmd: list[str]) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return ""


def get_commit_range(args: argparse.Namespace) -> str:
    """Determine commit range based on arguments."""
    if args.commits:
        return args.commits
    elif args.last_n:
        return f"HEAD~{args.last_n}..HEAD"
    elif args.since:
        return f"{args.since}..HEAD"
    else:
        # Default: last 5 commits
        return "HEAD~5..HEAD"


def get_commits(commit_range: str) -> list[dict[str, str]]:
    """Get list of commits in range."""
    log_output = run_git_command(
        [
            "log",
            "--pretty=format:%H|%s|%an|%ad",
            "--date=short",
            commit_range,
        ]
    )

    if not log_output:
        return []

    commits = []
    for line in log_output.split("\n"):
        if not line:
            continue
        hash_val, subject, author, date = line.split("|", 3)
        commits.append(
            {
                "hash": hash_val[:7],
                "subject": subject,
                "author": author,
                "date": date,
            }
        )
    return commits


def get_changed_files(commit_range: str) -> dict[str, tuple[int, int]]:
    """Get files changed with line counts."""
    diff_output = run_git_command(
        [
            "diff",
            "--numstat",
            commit_range,
        ]
    )

    if not diff_output:
        return {}

    files = {}
    for line in diff_output.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            added = int(parts[0]) if parts[0] != "-" else 0
            removed = int(parts[1]) if parts[1] != "-" else 0
            filepath = parts[2]
            files[filepath] = (added, removed)
    return files


def generate_summary(
    commits: list[dict[str, str]],
    files: dict[str, tuple[int, int]],
    phase_name: str,
) -> str:
    """Generate markdown summary."""
    total_added = sum(added for added, _ in files.values())
    total_removed = sum(removed for _, removed in files.values())
    net_change = total_added - total_removed

    summary = f"""# {phase_name}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Commits**: {len(commits)}
**Files Changed**: {len(files)}
**Lines**: +{total_added} -{total_removed} (net: {net_change:+d})

---

## Commits

"""

    for commit in commits:
        summary += f"- `{commit['hash']}` {commit['subject']} ({commit['date']})\n"

    summary += "\n---\n\n## Files Modified\n\n"

    # Group files by directory
    by_dir: dict[str, list[tuple[str, int, int]]] = {}
    for filepath, (added, removed) in files.items():
        dir_name = str(Path(filepath).parent) if "/" in filepath else "."
        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append((Path(filepath).name, added, removed))

    for dir_name in sorted(by_dir.keys()):
        summary += f"\n### `{dir_name}/`\n\n"
        for filename, added, removed in sorted(by_dir[dir_name]):
            summary += f"- `{filename}`: +{added} -{removed}\n"

    summary += "\n---\n\n## Next Steps\n\n"
    summary += "<!-- Add manual notes about what's next -->\n\n"
    summary += "---\n\n"
    summary += "*This summary was auto-generated to help manage context in large conversations.*\n"

    return summary


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate phase summaries for context management")
    parser.add_argument(
        "--commits",
        help="Commit range (e.g., 'abc123..def456')",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        help="Summarize last N commits",
    )
    parser.add_argument(
        "--since",
        help="Summarize commits since reference (e.g., 'HEAD~5')",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--phase-name",
        default="Phase Summary",
        help="Name for this phase (default: 'Phase Summary')",
    )

    args = parser.parse_args()

    # Ensure we're in a git repository
    if not Path(".git").exists():
        print("Error: Not in a git repository", file=sys.stderr)
        return 1

    commit_range = get_commit_range(args)
    commits = get_commits(commit_range)
    files = get_changed_files(commit_range)

    if not commits:
        print("No commits found in range", file=sys.stderr)
        return 1

    summary = generate_summary(commits, files, args.phase_name)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary)
        print(f"Summary written to {output_path}")
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
