"""Memory profiling dashboard for visualization.

Provides utilities for:
- Collecting and displaying memory usage over time
- Generating text-based visualizations for terminal
- Exporting memory data for external analysis
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from jarvis.metrics import MemorySampler, get_memory_sampler

# Constants
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3


@dataclass
class MemorySnapshot:
    """A comprehensive memory snapshot at a point in time."""

    timestamp: str
    process_rss_mb: float
    process_vms_mb: float
    system_total_gb: float
    system_available_gb: float
    system_used_gb: float
    system_percent: float
    metal_gpu_mb: float
    gc_objects: int


def _get_metal_memory_mb() -> float:
    """Get Metal GPU memory if available."""
    try:
        import mlx.core as mx

        return mx.metal.get_peak_memory() / BYTES_PER_MB
    except (ImportError, AttributeError):
        return 0.0


def _get_gc_objects() -> int:
    """Get count of tracked garbage collector objects."""
    import gc

    return len(gc.get_objects())


def take_snapshot() -> MemorySnapshot:
    """Take a comprehensive memory snapshot.

    Returns:
        MemorySnapshot with current memory state
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()

    return MemorySnapshot(
        timestamp=datetime.now(UTC).isoformat(),
        process_rss_mb=round(mem_info.rss / BYTES_PER_MB, 2),
        process_vms_mb=round(mem_info.vms / BYTES_PER_MB, 2),
        system_total_gb=round(system_mem.total / BYTES_PER_GB, 2),
        system_available_gb=round(system_mem.available / BYTES_PER_GB, 2),
        system_used_gb=round(system_mem.used / BYTES_PER_GB, 2),
        system_percent=round(system_mem.percent, 1),
        metal_gpu_mb=round(_get_metal_memory_mb(), 2),
        gc_objects=_get_gc_objects(),
    )


class MemoryDashboard:
    """Interactive memory dashboard for monitoring and visualization.

    Provides text-based visualizations suitable for terminal output.
    """

    def __init__(self, sampler: MemorySampler | None = None) -> None:
        """Initialize the dashboard.

        Args:
            sampler: MemorySampler to use (defaults to global singleton)
        """
        self._sampler = sampler or get_memory_sampler()
        self._snapshots: list[MemorySnapshot] = []

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background memory monitoring.

        Args:
            interval: Sampling interval in seconds
        """
        self._sampler._interval = interval
        self._sampler.start()

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._sampler.stop()

    def take_snapshot(self) -> MemorySnapshot:
        """Take and store a memory snapshot.

        Returns:
            The captured snapshot
        """
        snapshot = take_snapshot()
        self._snapshots.append(snapshot)
        return snapshot

    def get_current_status(self) -> dict[str, Any]:
        """Get current memory status.

        Returns:
            Dictionary with current memory state
        """
        snapshot = take_snapshot()
        return asdict(snapshot)

    def render_ascii_chart(self, width: int = 60, height: int = 10) -> str:
        """Render an ASCII chart of memory usage over time.

        Args:
            width: Chart width in characters
            height: Chart height in lines

        Returns:
            ASCII art representation of memory usage
        """
        samples = self._sampler.get_samples()

        if not samples:
            return "No memory samples available. Start monitoring first."

        # Use last N samples that fit in width
        samples = samples[-width:]
        rss_values = [s.rss_mb for s in samples]

        if not rss_values:
            return "No memory data to display."

        min_val = min(rss_values)
        max_val = max(rss_values)
        range_val = max_val - min_val

        if range_val == 0:
            range_val = 1  # Avoid division by zero

        lines = []

        # Header
        lines.append(f"Memory Usage (RSS MB) - Last {len(samples)} samples")
        lines.append(
            f"Max: {max_val:.1f} MB | Min: {min_val:.1f} MB | Current: {rss_values[-1]:.1f} MB"
        )
        lines.append("-" * width)

        # Build the chart
        chart = [[" " for _ in range(len(rss_values))] for _ in range(height)]

        for col, val in enumerate(rss_values):
            # Normalize to 0-height range
            normalized = (val - min_val) / range_val
            row = int(normalized * (height - 1))
            chart[height - 1 - row][col] = "*"

        # Render with Y-axis labels
        for i, chart_row in enumerate(chart):
            y_val = max_val - (i / (height - 1)) * range_val
            label = f"{y_val:6.1f} |"
            lines.append(label + "".join(chart_row))

        # X-axis
        lines.append(" " * 8 + "-" * len(rss_values))

        return "\n".join(lines)

    def render_summary(self) -> str:
        """Render a text summary of memory status.

        Returns:
            Formatted text summary
        """
        snapshot = take_snapshot()
        stats = self._sampler.get_stats()

        lines = [
            "=" * 50,
            "JARVIS Memory Dashboard",
            "=" * 50,
            "",
            "Current Status:",
            f"  Process RSS:     {snapshot.process_rss_mb:8.1f} MB",
            f"  Process VMS:     {snapshot.process_vms_mb:8.1f} MB",
            f"  Metal GPU:       {snapshot.metal_gpu_mb:8.1f} MB",
            f"  GC Objects:      {snapshot.gc_objects:8,}",
            "",
            "System Memory:",
            f"  Total:           {snapshot.system_total_gb:8.2f} GB",
            f"  Available:       {snapshot.system_available_gb:8.2f} GB",
            f"  Used:            {snapshot.system_used_gb:8.2f} GB ({snapshot.system_percent}%)",
            "",
            "Sampling Statistics:",
            f"  Sample Count:    {stats.get('sample_count', 0):8}",
            f"  Peak RSS:        {stats.get('peak_rss_mb', 0):8.1f} MB",
            f"  Average RSS:     {stats.get('avg_rss_mb', 0):8.1f} MB",
            f"  Min RSS:         {stats.get('min_rss_mb', 0):8.1f} MB",
            "",
            "=" * 50,
        ]

        return "\n".join(lines)

    def export_json(self, filepath: Path | str) -> None:
        """Export memory data to JSON file.

        Args:
            filepath: Path to write JSON data
        """
        samples = self._sampler.get_samples()
        data = {
            "exported_at": datetime.now(UTC).isoformat(),
            "sample_count": len(samples),
            "stats": self._sampler.get_stats(),
            "current": asdict(take_snapshot()),
            "samples": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "rss_mb": s.rss_mb,
                    "vms_mb": s.vms_mb,
                    "available_gb": s.available_gb,
                }
                for s in samples
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: Path | str) -> None:
        """Export memory samples to CSV file.

        Args:
            filepath: Path to write CSV data
        """
        samples = self._sampler.get_samples()

        with open(filepath, "w") as f:
            f.write("timestamp,rss_mb,vms_mb,percent,available_gb\n")
            for s in samples:
                f.write(
                    f"{s.timestamp.isoformat()},{s.rss_mb:.2f},{s.vms_mb:.2f},"
                    f"{s.percent:.2f},{s.available_gb:.2f}\n"
                )


def run_memory_watch(duration_seconds: int = 60, interval: float = 1.0) -> dict[str, Any]:
    """Run memory monitoring for a specified duration.

    Args:
        duration_seconds: How long to monitor
        interval: Sampling interval

    Returns:
        Dictionary with memory statistics
    """
    dashboard = MemoryDashboard()
    dashboard.start_monitoring(interval=interval)

    print(f"Monitoring memory for {duration_seconds} seconds...")

    try:
        time.sleep(duration_seconds)
    finally:
        dashboard.stop_monitoring()

    return dashboard._sampler.get_stats()


def main() -> None:
    """Main entry point for dashboard CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Memory Dashboard")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Monitoring duration in seconds",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export data to file (JSON or CSV based on extension)",
    )

    args = parser.parse_args()

    dashboard = MemoryDashboard()
    dashboard.start_monitoring(interval=args.interval)

    try:
        print("Starting memory monitoring. Press Ctrl+C to stop.")
        print()

        for _ in range(args.duration):
            # Clear screen and print summary
            print("\033[2J\033[H")  # ANSI clear screen
            print(dashboard.render_summary())
            print()
            print(dashboard.render_ascii_chart())
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dashboard.stop_monitoring()

        if args.export:
            export_path = Path(args.export)
            if export_path.suffix == ".csv":
                dashboard.export_csv(export_path)
            else:
                dashboard.export_json(export_path)
            print(f"Data exported to: {export_path}")

        print("\nFinal Statistics:")
        print(dashboard.render_summary())


if __name__ == "__main__":
    main()
