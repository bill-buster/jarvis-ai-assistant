"""Metrics API endpoints for performance monitoring.

Provides Prometheus-compatible metrics and detailed memory/latency breakdowns.
"""

from __future__ import annotations

import os
from typing import Any

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from api.ratelimit import RATE_LIMIT_READ, RATE_LIMIT_WRITE, limiter
from jarvis.metrics import (
    force_gc,
    get_latency_histogram,
    get_memory_sampler,
    get_request_counter,
)

router = APIRouter(prefix="/metrics", tags=["metrics"])

# Constants
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3


def _format_prometheus_metric(
    name: str,
    value: float | int,
    labels: dict[str, str] | None = None,
    metric_type: str = "gauge",
    help_text: str = "",
) -> str:
    """Format a metric in Prometheus text format.

    Args:
        name: Metric name
        value: Metric value
        labels: Optional labels
        metric_type: Type (gauge, counter, histogram)
        help_text: Help text

    Returns:
        Prometheus-formatted metric string
    """
    lines = []
    if help_text:
        lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")

    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        lines.append(f"{name}{{{label_str}}} {value}")
    else:
        lines.append(f"{name} {value}")

    return "\n".join(lines)


@router.get("", response_class=PlainTextResponse)
@limiter.limit(RATE_LIMIT_READ)
async def get_prometheus_metrics(request: Request) -> str:
    """Get all metrics in Prometheus text format.

    Returns metrics compatible with Prometheus scraping:
    - jarvis_memory_rss_bytes: Current RSS memory usage
    - jarvis_memory_vms_bytes: Current VMS memory usage
    - jarvis_memory_available_bytes: System available memory
    - jarvis_requests_total: Total request count by endpoint
    - jarvis_request_duration_seconds: Request latency histogram
    """
    lines: list[str] = []

    # Process memory metrics
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()

    lines.append(
        _format_prometheus_metric(
            "jarvis_memory_rss_bytes",
            mem_info.rss,
            metric_type="gauge",
            help_text="Resident Set Size in bytes",
        )
    )
    lines.append(
        _format_prometheus_metric(
            "jarvis_memory_vms_bytes",
            mem_info.vms,
            metric_type="gauge",
            help_text="Virtual Memory Size in bytes",
        )
    )
    lines.append(
        _format_prometheus_metric(
            "jarvis_memory_available_bytes",
            system_mem.available,
            metric_type="gauge",
            help_text="System available memory in bytes",
        )
    )
    lines.append(
        _format_prometheus_metric(
            "jarvis_memory_total_bytes",
            system_mem.total,
            metric_type="gauge",
            help_text="System total memory in bytes",
        )
    )

    # Request counter metrics
    counter = get_request_counter()
    all_counts = counter.get_all()
    lines.append("# HELP jarvis_requests_total Total number of requests by endpoint")
    lines.append("# TYPE jarvis_requests_total counter")
    for endpoint, methods in all_counts.items():
        for method, count in methods.items():
            lines.append(
                f'jarvis_requests_total{{endpoint="{endpoint}",method="{method}"}} {count}'
            )

    # Latency histogram metrics
    histogram = get_latency_histogram()
    stats = histogram.get_stats()

    if stats:
        lines.append("# HELP jarvis_request_duration_seconds Request duration in seconds")
        lines.append("# TYPE jarvis_request_duration_seconds histogram")
        for operation, op_stats in stats.items():
            data = histogram.get_histogram_data(operation)
            if data:
                # Output bucket counts
                for bucket, count in zip(data["buckets"], data["counts"], strict=True):
                    bucket_label = "+Inf" if bucket == float("inf") else f"{bucket}"
                    lines.append(
                        f"jarvis_request_duration_seconds_bucket"
                        f'{{operation="{operation}",le="{bucket_label}"}} {count}'
                    )
                lines.append(
                    f'jarvis_request_duration_seconds_sum{{operation="{operation}"}} '
                    f"{data['total_sum']}"
                )
                lines.append(
                    f'jarvis_request_duration_seconds_count{{operation="{operation}"}} '
                    f"{data['total_count']}"
                )

    # Uptime metric
    counter_stats = counter.get_stats()
    lines.append(
        _format_prometheus_metric(
            "jarvis_uptime_seconds",
            counter_stats["uptime_seconds"],
            metric_type="gauge",
            help_text="Time since metrics collection started",
        )
    )

    return "\n".join(lines) + "\n"


@router.get("/memory")
@limiter.limit(RATE_LIMIT_READ)
async def get_memory_metrics(request: Request) -> dict[str, Any]:
    """Get detailed memory breakdown.

    Returns:
        Detailed memory statistics including:
        - Current process memory (RSS, VMS)
        - System memory (total, available, used)
        - Memory sampling history stats
        - Metal GPU memory (if available)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()

    sampler = get_memory_sampler()
    sample_stats = sampler.get_stats()

    # Try to get Metal memory if MLX is available
    metal_mb = 0.0
    try:
        import mlx.core as mx

        metal_mb = mx.metal.get_peak_memory() / BYTES_PER_MB
    except (ImportError, AttributeError):
        pass

    # Get recent samples for trend data
    recent_samples = sampler.get_samples()
    trend_data = []
    for sample in recent_samples[-60:]:  # Last 60 samples
        trend_data.append(
            {
                "timestamp": sample.timestamp.isoformat(),
                "rss_mb": round(sample.rss_mb, 2),
            }
        )

    return {
        "process": {
            "rss_mb": round(mem_info.rss / BYTES_PER_MB, 2),
            "vms_mb": round(mem_info.vms / BYTES_PER_MB, 2),
            "percent": round(mem_info.rss / system_mem.total * 100, 2),
        },
        "system": {
            "total_gb": round(system_mem.total / BYTES_PER_GB, 2),
            "available_gb": round(system_mem.available / BYTES_PER_GB, 2),
            "used_gb": round(system_mem.used / BYTES_PER_GB, 2),
            "percent": round(system_mem.percent, 2),
        },
        "metal_gpu_mb": round(metal_mb, 2),
        "sampling": sample_stats,
        "trend": trend_data,
    }


@router.get("/latency")
@limiter.limit(RATE_LIMIT_READ)
async def get_latency_metrics(request: Request) -> dict[str, Any]:
    """Get request latency percentiles.

    Returns:
        Latency statistics by operation including:
        - Count of observations
        - Mean, min, max latency
        - p50, p90, p95, p99 percentiles
    """
    histogram = get_latency_histogram()
    all_stats = histogram.get_stats()

    result: dict[str, Any] = {"operations": {}}

    for operation, stats in all_stats.items():
        percentiles = histogram.get_percentiles(operation)
        result["operations"][operation] = {
            "count": stats["count"],
            "mean_ms": stats["mean_ms"],
            "p50_ms": round(percentiles["p50"] * 1000, 3),
            "p90_ms": round(percentiles["p90"] * 1000, 3),
            "p95_ms": round(percentiles["p95"] * 1000, 3),
            "p99_ms": round(percentiles["p99"] * 1000, 3),
        }

    # Add summary stats
    counter = get_request_counter()
    counter_stats = counter.get_stats()
    result["summary"] = {
        "total_requests": counter_stats["total_requests"],
        "requests_per_second": counter_stats["requests_per_second"],
        "uptime_seconds": counter_stats["uptime_seconds"],
        "endpoint_count": counter_stats["endpoints"],
    }

    return result


@router.get("/requests")
@limiter.limit(RATE_LIMIT_READ)
async def get_request_metrics(request: Request) -> dict[str, Any]:
    """Get request count metrics by endpoint.

    Returns:
        Request counts grouped by endpoint and method
    """
    counter = get_request_counter()
    return {
        "endpoints": counter.get_all(),
        "stats": counter.get_stats(),
    }


@router.post("/gc")
@limiter.limit(RATE_LIMIT_WRITE)
async def trigger_gc(request: Request) -> dict[str, Any]:
    """Trigger garbage collection and return memory delta.

    Returns:
        Memory usage before and after GC, plus objects collected
    """
    return force_gc()


@router.post("/sample")
@limiter.limit(RATE_LIMIT_WRITE)
async def take_memory_sample(request: Request) -> dict[str, Any]:
    """Take an immediate memory sample.

    Returns:
        Current memory sample data
    """
    sampler = get_memory_sampler()
    sample = sampler.sample_now()
    return {
        "timestamp": sample.timestamp.isoformat(),
        "rss_mb": round(sample.rss_mb, 2),
        "vms_mb": round(sample.vms_mb, 2),
        "percent": round(sample.percent, 2),
        "available_gb": round(sample.available_gb, 2),
    }


@router.post("/reset")
@limiter.limit(RATE_LIMIT_WRITE)
async def reset_metrics(request: Request) -> dict[str, str]:
    """Reset all metrics counters (not memory sampler).

    Returns:
        Confirmation message
    """
    counter = get_request_counter()
    histogram = get_latency_histogram()

    counter.reset()
    histogram.reset()

    return {"status": "ok", "message": "Metrics counters reset"}
