# Performance Tuning Guide

This document covers performance monitoring, optimization techniques, and tuning guidelines for JARVIS.

## Table of Contents

- [Metrics and Monitoring](#metrics-and-monitoring)
- [Performance Optimizations](#performance-optimizations)
- [Caching Strategy](#caching-strategy)
- [Memory Management](#memory-management)
- [Profiling Tools](#profiling-tools)
- [Best Practices](#best-practices)

## Metrics and Monitoring

### Prometheus-Compatible Metrics

JARVIS exposes Prometheus-compatible metrics at `/metrics`:

```bash
# Fetch metrics
curl http://localhost:8742/metrics
```

Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `jarvis_memory_rss_bytes` | gauge | Process RSS memory in bytes |
| `jarvis_memory_vms_bytes` | gauge | Process virtual memory in bytes |
| `jarvis_memory_available_bytes` | gauge | System available memory |
| `jarvis_memory_total_bytes` | gauge | System total memory |
| `jarvis_requests_total` | counter | Request count by endpoint/method |
| `jarvis_request_duration_seconds` | histogram | Request latency distribution |
| `jarvis_uptime_seconds` | gauge | Time since metrics collection started |

### Detailed Memory Breakdown

Get detailed memory statistics:

```bash
curl http://localhost:8742/metrics/memory
```

Response includes:
- Process memory (RSS, VMS, percent of system)
- System memory (total, available, used)
- Metal GPU memory (if MLX is loaded)
- Historical trend data

### Latency Percentiles

Get latency statistics by operation:

```bash
curl http://localhost:8742/metrics/latency
```

Response includes p50, p90, p95, and p99 percentiles for each tracked operation.

### Request Counts

Get request counts by endpoint:

```bash
curl http://localhost:8742/metrics/requests
```

### Manual Operations

```bash
# Take immediate memory sample
curl -X POST http://localhost:8742/metrics/sample

# Trigger garbage collection
curl -X POST http://localhost:8742/metrics/gc

# Reset metrics counters
curl -X POST http://localhost:8742/metrics/reset
```

## Performance Optimizations

### Template Matching (models/templates.py)

The template matcher uses several optimizations:

1. **Pre-normalized Embeddings**: Pattern embeddings are normalized once at initialization, reducing cosine similarity computation from O(n*d) to O(n).

2. **Query Embedding Cache**: An LRU cache (500 entries) stores query embeddings. Repeated queries skip the encoding step entirely.

3. **Batch Encoding**: All pattern embeddings are computed in a single batch call during initialization.

```python
# Cache statistics available via
matcher = TemplateMatcher()
stats = matcher.get_cache_stats()
# Returns: {"size": 42, "hits": 100, "misses": 10, "hit_rate": 0.91}
```

### Message Parsing (integrations/imessage/parser.py)

Optimizations for iMessage parsing:

1. **Pre-compiled Regex**: Patterns are compiled once at module load.

2. **Attributed Body Cache**: LRU cache (1000 entries) for parsed `attributedBody` blobs.

3. **Lookup Tables**: Reaction types use O(1) dictionary lookups instead of conditional chains.

4. **Frozen Sets**: Skip strings use frozen sets for O(1) membership testing.

### API Caching

TTL caches reduce load on expensive operations:

| Cache | TTL | Purpose |
|-------|-----|---------|
| Conversation list | 30s | Frequently refreshed conversation listing |
| Health status | 5s | Fast health checks without full computation |
| Model info | 60s | Model metadata rarely changes |

```python
from jarvis.metrics import get_conversation_cache, get_health_cache

# Invalidate caches when data changes
cache = get_conversation_cache()
cache.invalidate()  # Clear all
cache.invalidate("conversations:50:none:none")  # Clear specific key
```

## Caching Strategy

### When to Cache

Cache when:
- Data is expensive to compute
- Data doesn't change frequently
- Staleness is acceptable for the use case

### Cache Configuration

```python
from jarvis.metrics import TTLCache

# Create custom cache
cache = TTLCache(
    ttl_seconds=30.0,  # Time-to-live
    maxsize=100,       # Maximum entries
)

# Use the cache
found, value = cache.get("key")
if not found:
    value = expensive_computation()
    cache.set("key", value)
```

### Cache Monitoring

Monitor cache effectiveness:

```python
stats = cache.stats()
# {
#     "size": 45,
#     "maxsize": 100,
#     "ttl_seconds": 30.0,
#     "hits": 1000,
#     "misses": 50,
#     "hit_rate": 0.95
# }
```

Target hit rate: >80% for frequently accessed data.

## Memory Management

### Memory Budget

JARVIS targets 8GB minimum systems with:
- Model: ~3-4GB for Qwen2.5-0.5B-4bit
- Template embeddings: ~100MB for all-MiniLM-L6-v2
- Working memory: ~500MB for caches and operations

### Memory Modes

The memory controller operates in three modes:

| Mode | Available Memory | Behavior |
|------|-----------------|----------|
| FULL | >= 4GB | Full model + all features |
| LITE | 2-4GB | Reduced context, aggressive caching |
| MINIMAL | < 2GB | Template-only responses |

### Monitoring Memory

```python
from jarvis.metrics import get_memory_sampler

sampler = get_memory_sampler()
sampler.start()  # Begin background sampling

# Check current stats
stats = sampler.get_stats()
print(f"Current: {stats['current_rss_mb']}MB")
print(f"Peak: {stats['peak_rss_mb']}MB")
print(f"Available: {stats['available_gb']}GB")

sampler.stop()  # Stop sampling
```

### Freeing Memory

```python
from jarvis.metrics import force_gc

# Force garbage collection
result = force_gc()
print(f"Freed: {result['rss_freed_mb']}MB")

# Unload models when not needed
from models.templates import unload_sentence_model
unload_sentence_model()
```

## Profiling Tools

### Memory Dashboard

Run the interactive memory dashboard:

```bash
python -m benchmarks.memory.dashboard --duration 60 --interval 1.0
```

Options:
- `--duration`: Monitoring duration in seconds
- `--interval`: Sampling interval in seconds
- `--export`: Export data to file (JSON or CSV)

### Benchmark Suite

Run performance benchmarks:

```bash
# Memory profiling
python -m benchmarks.memory.run

# Latency benchmarks
python -m benchmarks.latency.run

# Full evaluation
./scripts/overnight_eval.sh
```

### Using the Python API

```python
from benchmarks.memory.dashboard import MemoryDashboard, take_snapshot

# Take a snapshot
snapshot = take_snapshot()
print(f"RSS: {snapshot.process_rss_mb}MB")
print(f"GPU: {snapshot.metal_gpu_mb}MB")

# Run dashboard
dashboard = MemoryDashboard()
dashboard.start_monitoring(interval=1.0)

# Print ASCII chart
print(dashboard.render_ascii_chart())

# Export data
dashboard.export_json("memory_data.json")
dashboard.export_csv("memory_data.csv")

dashboard.stop_monitoring()
```

## Best Practices

### Request Handling

1. **Use caching for repeated queries**: Conversation lists, health checks.
2. **Batch operations**: Group multiple iMessage queries when possible.
3. **Lazy loading**: Don't load models until needed.

### Memory Efficiency

1. **Unload unused models**: Call `unload_sentence_model()` when template matching isn't needed.
2. **Clear caches**: Invalidate caches when memory is low.
3. **Monitor trends**: Watch for memory leaks using the dashboard.

### Latency Optimization

1. **Template matching first**: Check templates before invoking the LLM.
2. **Cache warm-up**: Pre-populate caches during startup for critical paths.
3. **Async where possible**: Use async endpoints for I/O-bound operations.

### Monitoring

1. **Set up alerts**: Monitor `jarvis_memory_rss_bytes` for OOM prevention.
2. **Track percentiles**: Use p99 latency to catch tail latencies.
3. **Log slow operations**: Requests >1s should be investigated.

### Example: Prometheus + Grafana Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'jarvis'
    static_configs:
      - targets: ['localhost:8742']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Useful Grafana panels:
- Memory usage over time
- Request rate by endpoint
- Latency percentiles (p50, p90, p99)
- Error rate (from response codes)

## Validation Gates

JARVIS uses validation gates to ensure performance targets:

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Model stack memory | <5.5GB | 5.5-6.5GB | >6.5GB |
| G2 | Mean HHEM score | >=0.5 | 0.4-0.5 | <0.4 |
| G3 | Warm-start latency | <3s | 3-5s | >5s |
| G4 | Cold-start latency | <15s | 15-20s | >20s |

Run gate checks:

```bash
python scripts/check_gates.py
```

See [BENCHMARKS.md](BENCHMARKS.md) for detailed benchmark results.
