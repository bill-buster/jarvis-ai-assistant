# Performance Tuning Guide

This document covers performance monitoring, optimization techniques, and tuning guidelines for JARVIS.

## Table of Contents

- [Metrics and Monitoring](#metrics-and-monitoring)
- [Performance Optimizations](#performance-optimizations)
- [Metrics Collection Classes](#metrics-collection-classes)
- [Caching Strategy](#caching-strategy)
- [Memory Management](#memory-management)
- [Profiling Tools](#profiling-tools)
- [Best Practices](#best-practices)
- [Validation Gates](#validation-gates)

## Metrics and Monitoring

### Request Metrics Middleware

All API requests (except `/metrics` endpoints) are automatically instrumented via middleware in `api/main.py`:

- Request counts are tracked by endpoint path and HTTP method
- Request latency is recorded using histograms with configurable buckets
- Response headers include `X-Response-Time` with the request duration

### Prometheus-Compatible Metrics

JARVIS exposes Prometheus-compatible metrics at `/metrics`:

```bash
# Fetch metrics
curl http://localhost:8742/metrics
```

Available metrics:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `jarvis_memory_rss_bytes` | gauge | - | Process RSS memory in bytes |
| `jarvis_memory_vms_bytes` | gauge | - | Process virtual memory in bytes |
| `jarvis_memory_available_bytes` | gauge | - | System available memory |
| `jarvis_memory_total_bytes` | gauge | - | System total memory |
| `jarvis_requests_total` | counter | endpoint, method | Request count by endpoint/method |
| `jarvis_request_duration_seconds` | histogram | operation | Request latency distribution |
| `jarvis_uptime_seconds` | gauge | - | Time since metrics collection started |

Default histogram buckets (in seconds): 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf

### Routing Metrics (SQLite)

Routing decisions are logged to a local SQLite database for offline analysis:

- Storage: `~/.jarvis/metrics.db`
- Data: per-request routing decision, similarity score, cache hit, model loaded flag, FAISS candidate count, and latency breakdowns

Analyze collected data:

```bash
uv run python -m scripts.analyze_routing_metrics
```

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
# Returns: {"timestamp": "...", "rss_mb": 123.45, "vms_mb": 456.78, "percent": 1.5, "available_gb": 12.3}

# Trigger garbage collection
curl -X POST http://localhost:8742/metrics/gc
# Returns: {"collected_objects": 123, "rss_before_mb": 100, "rss_after_mb": 95, "rss_freed_mb": 5}

# Reset metrics counters (request counts and latency histograms, not memory sampler)
curl -X POST http://localhost:8742/metrics/reset
# Returns: {"status": "ok", "message": "Metrics counters reset"}
```

## Performance Optimizations

### Template Matching (models/templates.py)

The template matcher uses several optimizations:

1. **Pre-normalized Embeddings**: Pattern embeddings are normalized once at initialization, reducing cosine similarity computation from O(n*d) to O(n).

2. **Query Embedding Cache**: An LRU cache (500 entries, configurable via `QUERY_CACHE_SIZE`) stores query embeddings. Repeated queries skip the encoding step entirely.

3. **Batch Encoding**: All pattern embeddings are computed in a single batch call during initialization using the `all-MiniLM-L6-v2` sentence transformer model.

4. **Similarity Threshold**: TemplateMatcher uses a default 0.7 cosine threshold (`SIMILARITY_THRESHOLD`).

```python
# Cache statistics available via
from models.templates import TemplateMatcher

matcher = TemplateMatcher()
stats = matcher.get_cache_stats()
# Returns: {"size": 42, "maxsize": 500, "hits": 100, "misses": 10, "hit_rate": 0.91}

# Clear caches when memory is needed
matcher.clear_cache()
```

### Message Parsing (integrations/imessage/parser.py)

Optimizations for iMessage parsing:

1. **Pre-compiled Regex**: Patterns are compiled once at module load.

2. **Attributed Body Cache**: LRU cache (1000 entries) for parsed `attributedBody` blobs.

3. **Lookup Tables**: Reaction types use O(1) dictionary lookups instead of conditional chains.

4. **Frozen Sets**: Skip strings use frozen sets for O(1) membership testing.

### API Caching

TTL caches reduce load on expensive operations:

| Cache | TTL | Max Size | Purpose |
|-------|-----|----------|---------|
| Conversation list | 30s | 50 | Frequently refreshed conversation listing |
| Health status | 5s | 10 | Fast health checks without full computation |
| Model info | 60s | 10 | Model metadata rarely changes |

```python
from jarvis.metrics import get_conversation_cache, get_health_cache, get_model_info_cache

# Invalidate caches when data changes
cache = get_conversation_cache()
cache.invalidate()  # Clear all
cache.invalidate("conversations:50:none:none")  # Clear specific key

# Check cache statistics
stats = cache.stats()
# Returns: {"size": 5, "maxsize": 50, "ttl_seconds": 30.0, "hits": 100, "misses": 10, "hit_rate": 0.91}
```

## Metrics Collection Classes

The `jarvis/metrics.py` module provides thread-safe classes for collecting performance metrics:

### MemorySampler

Background memory sampling with configurable interval:

```python
from jarvis.metrics import get_memory_sampler

sampler = get_memory_sampler()
sampler.start()  # Start background sampling (default: 1s interval)

# Get statistics
stats = sampler.get_stats()
# Returns: {
#     "sample_count": 100,
#     "current_rss_mb": 150.5,
#     "peak_rss_mb": 200.3,
#     "avg_rss_mb": 175.2,
#     "min_rss_mb": 120.1,
#     "available_gb": 12.5,
#     "memory_percent": 1.8
# }

sampler.stop()  # Stop background sampling
```

### RequestCounter

Track API request counts by endpoint and method:

```python
from jarvis.metrics import get_request_counter

counter = get_request_counter()
counter.increment("/health", "GET")

# Get all counts
all_counts = counter.get_all()  # {"endpoint": {"GET": 10, "POST": 5}}

# Get stats
stats = counter.get_stats()
# Returns: {
#     "total_requests": 100,
#     "endpoints": 10,
#     "requests_per_second": 1.5,
#     "uptime_seconds": 66.7
# }
```

### LatencyHistogram

Prometheus-compatible histogram for latency tracking:

```python
from jarvis.metrics import get_latency_histogram

histogram = get_latency_histogram()

# Record an observation
histogram.observe("database_query", 0.05)

# Use as context manager
with histogram.time("model_inference"):
    # ... operation being timed
    pass

# Get percentiles
percentiles = histogram.get_percentiles("database_query")
# Returns: {"p50": 0.045, "p90": 0.08, "p95": 0.1, "p99": 0.15}

# Get detailed stats
stats = histogram.get_stats("database_query")
# Returns: {"count": 100, "mean_ms": 50.5, "min_ms": 10, "max_ms": 200, ...}
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

# Unload sentence transformer model when not needed
from models.templates import unload_sentence_model, is_sentence_model_loaded

if is_sentence_model_loaded():
    unload_sentence_model()  # Frees ~100MB

# Clear template matcher cache after unloading model
from models.templates import TemplateMatcher
matcher = TemplateMatcher()
matcher.clear_cache()  # Clear embeddings and query cache
```

## Profiling Tools

### Memory Dashboard

Run the interactive memory dashboard:

```bash
python -m benchmarks.memory.dashboard --duration 30 --interval 1.0
```

Options:
- `--duration`: Monitoring duration in seconds (default: 30)
- `--interval`: Sampling interval in seconds (default: 1.0)
- `--export`: Export data to file (JSON or CSV based on file extension)

The dashboard provides:
- Real-time ASCII chart of memory usage
- Process memory (RSS, VMS)
- System memory (total, available, used)
- Metal GPU memory (if MLX is loaded)
- GC object count

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
from benchmarks.memory.dashboard import MemoryDashboard, take_snapshot, run_memory_watch

# Take a snapshot
snapshot = take_snapshot()
print(f"RSS: {snapshot.process_rss_mb}MB")
print(f"VMS: {snapshot.process_vms_mb}MB")
print(f"GPU: {snapshot.metal_gpu_mb}MB")
print(f"GC Objects: {snapshot.gc_objects}")

# Run dashboard
dashboard = MemoryDashboard()
dashboard.start_monitoring(interval=1.0)

# Print ASCII chart
print(dashboard.render_ascii_chart())

# Print text summary
print(dashboard.render_summary())

# Export data
dashboard.export_json("memory_data.json")
dashboard.export_csv("memory_data.csv")

dashboard.stop_monitoring()

# Or use convenience function for timed monitoring
stats = run_memory_watch(duration_seconds=60, interval=1.0)
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

**Useful Grafana Panels:**

1. **Memory Usage Over Time**
   ```promql
   jarvis_memory_rss_bytes / 1024 / 1024  # RSS in MB
   ```

2. **System Memory Pressure**
   ```promql
   1 - (jarvis_memory_available_bytes / jarvis_memory_total_bytes)
   ```

3. **Request Rate by Endpoint**
   ```promql
   rate(jarvis_requests_total[5m])
   ```

4. **Latency Percentiles (p99)**
   ```promql
   histogram_quantile(0.99, rate(jarvis_request_duration_seconds_bucket[5m]))
   ```

5. **Mean Request Duration**
   ```promql
   rate(jarvis_request_duration_seconds_sum[5m]) / rate(jarvis_request_duration_seconds_count[5m])
   ```

**Alert Examples:**

```yaml
# Alert when memory usage exceeds 80% of system memory
- alert: JarvisHighMemory
  expr: jarvis_memory_rss_bytes / jarvis_memory_total_bytes > 0.8
  for: 5m
  labels:
    severity: warning

# Alert when p99 latency exceeds 5 seconds
- alert: JarvisHighLatency
  expr: histogram_quantile(0.99, rate(jarvis_request_duration_seconds_bucket[5m])) > 5
  for: 2m
  labels:
    severity: warning
```

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
