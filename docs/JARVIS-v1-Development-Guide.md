# JARVIS v1: Development Guide
## Parallel Agent Development with Built-In Self-Critique

**Document Status**: HISTORICAL - Development Complete
**Date**: January 25, 2026
**Purpose**: Operational guide for multi-agent parallel development with quality gates

> **Note**: This document was the original development plan. Implementation is now complete.
> All workstreams (WS1-WS8, WS10) are implemented. WS9 was removed from scope.
> The actual model used is Qwen2.5-0.5B-Instruct-4bit (smaller than the 3B originally planned).
> See `docs/GUIDE.md` for current documentation index.

---

## Executive Summary

This document serves as the operational guide for building JARVIS v1, a local-first AI assistant for macOS. It is optimized for:

1. **Parallel agent development**: 10 independent workstreams that can run simultaneously
2. **Overnight evaluation cycles**: Heavy benchmarks run while you sleep, results inform next day's work
3. **8GB RAM constraints**: Leverage Claude Code Web for development, local Mac for sequential evals
4. **Self-critique mechanisms**: Every agent validates its own work before declaring "done"
5. **Portfolio showcase**: Structure that demonstrates engineering rigor for job applications

The core insight: with AI agents, you're not limited by your own typing speed. You can spin up multiple agents working in parallel, but only if the work is structured to avoid conflicts. This guide provides that structure.

---

## Part 1: Repository Strategy

### Decision: Start Fresh, Extract Surgically

**Do NOT** clone or fork summarizationv2 or jarvisv0. Instead:

1. Create a fresh repository with clean commit history (portfolio-friendly)
2. Identify specific, working code from existing projects
3. Extract and adapt that code into the new structure
4. Credit original work in commit messages ("Ported from summarizationv2: chat.db parsing logic")

**Why this approach matters for your portfolio:**
- Clean commit history tells a coherent story
- No embarrassing early experiments visible
- Architecture decisions are intentional, not inherited
- Reviewers see professional project structure from commit #1

### What to Extract from Existing Projects

Before creating the new repo, audit existing projects for reusable code:

**From summarizationv2** (likely candidates):
- chat.db SQLite query patterns
- iMessage schema parsing
- Any working memory profiling code
- Test fixtures with sample data

**From jarvisv0** (likely candidates):
- MLX model loading patterns
- HHEM integration code
- Configuration management patterns
- Any prompt templates that worked well

**Extraction Process** (run this as a dedicated task):
```
TASK: Audit existing projects for reusable code

For each of [summarizationv2, jarvisv0]:
1. List all Python files with brief description of contents
2. Identify code that is:
   - Working (has tests or has been manually verified)
   - Relevant to JARVIS v1 workstreams
   - Clean enough to port (not heavily coupled to old architecture)
3. For each identified piece:
   - Note the source file and line numbers
   - Note what workstream it belongs to
   - Note any modifications needed for new interfaces
4. Create a manifest: extraction_manifest.json

SELF-CRITIQUE CHECKPOINT:
- Did I actually verify the code works, or am I assuming?
- Is this code better than writing fresh, or am I just being lazy?
- Does porting this code bring along unwanted dependencies?
```

---

## Part 2: Repository Structure

Create this structure before spawning any workstream agents:

```
jarvis-ai-assistant/
├── README.md                    # Hero README (write last, after project works)
├── ARCHITECTURE.md              # Technical deep-dive for interviews
├── BENCHMARKS.md                # Auto-generated from eval results
├── pyproject.toml               # Single source of dependencies
├── .github/
│   └── workflows/
│       └── ci.yml               # Run tests on PR
│
├── contracts/                   # 🔑 THE KEY TO PARALLEL WORK
│   ├── __init__.py
│   ├── memory.py                # MemoryProfile, MemoryController interfaces
│   ├── hallucination.py         # HHEMResult, evaluate_summary interfaces
│   ├── coverage.py              # CoverageResult, match_to_template interfaces
│   ├── latency.py               # LatencyResult, benchmark_latency interfaces
│   ├── health.py                # PermissionStatus, DegradationPolicy interfaces
│   ├── models.py                # Generator, GenerationRequest interfaces
│   └── imessage.py              # Message, iMessageReader interfaces
│
├── benchmarks/                  # WORKSTREAMS 1-4
│   ├── __init__.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── profiler.py          # Implementation of memory profiling
│   │   └── run.py               # CLI entrypoint
│   ├── hallucination/
│   │   ├── __init__.py
│   │   ├── hhem.py              # HHEM wrapper
│   │   ├── datasets.py          # Test case generation
│   │   └── run.py               # CLI entrypoint
│   ├── coverage/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Semantic similarity matching
│   │   ├── templates.py         # Template definitions
│   │   └── run.py               # CLI entrypoint
│   └── latency/
│       ├── __init__.py
│       ├── timer.py             # High-precision timing
│       └── run.py               # CLI entrypoint
│
├── core/                        # WORKSTREAMS 5-7
│   ├── __init__.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── controller.py        # Adaptive memory management
│   │   └── monitor.py           # System memory monitoring
│   ├── health/
│   │   ├── __init__.py
│   │   ├── degradation.py       # Graceful degradation
│   │   ├── permissions.py       # TCC permission monitoring
│   │   └── schema.py            # chat.db schema detection
│   └── config/
│       ├── __init__.py
│       └── settings.py          # Configuration management
│
├── models/                      # WORKSTREAM 8
│   ├── __init__.py
│   ├── loader.py                # MLX model loading
│   ├── generator.py             # RAG + few-shot generation
│   └── templates.py             # Template matching
│
├── integrations/                # WORKSTREAM 10
│   ├── __init__.py
│   └── imessage/
│       ├── __init__.py
│       ├── reader.py            # chat.db reader
│       ├── queries.py           # SQL for different schemas
│       └── parser.py            # Message normalization
│
├── tests/
│   ├── __init__.py
│   ├── unit/                    # Fast tests, no external deps
│   ├── integration/             # Tests with real DBs/APIs
│   └── fixtures/                # Sample data for tests
│
├── scripts/
│   ├── overnight_eval.sh        # Run all benchmarks sequentially
│   ├── setup_repo.py            # Initial repo setup
│   └── generate_report.py       # Create BENCHMARKS.md from results
│
├── results/                     # Gitignored except for published results
│   ├── .gitkeep
│   └── published/               # Curated results for portfolio
│
└── docs/
    ├── decisions/               # Architecture Decision Records
    ├── extraction_manifest.json # What was ported from where
    └── blog_posts/              # Pre-written LinkedIn content
```

---

## Part 3: The Contracts (Interfaces)

The `contracts/` directory is what enables parallel development. Every workstream implements these interfaces. Agents can work independently because they code against contracts, not implementations.

### contracts/memory.py

```python
"""Memory profiling and control interfaces.

Workstreams 1 and 5 implement against these contracts.
"""
from dataclasses import dataclass
from typing import Protocol
from enum import Enum


@dataclass
class MemoryProfile:
    """Result of profiling a model's memory usage."""
    model_name: str
    quantization: str
    context_length: int
    rss_mb: float           # Resident Set Size (actual RAM used)
    virtual_mb: float       # Virtual memory allocated
    metal_mb: float         # GPU memory (Apple Metal)
    load_time_seconds: float
    timestamp: str          # ISO format


class MemoryMode(Enum):
    """Operating modes based on available memory."""
    FULL = "full"           # 16GB+ : All features, concurrent models
    LITE = "lite"           # 8-16GB : Sequential loading, reduced context
    MINIMAL = "minimal"     # <8GB : Templates only, cloud fallback


@dataclass 
class MemoryState:
    """Current memory status of the system."""
    available_mb: float
    used_mb: float
    model_loaded: bool
    current_mode: MemoryMode
    pressure_level: str     # "green", "yellow", "red", "critical"


class MemoryProfiler(Protocol):
    """Interface for memory profiling (Workstream 1)."""
    
    def profile_model(self, model_path: str, context_length: int) -> MemoryProfile:
        """Profile a model's memory usage. Must unload model after profiling."""
        ...


class MemoryController(Protocol):
    """Interface for memory management (Workstream 5)."""
    
    def get_state(self) -> MemoryState:
        """Get current memory state."""
        ...
    
    def get_mode(self) -> MemoryMode:
        """Determine appropriate mode based on available memory."""
        ...
    
    def can_load_model(self, required_mb: float) -> bool:
        """Check if we have enough memory to load a model."""
        ...
    
    def request_memory(self, required_mb: float, priority: int) -> bool:
        """Request memory, potentially unloading lower-priority components."""
        ...
    
    def register_pressure_callback(self, callback: callable) -> None:
        """Register callback for memory pressure events."""
        ...
```

### contracts/hallucination.py

```python
"""Hallucination evaluation interfaces.

Workstream 2 implements against these contracts.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class HHEMResult:
    """Result of evaluating a single source/summary pair."""
    model_name: str
    prompt_template: str
    source_text: str
    generated_summary: str
    hhem_score: float       # 0.0 = hallucinated, 1.0 = grounded
    timestamp: str


@dataclass
class HHEMBenchmarkResult:
    """Aggregate results from a benchmark run."""
    model_name: str
    num_samples: int
    mean_score: float
    median_score: float
    std_score: float
    pass_rate_at_05: float  # % of samples with score >= 0.5
    pass_rate_at_07: float  # % of samples with score >= 0.7
    results: list[HHEMResult]
    timestamp: str


class HallucinationEvaluator(Protocol):
    """Interface for hallucination evaluation (Workstream 2)."""
    
    def evaluate_single(self, source: str, summary: str) -> float:
        """Return HHEM score for a source/summary pair."""
        ...
    
    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch evaluate multiple pairs. More efficient than single calls."""
        ...
    
    def run_benchmark(
        self, 
        model_name: str, 
        dataset_path: str,
        prompt_templates: list[str]
    ) -> HHEMBenchmarkResult:
        """Run full benchmark and return aggregate results."""
        ...
```

### contracts/coverage.py

```python
"""Template coverage analysis interfaces.

Workstream 3 implements against these contracts.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class TemplateMatch:
    """Result of matching a query to templates."""
    query: str
    best_template: str | None
    similarity_score: float
    matched: bool           # True if score >= threshold


@dataclass
class CoverageResult:
    """Aggregate coverage analysis results."""
    total_queries: int
    coverage_at_50: float   # % matching at 0.5 similarity
    coverage_at_70: float   # % matching at 0.7 similarity  
    coverage_at_90: float   # % matching at 0.9 similarity
    unmatched_examples: list[str]  # Sample queries that didn't match
    template_usage: dict[str, int]  # How often each template matched
    timestamp: str


class CoverageAnalyzer(Protocol):
    """Interface for template coverage analysis (Workstream 3)."""
    
    def match_query(self, query: str, threshold: float = 0.7) -> TemplateMatch:
        """Find best matching template for a query."""
        ...
    
    def analyze_dataset(self, queries: list[str]) -> CoverageResult:
        """Analyze coverage across a dataset of queries."""
        ...
    
    def get_templates(self) -> list[str]:
        """Return all available templates."""
        ...
    
    def add_template(self, template: str) -> None:
        """Add a new template to the matcher."""
        ...
```

### contracts/latency.py

```python
"""Latency benchmarking interfaces.

Workstream 4 implements against these contracts.
"""
from dataclasses import dataclass
from typing import Literal, Protocol


Scenario = Literal["cold", "warm", "hot"]


@dataclass
class LatencyResult:
    """Result of a single latency measurement."""
    scenario: Scenario
    model_name: str
    context_length: int
    output_tokens: int
    load_time_ms: float
    prefill_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    timestamp: str


@dataclass
class LatencyBenchmarkResult:
    """Aggregate latency benchmark results."""
    scenario: Scenario
    model_name: str
    num_runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    results: list[LatencyResult]
    timestamp: str


class LatencyBenchmarker(Protocol):
    """Interface for latency benchmarking (Workstream 4)."""
    
    def measure_single(
        self,
        model_path: str,
        scenario: Scenario,
        prompt: str,
        max_tokens: int
    ) -> LatencyResult:
        """Measure latency for a single generation."""
        ...
    
    def run_benchmark(
        self,
        model_path: str,
        scenario: Scenario,
        num_runs: int = 10
    ) -> LatencyBenchmarkResult:
        """Run full benchmark with statistical analysis."""
        ...
```

### contracts/health.py

```python
"""System health monitoring interfaces.

Workstreams 6 and 7 implement against these contracts.
"""
from dataclasses import dataclass
from typing import Protocol, Callable, Any
from enum import Enum


class FeatureState(Enum):
    """Health state of a feature."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class Permission(Enum):
    """macOS permissions required by JARVIS."""
    FULL_DISK_ACCESS = "full_disk_access"
    CONTACTS = "contacts"
    CALENDAR = "calendar"
    AUTOMATION = "automation"


@dataclass
class PermissionStatus:
    """Status of a single permission."""
    permission: Permission
    granted: bool
    last_checked: str
    fix_instructions: str   # User-friendly instructions if not granted


@dataclass
class SchemaInfo:
    """Information about iMessage chat.db schema."""
    version: str
    tables: list[str]
    compatible: bool
    migration_needed: bool
    known_schema: bool      # False if we don't recognize this version


@dataclass
class DegradationPolicy:
    """Policy for degrading a feature gracefully."""
    feature_name: str
    health_check: Callable[[], bool]
    degraded_behavior: Callable[..., Any]
    fallback_behavior: Callable[..., Any]
    recovery_check: Callable[[], bool]
    max_failures: int = 3


class DegradationController(Protocol):
    """Interface for graceful degradation (Workstream 6)."""
    
    def register_feature(self, policy: DegradationPolicy) -> None:
        """Register a feature with its degradation policy."""
        ...
    
    def execute(self, feature_name: str, *args, **kwargs) -> Any:
        """Execute feature with automatic fallback on failure."""
        ...
    
    def get_health(self) -> dict[str, FeatureState]:
        """Return health status of all features."""
        ...
    
    def reset_feature(self, feature_name: str) -> None:
        """Reset failure count and try healthy mode again."""
        ...


class PermissionMonitor(Protocol):
    """Interface for TCC permission monitoring (Workstream 7)."""
    
    def check_permission(self, permission: Permission) -> PermissionStatus:
        """Check if a specific permission is granted."""
        ...
    
    def check_all(self) -> list[PermissionStatus]:
        """Check all required permissions."""
        ...
    
    def wait_for_permission(self, permission: Permission, timeout_seconds: int) -> bool:
        """Block until permission granted or timeout."""
        ...


class SchemaDetector(Protocol):
    """Interface for chat.db schema detection (Workstream 7)."""
    
    def detect(self, db_path: str) -> SchemaInfo:
        """Detect schema version and compatibility."""
        ...
    
    def get_query(self, query_name: str, schema_version: str) -> str:
        """Get appropriate SQL query for the detected schema."""
        ...
```

### contracts/models.py

```python
"""Model loading and generation interfaces.

Workstream 8 implements against these contracts.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    context_documents: list[str]        # RAG context to inject
    few_shot_examples: list[tuple[str, str]]  # (input, output) pairs
    max_tokens: int = 100
    temperature: float = 0.7
    stop_sequences: list[str] | None = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    text: str
    tokens_used: int
    generation_time_ms: float
    model_name: str
    used_template: bool
    template_name: str | None
    finish_reason: str      # "stop", "length", "template"


class Generator(Protocol):
    """Interface for text generation (Workstream 8)."""
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response. May use template or model."""
        ...
    
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        ...
    
    def load(self) -> bool:
        """Load model into memory. Returns success."""
        ...
    
    def unload(self) -> None:
        """Unload model to free memory."""
        ...
    
    def get_memory_usage_mb(self) -> float:
        """Return current memory usage of the model."""
        ...
```

### contracts/imessage.py

```python
"""iMessage integration interfaces.

Workstream 10 implements against these contracts.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Message:
    """Normalized iMessage representation."""
    id: int
    chat_id: str
    sender: str             # Phone number or email
    sender_name: str | None # Resolved from contacts if available
    text: str
    date: datetime
    is_from_me: bool
    attachments: list[str]
    reply_to_id: int | None
    reactions: list[str]    # Tapback reactions


@dataclass
class Conversation:
    """iMessage conversation summary."""
    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool


class iMessageReader(Protocol):
    """Interface for iMessage integration (Workstream 10)."""
    
    def check_access(self) -> bool:
        """Check if we have permission to read chat.db."""
        ...
    
    def get_conversations(
        self, 
        limit: int = 50,
        since: datetime | None = None
    ) -> list[Conversation]:
        """Get recent conversations."""
        ...
    
    def get_messages(
        self, 
        chat_id: str, 
        limit: int = 100,
        before: datetime | None = None
    ) -> list[Message]:
        """Get messages from a conversation."""
        ...
    
    def search(self, query: str, limit: int = 50) -> list[Message]:
        """Full-text search across messages."""
        ...
    
    def get_conversation_context(
        self, 
        chat_id: str, 
        around_message_id: int,
        context_messages: int = 5
    ) -> list[Message]:
        """Get messages around a specific message for context."""
        ...
```

---

## Part 4: The 10 Workstreams

Each workstream is fully independent once contracts are defined. They can run simultaneously across multiple Claude Code Web sessions.

### Workstream Dependency Graph

```
                         contracts/
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐          ┌─────────┐
   │ WS 1-4  │         │ WS 5-7  │          │  WS 8   │
   │Benchmarks│         │  Core   │          │ Models  │
   │         │         │         │          │         │
   │ Memory  │         │ Memory  │          │ Loader  │
   │ HHEM    │         │ Health  │          │ Gen     │
   │ Coverage│         │ Degrade │          │Templates│
   │ Latency │         │ Perms   │          │         │
   └─────────┘         └────┬────┘          └────┬────┘
        │                   │                    │
        │                   └────────┬───────────┘
        │                            │
        │                            ▼
        │                      ┌───────────┐
        │                      │  WS 9-10  │
        │                      │Integration│
        │                      │           │
        │                      │ iMessage  │
        │                      └───────────┘
        │                            │
        └────────────────────────────┘
                      │
                      ▼
              [Final Integration]
```

**What this means:**
- WS 1-4 (benchmarks) have ZERO dependencies on other workstreams
- WS 5-7 (core) have ZERO dependencies on other workstreams  
- WS 8 (models) has ZERO dependencies on other workstreams
- WS 9-10 (integrations) depend on core/models INTERFACES only, not implementations
- All workstreams can start immediately after contracts are defined

---

## Part 5: Self-Critique Requirements

Every workstream agent MUST implement these self-critique mechanisms. This is non-negotiable.

### Pre-Execution Checklist

Before writing any code, the agent must answer:

```
PRE-EXECUTION CHECKLIST:

□ What contract(s) am I implementing?
  [List the specific Protocol classes from contracts/]

□ What files will I create?
  [Exact file paths]

□ What are the expensive operations?
  [List operations that might be slow or memory-intensive]

□ Is there an N+1 problem in my approach?
  [Am I doing N things that could be batched?]

□ What's the simplest approach that could work?
  [Describe it]

□ What would make me stop and reconsider?
  [List specific triggers]
```

### Batching Check

Before any loop over items, the agent must explicitly consider:

```
BATCHING CHECK:

Operation: [what I'm about to do N times]
N = [how many iterations]
Cost per iteration: [API call / DB query / model inference / file I/O]

Can this be batched? [yes / no / maybe]
If yes, batched alternative: [describe]
If no, why not: [explain]

Decision: [batch / don't batch] because [reason]
```

### Progress Checkpoints

For any task with multiple steps, report at intervals:

```
CHECKPOINT [n/total]:

Progress: [X]% complete
Time elapsed: [T]
Estimated remaining: [T']

Anomalies: [list anything unexpected]
- [anomaly 1]
- [anomaly 2]

Decision: [continue / pause to investigate / request guidance]
```

### Anomaly Triggers

These conditions MUST cause the agent to stop and report:

| Anomaly | Trigger | Required Action |
|---------|---------|-----------------|
| Too slow | Operation takes >10x expected | STOP, investigate batching |
| Too fast | Operation takes <0.1x expected | STOP, verify results aren't empty |
| Memory spike | >50% increase during operation | STOP, check for leaks |
| Uniform results | >90% identical outputs | STOP, check for bugs |
| Import errors | Any import fails | STOP, report missing dependency |
| Test failures | Any test fails unexpectedly | STOP, investigate before continuing |

### Completion Verification

Before declaring "done", the agent must:

```
COMPLETION VERIFICATION:

□ All files created: [list with paths]

□ All tests pass:
  $ pytest tests/unit/test_[workstream].py -v
  [paste output]

□ Contract compliance verified:
  $ python -c "from contracts.[x] import [Protocol]; from [impl] import [Class]"
  [paste output showing no errors]

□ Code runs without errors:
  [show a minimal working example]

□ Self-critique: What would a senior engineer criticize?
  - [criticism 1]: [my response]
  - [criticism 2]: [my response]

□ Documentation: 
  - Docstrings on all public functions? [yes/no]
  - README or usage example? [yes/no]
```

---

## Part 6: Workstream Specifications

### Workstream 1: Memory Profiler

**Implements**: `contracts.memory.MemoryProfiler`

**Purpose**: Measure actual runtime memory usage of candidate models on M2 hardware.

**Creates**:
```
benchmarks/memory/
├── __init__.py
├── profiler.py      # Core profiling logic
├── run.py           # CLI: python -m benchmarks.memory.run
└── models.py        # Model configurations to test
```

**Key Requirements**:
- Profile RSS, virtual memory, AND Metal GPU memory
- Models must be unloaded between profiles (8GB safety)
- Test at context lengths: 512, 1024, 2048, 4096
- Output JSON to results/memory/

**Agent Prompt**:
```
You are implementing Workstream 1: Memory Profiler for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/memory.py MemoryProfiler section]

FILES TO CREATE:
- benchmarks/memory/profiler.py (implements MemoryProfiler protocol)
- benchmarks/memory/run.py (CLI entrypoint)
- benchmarks/memory/models.py (model configs)
- tests/unit/test_memory_profiler.py

TECHNICAL REQUIREMENTS:
1. Use psutil for RSS/virtual memory
2. Use subprocess to call `sudo memory_pressure` or parse /proc for Metal
3. Models to profile: qwen2.5-3b-instruct-q4_k_m, smollm3-3b-q4
4. MUST unload model after each profile (critical for 8GB systems)
5. Output MemoryProfile dataclass as JSON

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]

DO NOT:
- Assume models are already downloaded (check and report if missing)
- Leave models loaded after profiling
- Skip the completion verification checklist
```

### Workstream 2: HHEM Benchmark

**Implements**: `contracts.hallucination.HallucinationEvaluator`

**Purpose**: Measure hallucination rates for email summarization task.

**Creates**:
```
benchmarks/hallucination/
├── __init__.py
├── hhem.py          # HHEM wrapper with batching
├── datasets.py      # Generate email/summary test cases
└── run.py           # CLI: python -m benchmarks.hallucination.run
```

**Key Requirements**:
- Use `vectara/hallucination_evaluation_model` from HuggingFace
- MUST implement batched evaluation (critical for overnight runs)
- Generate 100+ realistic email summarization test cases
- Test multiple prompt templates
- Target: mean HHEM >= 0.5

**Agent Prompt**:
```
You are implementing Workstream 2: HHEM Benchmark for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/hallucination.py]

FILES TO CREATE:
- benchmarks/hallucination/hhem.py (implements HallucinationEvaluator)
- benchmarks/hallucination/datasets.py (test case generation)
- benchmarks/hallucination/run.py (CLI entrypoint)
- tests/unit/test_hhem.py

TECHNICAL REQUIREMENTS:
1. Use vectara/hallucination_evaluation_model
2. MUST implement evaluate_batch() for efficiency
3. Generate realistic email content (professional, personal, newsletters)
4. Test prompt templates: basic, RAG-augmented, few-shot
5. Output HHEMBenchmarkResult as JSON

CRITICAL - BATCHING:
HHEM evaluation is the expensive operation. Before implementing:
1. Research if HHEM model supports batched inference
2. If yes, implement batching (batch size 16-32)
3. If no, document why and implement sequential with progress reporting

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 3: Template Coverage Analyzer

**Implements**: `contracts.coverage.CoverageAnalyzer`

**Purpose**: Determine what percentage of user queries can be handled by pre-written templates.

**Creates**:
```
benchmarks/coverage/
├── __init__.py
├── analyzer.py      # Semantic similarity matching
├── templates.py     # 50+ email quick-reply templates
├── datasets.py      # 1000 sample email scenarios
└── run.py           # CLI: python -m benchmarks.coverage.run
```

**Key Requirements**:
- Use sentence-transformers for semantic similarity (all-MiniLM-L6-v2 or similar)
- Create 50+ realistic email reply templates
- Generate 1000 diverse email query scenarios
- Measure coverage at 0.5, 0.7, 0.9 thresholds
- Target: ≥60% coverage at 0.7

**Agent Prompt**:
```
You are implementing Workstream 3: Template Coverage Analyzer for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/coverage.py]

FILES TO CREATE:
- benchmarks/coverage/analyzer.py (implements CoverageAnalyzer)
- benchmarks/coverage/templates.py (50+ templates)
- benchmarks/coverage/datasets.py (1000 scenarios)
- benchmarks/coverage/run.py (CLI entrypoint)
- tests/unit/test_coverage.py

TECHNICAL REQUIREMENTS:
1. Use sentence-transformers (all-MiniLM-L6-v2) - small, fast
2. Templates should cover: acknowledgments, scheduling, requests, follow-ups, 
   declines, thanks, confirmations, questions, introductions, closings
3. Scenarios should be diverse: professional, personal, urgent, casual
4. Cache embeddings (don't recompute for each query)

TEMPLATE DESIGN:
Create templates that are semantically meaningful, not just text:
- "I'll get back to you by [TIME]" covers many timing responses
- "Thanks for sending [THING], I'll review it" covers acknowledgments
Think about INTENT, not exact wording.

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 4: Latency Benchmark

**Implements**: `contracts.latency.LatencyBenchmarker`

**Purpose**: Measure cold-start and warm-start latency for generation.

**Creates**:
```
benchmarks/latency/
├── __init__.py
├── timer.py         # High-precision timing utilities
├── scenarios.py     # Test scenario definitions
└── run.py           # CLI: python -m benchmarks.latency.run
```

**Key Requirements**:
- Measure cold start (model not in memory, load from SSD)
- Measure warm start (model loaded, new context)
- Measure hot start (model loaded, cached context)
- Report p50, p95, p99 latencies
- Target: warm start < 3s for 100 tokens

**Agent Prompt**:
```
You are implementing Workstream 4: Latency Benchmark for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/latency.py]

FILES TO CREATE:
- benchmarks/latency/timer.py (high-precision timing)
- benchmarks/latency/scenarios.py (test scenarios)
- benchmarks/latency/run.py (CLI entrypoint)
- tests/unit/test_latency.py

TECHNICAL REQUIREMENTS:
1. Use time.perf_counter_ns() for high precision
2. Separate timing for: model load, prefill, generation
3. Cold start: must fully unload model first (gc.collect(), clear Metal cache)
4. Warm start: model loaded, new prompt
5. Hot start: model loaded, same prompt prefix
6. Run 10+ iterations per scenario for statistical validity

MEASUREMENT ACCURACY:
- First run is often an outlier (JIT compilation) - exclude or note
- Ensure no other heavy processes running during benchmark
- Report system state (available RAM, thermal state if possible)

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 5: Memory Controller

**Implements**: `contracts.memory.MemoryController`

**Purpose**: Adaptive memory management based on system state.

**Creates**:
```
core/memory/
├── __init__.py
├── controller.py    # Main memory controller
└── monitor.py       # System memory monitoring
```

**Key Requirements**:
- Integrate with macOS memory pressure notifications
- Implement FULL/LITE/MINIMAL mode selection
- Provide callbacks for memory pressure events
- Support priority-based resource allocation

**Agent Prompt**:
```
You are implementing Workstream 5: Memory Controller for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/memory.py MemoryController section]

FILES TO CREATE:
- core/memory/controller.py (implements MemoryController)
- core/memory/monitor.py (system monitoring)
- tests/unit/test_memory_controller.py

TECHNICAL REQUIREMENTS:
1. Use psutil for cross-platform memory monitoring
2. On macOS, also check vm_stat or memory_pressure for Metal usage
3. Mode thresholds:
   - FULL: available_mb > 8000
   - LITE: available_mb > 4000
   - MINIMAL: available_mb <= 4000
4. Pressure levels:
   - green: used < 70%
   - yellow: used 70-85%
   - red: used 85-95%
   - critical: used > 95%
5. Callbacks must be thread-safe

DESIGN DECISION:
This controller doesn't load/unload models directly. It provides:
- Current state information
- Mode recommendations
- Callbacks when pressure changes
The Generator (WS8) uses this info to decide when to load/unload.

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 6: Graceful Degradation

**Implements**: `contracts.health.DegradationController`

**Purpose**: Handle failures gracefully without crashing.

**Creates**:
```
core/health/
├── __init__.py
├── degradation.py   # Degradation controller
└── circuit.py       # Circuit breaker pattern
```

**Key Requirements**:
- Implement circuit breaker pattern
- Support feature registration with fallback behaviors
- Track failure counts and auto-recovery
- Log all degradation events

**Agent Prompt**:
```
You are implementing Workstream 6: Graceful Degradation for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/health.py DegradationController section]

FILES TO CREATE:
- core/health/degradation.py (implements DegradationController)
- core/health/circuit.py (circuit breaker)
- tests/unit/test_degradation.py

TECHNICAL REQUIREMENTS:
1. Circuit breaker states: CLOSED (healthy) -> OPEN (failing) -> HALF_OPEN (testing)
2. Configurable failure threshold (default: 3 failures)
3. Configurable recovery timeout (default: 60 seconds)
4. Thread-safe state management
5. Structured logging for all state transitions

EXAMPLE USAGE:
```python
controller = DegradationController()
controller.register_feature(DegradationPolicy(
    feature_name="imessage",
    health_check=lambda: check_chat_db_access(),
    degraded_behavior=lambda: return_cached_messages(),
    fallback_behavior=lambda: return_empty_with_error(),
    recovery_check=lambda: check_chat_db_access(),
    max_failures=3
))

# Later, in iMessage code:
messages = controller.execute("imessage", get_messages, chat_id="...")
```

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 7: Permission & Schema Health

**Implements**: `contracts.health.PermissionMonitor`, `contracts.health.SchemaDetector`

**Purpose**: Monitor TCC permissions and detect chat.db schema changes.

**Creates**:
```
core/health/
├── permissions.py   # TCC permission monitoring
└── schema.py        # chat.db schema detection
```

**Key Requirements**:
- Check Full Disk Access permission (required for chat.db)
- Detect chat.db schema version
- Provide user-friendly fix instructions
- Handle unknown schema versions gracefully

**Agent Prompt**:
```
You are implementing Workstream 7: Permission & Schema Health for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/health.py PermissionMonitor and SchemaDetector sections]

FILES TO CREATE:
- core/health/permissions.py (implements PermissionMonitor)
- core/health/schema.py (implements SchemaDetector)
- tests/unit/test_permissions.py
- tests/unit/test_schema.py

TECHNICAL REQUIREMENTS:

For Permissions:
1. Check FDA by attempting to read ~/Library/Messages/chat.db
2. Do NOT actually parse the DB, just check if open() succeeds
3. Provide specific System Preferences path for fix instructions
4. Cache permission status (don't check every call)

For Schema:
1. Known schemas: macOS 12 (Monterey), 13 (Ventura), 14 (Sonoma), 15 (Sequoia)
2. Key tables to check: message, handle, chat, chat_message_join
3. Key columns that changed: attributedBody format, reply threading
4. If unknown schema, still try to extract basic messages

SCHEMA DETECTION APPROACH:
1. Query sqlite_master for table list
2. Query PRAGMA table_info() for each key table
3. Compare against known schema fingerprints
4. Return compatibility assessment

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 8: Model Loader & Generator

**Implements**: `contracts.models.Generator`

**Purpose**: Load MLX models and generate responses with RAG/few-shot.

**Creates**:
```
models/
├── __init__.py
├── loader.py        # MLX model loading
├── generator.py     # Generation with RAG/few-shot
└── templates.py     # Template-based fallback
```

**Key Requirements**:
- Load/unload MLX models on demand
- Inject RAG context into prompts
- Format few-shot examples
- Fall back to templates when appropriate

**Agent Prompt**:
```
You are implementing Workstream 8: Model Loader & Generator for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/models.py]

FILES TO CREATE:
- models/loader.py (MLX model loading)
- models/generator.py (implements Generator)
- models/templates.py (template fallback)
- tests/unit/test_generator.py

TECHNICAL REQUIREMENTS:
1. Use mlx-lm for model loading
2. Support Qwen 2.5-3B-Instruct Q4_K_M format
3. Implement proper unloading (gc.collect, mlx.core.metal.clear_cache)
4. RAG context injection: prepend context documents to prompt
5. Few-shot formatting: structure examples in prompt

GENERATION FLOW:
1. Check if request matches a template (high confidence)
   - If yes, return template response (fast, no model needed)
2. If model not loaded, check MemoryController
   - If can_load_model(), load it
   - If not, return template or error
3. Generate with RAG context and few-shot examples
4. Return response with metadata

TEMPLATE INTEGRATION:
The Generator should use the same templates from WS3 (coverage analyzer).
Import from benchmarks.coverage.templates or create shared location.

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

### Workstream 10: iMessage Integration

**Implements**: `contracts.imessage.iMessageReader`

**Purpose**: Read iMessage history from chat.db.

**Creates**:
```
integrations/imessage/
├── __init__.py
├── reader.py        # chat.db reader
├── queries.py       # SQL queries for different schemas
└── parser.py        # Message normalization
```

**Key Requirements**:
- Read-only access to chat.db
- Handle different schema versions
- Parse attributedBody (NSKeyedArchive format)
- Graceful handling of permission issues

**Agent Prompt**:
```
You are implementing Workstream 10: iMessage Integration for JARVIS.

CONTRACTS TO IMPLEMENT:
[paste contracts/imessage.py]

FILES TO CREATE:
- integrations/imessage/reader.py (implements iMessageReader)
- integrations/imessage/queries.py (SQL for different schemas)
- integrations/imessage/parser.py (message parsing)
- tests/unit/test_imessage.py

TECHNICAL REQUIREMENTS:
1. Open chat.db in read-only mode: sqlite3.connect("file:...?mode=ro", uri=True)
2. Use SchemaDetector (WS7) to get appropriate queries
3. Parse attributedBody using plistlib or biplist
4. Map handle_id to phone/email via handle table
5. Handle group chats (chat_handle_join table)

CRITICAL - DATABASE SAFETY:
- NEVER write to chat.db
- Use read-only mode in connection string
- Handle SQLITE_BUSY gracefully (iMessage may be writing)
- Don't hold long-running transactions

ATTRIBUTEDBODY PARSING:
This is the tricky part. attributedBody contains NSKeyedArchive data.
Options:
1. Use Foundation.NSKeyedUnarchiver (requires pyobjc)
2. Parse plist manually and extract __NSAttributedString
3. Fall back to 'text' column if attributedBody fails

SELF-CRITIQUE REQUIREMENTS:
[paste Part 5 self-critique section]
```

---

## Part 7: Overnight Evaluation Workflow

### The Script

```bash
#!/bin/bash
# scripts/overnight_eval.sh
# Run all benchmarks sequentially (8GB safe)

set -e  # Exit on error

RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "🌙 JARVIS Overnight Evaluation Suite"
echo "======================================"
echo "Started: $(date)"
echo "Results: $RESULTS_DIR"
echo ""

# Pre-flight checks
echo "📋 Pre-flight checks..."
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
python -c "import psutil; m = psutil.virtual_memory(); print(f'Available RAM: {m.available / 1e9:.1f}GB')"
echo ""

# Memory profiling (models loaded/unloaded one at a time)
echo "📊 [1/4] Memory Profiling..."
python -m benchmarks.memory.run \
    --output "$RESULTS_DIR/memory.json" \
    2>&1 | tee "$RESULTS_DIR/memory.log"
echo ""

# HHEM benchmark (batched for efficiency)
echo "🔍 [2/4] HHEM Hallucination Benchmark..."
python -m benchmarks.hallucination.run \
    --output "$RESULTS_DIR/hhem.json" \
    2>&1 | tee "$RESULTS_DIR/hhem.log"
echo ""

# Template coverage (lightweight, embedding-based)
echo "📋 [3/4] Template Coverage Analysis..."
python -m benchmarks.coverage.run \
    --output "$RESULTS_DIR/coverage.json" \
    2>&1 | tee "$RESULTS_DIR/coverage.log"
echo ""

# Latency benchmarks (cold/warm/hot)
echo "⏱️ [4/4] Latency Benchmarks..."
python -m benchmarks.latency.run \
    --output "$RESULTS_DIR/latency.json" \
    2>&1 | tee "$RESULTS_DIR/latency.log"
echo ""

# Generate report
echo "📈 Generating report..."
python scripts/generate_report.py \
    --results-dir "$RESULTS_DIR" \
    --output docs/BENCHMARKS.md
echo ""

echo "======================================"
echo "✅ Completed: $(date)"
echo "Results: $RESULTS_DIR"
echo "Report: docs/BENCHMARKS.md"
```

### Morning Review Process

When you wake up, review results in this order:

1. **Check for errors**: `grep -r "ERROR\|FAIL\|Exception" results/latest/*.log`

2. **Review gates**:
   ```bash
   python scripts/check_gates.py --results-dir results/latest
   # Outputs: G1 PASS, G2 CONDITIONAL, G3 PASS, G4 PASS, G5 PASS
   ```

3. **If any gate FAILED**: Stop and reassess before continuing development

4. **If all gates PASS/CONDITIONAL**: Proceed with day's workstream assignments

---

## Part 8: Decision Gates

### Gate Definitions

| Gate | Metric | PASS | CONDITIONAL | FAIL |
|------|--------|------|-------------|------|
| G1: Coverage | coverage@0.7 | ≥60% | 40-60% | <40% |
| G2: Memory | Total model stack | <5.5GB | 5.5-6.5GB | >6.5GB |
| G3: HHEM | Mean score | ≥0.5 | 0.4-0.5 | <0.4 |
| G4: Warm Latency | p95 | <3s | 3-5s | >5s |
| G5: Cold Latency | p95 | <15s | 15-20s | >20s |

### Decision Matrix

| Scenario | Action |
|----------|--------|
| All PASS | Full speed ahead |
| G2 CONDITIONAL | Drop 8GB support, target 16GB minimum |
| G3 CONDITIONAL | Template-first approach, generation as fallback |
| G4/G5 CONDITIONAL | Implement cloud fallback for time-sensitive requests |
| Any FAIL | Stop development, reassess project viability |
| 2+ FAIL | Consider project cancellation |

### Gate Checking Script

```python
# scripts/check_gates.py
import json
import sys
from pathlib import Path

def check_gates(results_dir: Path) -> dict:
    """Check all gates and return status."""
    
    gates = {}
    
    # G1: Coverage
    coverage = json.loads((results_dir / "coverage.json").read_text())
    cov_70 = coverage["coverage_at_70"]
    if cov_70 >= 0.60:
        gates["G1"] = ("PASS", f"coverage@0.7 = {cov_70:.1%}")
    elif cov_70 >= 0.40:
        gates["G1"] = ("CONDITIONAL", f"coverage@0.7 = {cov_70:.1%}")
    else:
        gates["G1"] = ("FAIL", f"coverage@0.7 = {cov_70:.1%}")
    
    # G2: Memory
    memory = json.loads((results_dir / "memory.json").read_text())
    # Sum up the model stack (LLM + embeddings)
    total_mb = sum(p["rss_mb"] for p in memory["profiles"])
    if total_mb < 5500:
        gates["G2"] = ("PASS", f"total = {total_mb:.0f}MB")
    elif total_mb < 6500:
        gates["G2"] = ("CONDITIONAL", f"total = {total_mb:.0f}MB")
    else:
        gates["G2"] = ("FAIL", f"total = {total_mb:.0f}MB")
    
    # G3: HHEM
    hhem = json.loads((results_dir / "hhem.json").read_text())
    mean_score = hhem["mean_score"]
    if mean_score >= 0.5:
        gates["G3"] = ("PASS", f"mean HHEM = {mean_score:.3f}")
    elif mean_score >= 0.4:
        gates["G3"] = ("CONDITIONAL", f"mean HHEM = {mean_score:.3f}")
    else:
        gates["G3"] = ("FAIL", f"mean HHEM = {mean_score:.3f}")
    
    # G4: Warm Latency
    latency = json.loads((results_dir / "latency.json").read_text())
    warm_p95 = next(r["p95_ms"] for r in latency["results"] if r["scenario"] == "warm")
    if warm_p95 < 3000:
        gates["G4"] = ("PASS", f"warm p95 = {warm_p95:.0f}ms")
    elif warm_p95 < 5000:
        gates["G4"] = ("CONDITIONAL", f"warm p95 = {warm_p95:.0f}ms")
    else:
        gates["G4"] = ("FAIL", f"warm p95 = {warm_p95:.0f}ms")
    
    # G5: Cold Latency
    cold_p95 = next(r["p95_ms"] for r in latency["results"] if r["scenario"] == "cold")
    if cold_p95 < 15000:
        gates["G5"] = ("PASS", f"cold p95 = {cold_p95:.0f}ms")
    elif cold_p95 < 20000:
        gates["G5"] = ("CONDITIONAL", f"cold p95 = {cold_p95:.0f}ms")
    else:
        gates["G5"] = ("FAIL", f"cold p95 = {cold_p95:.0f}ms")
    
    return gates

if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/latest")
    gates = check_gates(results_dir)
    
    print("\n" + "="*50)
    print("GATE STATUS")
    print("="*50)
    
    for gate, (status, detail) in gates.items():
        emoji = {"PASS": "✅", "CONDITIONAL": "⚠️", "FAIL": "❌"}[status]
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
```

---

## Part 9: Portfolio Artifacts

### README.md Template

```markdown
# 🤖 JARVIS - Local AI Assistant for macOS

> Your private AI assistant that runs entirely on your Mac.
> No cloud dependency. No data collection. Just intelligence.

![Status](https://img.shields.io/badge/status-active-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![macOS](https://img.shields.io/badge/macOS-14+-black)
![License](https://img.shields.io/badge/license-MIT-purple)

## 🎯 What is JARVIS?

JARVIS is a local-first AI assistant that helps you manage email and messages without sending your data to the cloud. It runs a 3B parameter language model directly on your Mac's Apple Silicon chip.

**Key Features:**
- 💬 iMessage conversation context
- 🧠 Runs 100% locally on Apple Silicon
- 🔒 Your data never leaves your device
- ⚡ <3 second response time (warm start)

## 📊 Benchmarks

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Memory (8GB mode) | <6GB | X.XGB | ✅ |
| Warm start latency | <3s | X.Xs | ✅ |
| Hallucination rate | HHEM ≥0.5 | 0.XX | ✅ |
| Template coverage | ≥60% | XX% | ✅ |

[Full benchmark methodology →](BENCHMARKS.md)

## 🏗️ Architecture

[Architecture diagram]

[Full architecture deep-dive →](ARCHITECTURE.md)

## 🚀 Quick Start

[Installation instructions]

## 🧪 Development

[Development setup instructions]

## 📜 License

MIT License - see [LICENSE](LICENSE)
```

### LinkedIn Post Templates

Store in `docs/blog_posts/`:

**Post 1: Project Announcement**
```
🚀 Just shipped JARVIS - a local AI assistant that runs entirely on your Mac.

The challenge: Run a 3B parameter LLM on an 8GB MacBook.

My approach: Validation-first engineering.

Before writing a single line of production code, I built a benchmark suite that proved:
✅ Memory footprint fits in budget
✅ Hallucination rate meets quality bar  
✅ Response latency is acceptable
✅ Template coverage handles common cases

Only after all gates passed did I write the actual features.

Result: A working assistant that doesn't melt your laptop.

Key learnings:
1. Model file size ≠ runtime memory (it's 2-3x more)
2. "Graceful degradation" isn't optional, it's the product
3. Benchmark everything before you build anything

Check it out: [GitHub link]

#MachineLearning #LocalAI #Engineering #MLOps
```

**Post 2: Technical Deep Dive**
```
🔬 How I measure AI hallucinations in a local assistant:

When building JARVIS, I needed to ensure the AI wasn't making things up about your emails.

The metric: HHEM (Hughes Hallucination Evaluation Model)
- Score 1.0 = Perfectly grounded in source text
- Score 0.5 = Industry threshold for "acceptable"
- Score 0.0 = Complete hallucination

My initial results: 0.02-0.05 (catastrophic!)

The fix:
1. RAG augmentation - inject actual email text into the prompt
2. Few-shot examples - show the model what good looks like
3. Template fallback - for common cases, don't generate at all

Final score: 0.XX (exceeds threshold!)

The benchmark suite is open source: [link]

#MLOps #LLM #AIEngineering #Benchmarking
```

---

## Part 10: Execution Checklist

### Phase 0: Repository Setup

```
□ Create fresh git repository
□ Set up directory structure per Part 2
□ Create all contract files per Part 3
□ Create pyproject.toml with dependencies
□ Create scripts/overnight_eval.sh
□ Create scripts/check_gates.py
□ Audit existing projects for reusable code
□ Create extraction manifest
□ Port identified code with proper attribution
□ Verify basic structure: `python -c "import contracts"`
□ Push initial commit
```

### Phase 1: Benchmark Workstreams (WS 1-4)

```
□ Spawn WS1 agent (Memory Profiler)
□ Spawn WS2 agent (HHEM Benchmark)
□ Spawn WS3 agent (Template Coverage)
□ Spawn WS4 agent (Latency Benchmark)
□ Review and merge each workstream
□ Run overnight evaluation suite
□ Check gates - proceed only if no FAILs
```

### Phase 2: Core Workstreams (WS 5-7)

```
□ Spawn WS5 agent (Memory Controller)
□ Spawn WS6 agent (Graceful Degradation)
□ Spawn WS7 agent (Permission & Schema Health)
□ Review and merge each workstream
□ Integration test: core modules work together
□ Run overnight eval (regression check)
```

### Phase 3: Model & Integration Workstreams (WS 8-10)

```
□ Spawn WS8 agent (Model Loader & Generator)
□ Spawn WS10 agent (iMessage Integration)
□ Review and merge each workstream
□ Integration test: end-to-end flow works
□ Run overnight eval (full regression)
```

### Phase 4: Polish & Portfolio

```
□ Final benchmark run with all components
□ Update BENCHMARKS.md with final numbers
□ Write README.md with demo GIF
□ Write ARCHITECTURE.md
□ Create Architecture Decision Records
□ Write LinkedIn posts
□ Record demo video (optional)
□ Publish to GitHub
□ Post to LinkedIn
```

---

## Appendix A: pyproject.toml

```toml
[project]
name = "jarvis-ai-assistant"
version = "1.0.0"
description = "Local-first AI assistant for macOS with iMessage integration"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["ai", "assistant", "macos", "local", "privacy", "mlx"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: MacOS X",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Communications :: Email",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "mlx>=0.5.0",
    "mlx-lm>=0.5.0",
    "sentence-transformers>=2.2.0",
    "google-api-python-client>=2.100.0",
    "google-auth-oauthlib>=1.1.0",
    "psutil>=5.9.0",
    "pydantic>=2.5.0",
    "rich>=13.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]
benchmarks = [
    "transformers>=4.36.0",
    "torch>=2.1.0",
    "matplotlib>=3.8.0",
    "pandas>=2.1.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/jarvis-ai-assistant"
Documentation = "https://github.com/yourusername/jarvis-ai-assistant#readme"
Repository = "https://github.com/yourusername/jarvis-ai-assistant.git"
Issues = "https://github.com/yourusername/jarvis-ai-assistant/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --cov=. --cov-report=term-missing"
```

---

## Appendix B: Quick Reference

### Starting a Workstream Agent

1. Open new Claude Code Web session
2. Upload this document
3. Say: "I need you to implement Workstream [N]: [Name]. Here are the contracts: [paste relevant contracts/]"
4. Monitor progress, review code when done

### Checking Progress

```bash
# See what's implemented
python -c "from contracts import *; print('Contracts OK')"

# Run tests for a workstream
pytest tests/unit/test_[workstream].py -v

# Run all tests
pytest tests/ -v

# Check type hints
mypy core/ models/ integrations/ benchmarks/
```

### Overnight Eval

```bash
# Start overnight run
./scripts/overnight_eval.sh

# Check results in morning
python scripts/check_gates.py results/latest
```
