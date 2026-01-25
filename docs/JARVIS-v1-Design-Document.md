# JARVIS v1: Design Document
## Local-First AI Assistant for macOS

**Document Type**: Architecture & Strategy  
**Version**: 2.0  
**Date**: January 25, 2026  
**Status**: YELLOW/CONDITIONAL-GO (pending validation gates)

---

## 1. Executive Summary

JARVIS is a local-first AI assistant for macOS that provides intelligent email and message management without sending user data to the cloud. The system runs a 3B parameter language model directly on Apple Silicon, integrating with Gmail and iMessage to provide contextual assistance.

This document describes the architectural decisions, risk mitigations, and validation strategy for JARVIS v1. It supersedes the original v0 design, which was declared non-viable due to five critical blockers identified during adversarial review.

### 1.1 Why This Project Exists

The core value proposition is privacy-preserving AI assistance. Users increasingly want AI capabilities but are uncomfortable with their personal communications being processed by cloud services. JARVIS addresses this by running entirely on the user's device, ensuring that emails and messages never leave the local machine.

### 1.2 What Changed from v0

The original design made several fatal assumptions that real-world testing invalidated. The revised design adopts a validation-first approach: we prove the system can work before building it. The table below summarizes the key changes.

| Aspect | v0 Assumption | v1 Reality |
|--------|---------------|------------|
| Memory budget | 6.5GB total, fits in 8GB | 11.8GB realistic, requires 16GB or degraded mode |
| HHEM scores | 0.02-0.05 is acceptable | Those scores mean 95-98% hallucination; need ≥0.5 |
| Model loading | Always warm, instant | Cold start takes 10-18 seconds |
| Fine-tuning | Improves quality | Research shows it increases hallucinations |
| Permissions | "Seamless" access | TCC gates require explicit user action and can break |

### 1.3 Document Structure

This design document covers the "what" and "why" of JARVIS. A companion Development Guide covers the "how" of implementation, including workstream definitions, agent prompts, and execution checklists. The two documents should be read together but serve different purposes: this document is for architectural understanding and interview discussions, while the Development Guide is for day-to-day execution.

---

## 2. Problem Statement

### 2.1 User Need

Knowledge workers spend significant time managing email and messages. They need quick, contextual responses but don't have time to craft each one carefully. Existing solutions either require cloud processing (privacy concern) or are too simplistic (template-only, no intelligence).

### 2.2 Technical Challenge

Running a capable language model on consumer hardware within acceptable latency and memory constraints. An 8GB MacBook Air represents the minimum viable target, but the math is unforgiving: macOS itself uses 4GB, typical user applications another 2-3GB, leaving only 1-2GB for our entire system. This is insufficient for naive model loading.

### 2.3 Success Criteria

The system succeeds if it can provide useful assistance for common email and message scenarios while meeting these constraints.

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Response quality | HHEM ≥ 0.5 | Hallucination evaluation benchmark |
| Response latency | < 3 seconds (warm) | End-to-end timing |
| Memory footprint | < 6GB total | RSS measurement during operation |
| Common case coverage | ≥ 60% | Template matching against real queries |
| User data privacy | 100% local | Architecture audit (no network calls for user data) |

---

## 3. Architecture Overview

### 3.1 System Context

JARVIS operates as a local application on macOS, integrating with two data sources (Gmail API and iMessage chat.db) and providing responses through a menu bar interface.

```
┌─────────────────────────────────────────────────────────────────┐
│                         macOS System                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      JARVIS                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │
│  │  │   Gmail     │  │  iMessage   │  │   Model     │       │   │
│  │  │ Integration │  │ Integration │  │  Generator  │       │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │   │
│  │         │                │                │               │   │
│  │         ▼                ▼                ▼               │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │              Core Services                       │     │   │
│  │  │  Memory Controller | Health Monitor | Config    │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Gmail API   │  │   chat.db    │  │  MLX Model   │          │
│  │  (Network)   │  │  (Local DB)  │  │  (Local)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

When a user requests assistance, the system follows this flow.

First, the request is classified to determine if it matches a known template. Template matching is fast (milliseconds) and requires no model loading, making it the preferred path for common cases.

Second, if no template matches with sufficient confidence, the system checks memory availability. If the model can be loaded (or is already loaded), generation proceeds with RAG context injection. If memory is constrained, the system falls back to a degraded response or cloud API.

Third, for generation requests, relevant context is retrieved from Gmail or iMessage and injected into the prompt. Few-shot examples guide the model toward appropriate response style.

Fourth, the response is validated. If HHEM scoring indicates hallucination risk above threshold, the response is rejected and either regenerated or replaced with a safer template response.

### 3.3 Key Components

**Memory Controller**: Monitors system memory and determines operating mode (Full, Lite, or Minimal). Provides callbacks for memory pressure events so components can respond appropriately.

**Degradation Controller**: Implements circuit breaker pattern for graceful failure handling. Each feature registers health checks and fallback behaviors; the controller automatically degrades when failures occur and recovers when health is restored.

**Permission Monitor**: Tracks TCC (Transparency, Consent, and Control) permissions required for iMessage access. Provides user-friendly guidance when permissions are missing.

**Schema Detector**: Identifies chat.db schema version across macOS releases. Apple changes this schema periodically; the detector ensures queries work correctly or falls back gracefully.

**Template Matcher**: Uses semantic similarity to match user requests against pre-defined response templates. High-confidence matches bypass model generation entirely.

**Model Generator**: Loads and runs the MLX model for text generation. Handles RAG context injection, few-shot prompting, and memory-aware loading/unloading.

**Gmail Client**: OAuth-authenticated wrapper for Gmail API. Provides search, retrieval, and parsing of email content.

**iMessage Reader**: Read-only SQLite access to chat.db. Handles schema variations, attributedBody parsing, and contact resolution.

---

## 4. Critical Blocker Resolution

The v0 design failed adversarial review due to five critical blockers. This section describes each blocker and its resolution strategy.

### 4.1 Blocker 1: Memory Budget

**The Problem**: The original design estimated 6.5GB total memory usage, claiming it would fit comfortably in 8GB. Realistic measurement showed 11.8GB actual usage, making the system impossible to run on the target hardware.

The discrepancy arose from confusing model file size with runtime memory footprint. A 2GB model file requires 2.5-3.2GB of RAM when loaded, plus Metal GPU context (300-800MB), plus framework overhead (300-500MB). The original estimate ignored all of this.

**Resolution Strategy**: The system now supports three operating modes based on available memory.

Full Mode (16GB+ systems) loads all components concurrently with 4K context windows. This is the optimal experience but requires more RAM than the original target.

Lite Mode (8-16GB systems) loads components sequentially, never running the LLM and embedding model simultaneously. Context windows are reduced to 2K. A cloud API fallback handles requests when local generation isn't possible.

Minimal Mode (under 8GB) relies entirely on template responses with cloud fallback for any generation needs. The local model is never loaded.

**Validation Gate**: G2 measures actual memory usage. If the model stack exceeds 6.5GB, 8GB support is dropped entirely.

### 4.2 Blocker 2: HHEM Metric Misunderstanding

**The Problem**: The original design reported HHEM scores of 0.02-0.05 and treated these as acceptable. This reflected a catastrophic misunderstanding of the metric. HHEM scores range from 0 (complete hallucination) to 1 (fully grounded). A score of 0.02-0.05 means the system hallucinates 95-98% of the time.

**Resolution Strategy**: The correct threshold for acceptable quality is HHEM ≥ 0.5, per Vectara documentation. Achieving this requires several mitigations.

RAG augmentation injects the actual source text (email or message content) directly into the prompt, giving the model factual grounding rather than requiring it to recall or fabricate information.

Few-shot prompting provides examples of good source-to-summary transformations, teaching the model the expected behavior.

Rejection sampling regenerates responses that score below threshold, or falls back to template responses when generation repeatedly fails quality checks.

Template-first architecture handles the majority of common cases without generation at all, eliminating hallucination risk for those scenarios.

**Validation Gate**: G3 measures mean HHEM score across a benchmark dataset. If the score falls below 0.4, the summarization feature is dropped and the system becomes template-only.

### 4.3 Blocker 3: Unknown Template Coverage

**The Problem**: The design assumed templates could handle most cases, but had no empirical data on what percentage of real user queries actually match templates.

**Resolution Strategy**: Build a coverage measurement system before committing to the template-first architecture. This involves creating a diverse set of response templates (50+), generating a representative dataset of user queries (1000+), measuring semantic similarity between queries and templates, and determining coverage at various confidence thresholds.

**Validation Gate**: G1 measures coverage at 70% confidence threshold. If coverage falls below 40%, the template-first architecture is not viable and the project must pivot to generation-heavy or be cancelled.

### 4.4 Blocker 4: Missing Error Handling

**The Problem**: The original design assumed the "happy path" where permissions are granted, schemas are known, memory is available, and operations succeed. Real systems face constant failures.

**Resolution Strategy**: Build error handling infrastructure before feature development. This includes TCC permission monitoring with health checks and user guidance, SQLite WAL-aware reading that handles database locks gracefully, memory pressure detection with automatic degradation, schema version detection with fallback queries, and circuit breaker patterns to prevent cascade failures.

**Validation Approach**: Integration tests simulate failure scenarios (permission revocation, memory pressure, schema changes) and verify graceful degradation.

### 4.5 Blocker 5: Cold-Start Latency

**The Problem**: The original design targeted less than 2 second response time but ignored model loading. Cold start (model not in memory) realistically takes 10-18 seconds: 5-10 seconds for SSD read, 2-5 seconds for Metal graph compilation, plus inference time.

**Resolution Strategy**: Accept cold-start penalty but optimize for warm-start scenarios. Model preloading triggers when the user opens Messages or Mail applications, speculatively loading the model before it's needed. A "thinking" UI indicator sets appropriate expectations during cold starts. Warm-start optimization ensures that once loaded, responses are fast. Cloud fallback provides instant responses for time-sensitive requests when cold-start delay is unacceptable.

**Validation Gates**: G4 measures warm-start latency (target: under 3 seconds). G5 measures cold-start latency (target: under 15 seconds). If warm-start exceeds 5 seconds, cloud fallback becomes the primary path.

---

## 5. Technology Decisions

### 5.1 Model Selection: Qwen 2.5-3B vs SmolLM3-3B

The base model must balance capability against memory footprint. Two candidates are under evaluation.

Qwen 2.5-3B-Instruct has good MLX support and reasonable quality on instruction-following tasks. The Q4_K_M quantization brings it to approximately 2.8-3.2GB memory. The Qwen license requires verification for commercial use.

SmolLM3-3B offers Apache 2.0 licensing (clear commercial use) and is designed for efficiency. MLX support and actual memory footprint require measurement.

**Decision**: Benchmark both on RAGTruth dataset and measure actual memory. Select whichever achieves HHEM ≥ 0.5 with lower memory footprint. Qwen is the default unless SmolLM3 proves clearly superior.

### 5.2 Embedding Model: Lightweight Priority

The original design specified GLiNER for entity recognition, but realistic memory measurement shows 800MB-1.2GB footprint. This is too heavy for 8GB systems.

**Decision**: Use all-MiniLM-L6-v2 (approximately 100MB) for semantic similarity in template matching. This is sufficient for matching queries to templates and consumes minimal memory. If richer entity extraction is needed, evaluate NuNER Zero as a lighter alternative to GLiNER.

### 5.3 Inference Framework: MLX

MLX is Apple's framework optimized for Apple Silicon. It provides better memory control than alternatives like Ollama, which wraps models in a server process with less predictable memory behavior.

**Decision**: Use MLX via mlx-lm library. Accept the additional complexity in exchange for precise memory control, which is critical for 8GB systems.

### 5.4 Fine-Tuning: Avoided

The original design planned LoRA fine-tuning for style matching. Research by Gekhman et al. (EMNLP 2024) demonstrates that fine-tuning on new knowledge exacerbates hallucinations rather than reducing them.

**Decision**: No fine-tuning for knowledge injection. Use RAG for factual grounding and few-shot prompting for style matching. These approaches are cheaper, reversible, and don't risk increasing hallucination rates.

### 5.5 Gmail Integration: API over Local Indexing

The original design proposed full local indexing of Gmail. This creates significant complexity (sync logic, storage requirements, freshness issues) for marginal benefit.

**Decision**: Use Gmail API directly with a working set approach. Fetch recent emails (30 days) on demand. Leverage Gmail's server-side search rather than building local search infrastructure. This trades some offline capability for dramatically reduced complexity.

### 5.6 iMessage Integration: Read-Only with Guardrails

iMessage access requires Full Disk Access permission and direct SQLite queries against chat.db. This is fragile: Apple changes the schema between macOS versions, and the database is actively written by the Messages process.

**Decision**: Read-only access with defensive coding. Open database in read-only mode to prevent any possibility of corruption. Detect schema version at startup and use appropriate queries. Handle SQLITE_BUSY gracefully when Messages is writing. Fall back to basic text extraction if schema is unrecognized.

---

## 6. Validation-First Methodology

### 6.1 Philosophy

Traditional development writes code first, then tests whether it works. JARVIS adopts validation-first development: prove the approach works before writing production code. This is especially important given the narrow margins for error on memory and quality constraints.

The methodology has three principles.

**Measure before building**: Create benchmarks for memory, quality, coverage, and latency before implementing features. Run benchmarks to validate assumptions before committing engineering effort.

**Gate decisions on data**: Each phase has explicit go/no-go criteria based on benchmark results. Conditional results trigger scope adjustments. Failures trigger project reassessment.

**Self-critique at every level**: Agents implementing the system must validate their own work against explicit checklists. Anomalies trigger investigation, not continuation.

### 6.2 Self-Critique Integration

The overnight evaluation failure described in the project retrospective highlighted a critical gap: agents in execution mode don't question their approach's efficiency. The system now requires explicit self-critique at multiple levels.

**Pre-execution checks** require identifying expensive operations, considering batching opportunities, and documenting the approach before starting work.

**Progress checkpoints** at regular intervals compare actual versus expected progress. Significant deviations trigger investigation rather than blind continuation.

**Anomaly triggers** define specific conditions (operations taking 10x expected time, uniform results suggesting bugs, memory spikes) that require stopping and reporting rather than proceeding.

**Completion verification** requires demonstrating that code actually works (tests pass, imports succeed, minimal examples run) before declaring work complete.

The Development Guide specifies these mechanisms in detail for each workstream.

### 6.3 Validation Gates

Five gates determine project viability. All gates are evaluated after the benchmark phase completes.

| Gate | Metric | Pass | Conditional | Fail |
|------|--------|------|-------------|------|
| G1 | Template coverage at 0.7 threshold | ≥ 60% | 40-60% | < 40% |
| G2 | Total model stack memory | < 5.5GB | 5.5-6.5GB | > 6.5GB |
| G3 | Mean HHEM score | ≥ 0.5 | 0.4-0.5 | < 0.4 |
| G4 | Warm-start latency (p95) | < 3s | 3-5s | > 5s |
| G5 | Cold-start latency (p95) | < 15s | 15-20s | > 15s |

**Pass** means the constraint is met and development proceeds normally.

**Conditional** means the constraint is partially met and scope adjustments are required. G1 conditional triggers hybrid template+generation. G2 conditional drops 8GB support. G3 conditional triggers template-first with minimal generation. G4/G5 conditional triggers cloud fallback for latency-sensitive requests.

**Fail** means the constraint cannot be met and the project viability is in question. Any single failure requires stopping and reassessing. Two or more failures trigger consideration of project cancellation.

---

## 7. Risk Register

### 7.1 Critical Risks

**R1: Memory Budget Impossible for 8GB** (Likelihood: HIGH, Impact: CRITICAL)

The math may simply not work for 8GB systems even with aggressive optimization. Primary mitigation is accepting 16GB as the real minimum and positioning 8GB as a degraded cloud-hybrid mode. Secondary mitigation is pure cloud architecture with local triage only. The benchmark phase memory profiling determines which path to take.

**R2: HHEM Never Reaches 0.5** (Likelihood: MEDIUM, Impact: HIGH)

Small models may be fundamentally incapable of sufficient faithfulness for summarization tasks. Primary mitigation is RAG augmentation with explicit source citation. Secondary mitigation is rejection sampling with regeneration. Tertiary mitigation is dropping summarization entirely and keeping only template responses.

**R8: Fine-Tuning Increases Hallucinations** (Likelihood: HIGH, Impact: HIGH)

Per Gekhman et al., this is likely if fine-tuning is attempted. Mitigation is simply not fine-tuning; the decision has already been made to use RAG and few-shot instead.

### 7.2 High Risks

**R3: Template Coverage Too Low** (Likelihood: MEDIUM, Impact: HIGH)

If fewer than 40% of real queries match templates, the template-first architecture collapses. Mitigation options include heavy investment in generation quality, accepting cloud API costs, or project cancellation if the value proposition is lost.

**R4: Apple Changes chat.db Schema** (Likelihood: HIGH, Impact: MEDIUM)

Apple changes this schema periodically without notice. Mitigation includes schema version detection at startup, fallback to basic text extraction for unknown schemas, and monitoring against macOS betas.

**R5: TCC Permissions Blocked by Update** (Likelihood: MEDIUM, Impact: HIGH)

macOS updates can reset or change permission requirements. Mitigation includes permission health checks at startup, clear user notification when permissions are lost, and graceful degradation of iMessage features.

**R6: Cold-Start Latency Unacceptable** (Likelihood: HIGH, Impact: MEDIUM)

Users may not tolerate 10-15 second cold starts. Mitigation includes speculative preloading, cloud fallback for cold-start scenarios, and UI feedback that sets appropriate expectations.

**R9: GLiNER Too Memory-Heavy** (Likelihood: HIGH, Impact: MEDIUM)

At 800MB-1.2GB, GLiNER may be too expensive for the memory budget. Mitigation is using lighter alternatives (all-MiniLM-L6-v2 for similarity, NuNER Zero for NER if needed).

**R10: Qwen License Issues** (Likelihood: MEDIUM, Impact: HIGH)

Qwen license terms for commercial use require verification. Mitigation is having SmolLM3 (Apache 2.0) as a backup.

### 7.3 Moderate Risks

**R7: Thermal Throttling Degrades UX** (Likelihood: HIGH, Impact: MEDIUM)

Sustained inference on a MacBook Air causes thermal throttling. Mitigation is on-demand activation (not always-on) and power awareness that reduces activity on battery.

---

## 8. Contingency Paths

### 8.1 If Memory Gate Fails

If G2 fails (model stack exceeds 6.5GB), 8GB support is not viable. The project repositions as a "16GB+ premium local AI" product. Users with 8GB systems receive a cloud-only mode where local processing handles only triage and routing while generation happens via API.

### 8.2 If Quality Gate Fails

If G3 fails (HHEM below 0.4), summarization and drafting features are not viable. The project pivots to template-only quick replies, essentially becoming a smart reply button system similar to Gmail circa 2016. This is a significant reduction in ambition but may still provide value.

### 8.3 If Coverage Gate Fails

If G1 fails (template coverage below 40%), the template-first architecture is not viable. Options include heavy investment in generation quality (accepting higher memory and latency costs), accepting cloud API costs for generation, or project cancellation if no viable path exists.

### 8.4 If Latency Gates Fail

If G4 or G5 fail (latency exceeds acceptable thresholds), local inference becomes a backup rather than primary mode. Cloud fallback handles most requests with local inference used only when offline or for privacy-critical content.

### 8.5 Project Cancellation Criteria

If two or more gates fail with no viable mitigation, the project should be cancelled. The fundamental value proposition (local, private, fast AI assistant on consumer hardware) may be ahead of its time given current hardware constraints. This is an acceptable outcome; validation-first methodology exists precisely to fail fast when assumptions don't hold.

---

## 9. Development Phases

Development proceeds in four phases. The Development Guide provides detailed workstream specifications and agent prompts for each phase.

### Phase 0: Foundation

Establish the repository structure and interface contracts. Extract reusable code from previous projects (summarizationv2, jarvisv0). This phase must complete before any parallel work can begin because it defines the contracts that enable independent development.

**Deliverables**: Repository structure, interface contracts (Python Protocol definitions), pyproject.toml, extraction manifest documenting what was ported from where.

### Phase 1: Validation

Build and run the benchmark suite to evaluate all five gates. This phase answers the fundamental question: is this project viable?

**Workstreams**: Memory profiler, HHEM benchmark, template coverage analyzer, latency benchmark. These four workstreams have no dependencies on each other and can run in parallel.

**Deliverables**: Benchmark implementations, overnight evaluation results, gate status report, go/no-go decision.

### Phase 2: Core Infrastructure

Build the core services that other components depend on: memory management, health monitoring, graceful degradation.

**Workstreams**: Memory controller, degradation controller, permission and schema health. These three workstreams have no dependencies on each other and can run in parallel.

**Deliverables**: Core service implementations, unit tests, integration tests for failure scenarios.

### Phase 3: Features

Build the model loader, generator, and integrations with Gmail and iMessage.

**Workstreams**: Model loader and generator, Gmail integration, iMessage integration. The integrations depend on core interfaces but not implementations, so they can run in parallel with core work using mock implementations.

**Deliverables**: Feature implementations, end-to-end tests, final regression benchmark.

### Phase 4: Polish

Finalize documentation, create portfolio artifacts, publish.

**Deliverables**: README with demo, architecture documentation, benchmark report, LinkedIn posts, GitHub publication.

---

## 10. Architecture Decision Records

Key decisions are documented as Architecture Decision Records (ADRs) for future reference and interview discussion.

### ADR-001: Validation-First Development

**Context**: The v0 design failed because assumptions were wrong. We need a methodology that catches bad assumptions early.

**Decision**: Adopt validation-first development. Build benchmarks before features. Gate progress on empirical results.

**Consequences**: Slower initial progress but much lower risk of building something that doesn't work. Forces explicit go/no-go decisions rather than hopeful continuation.

### ADR-002: Three-Tier Memory Modes

**Context**: 8GB is the aspirational target but the math doesn't work for full functionality.

**Decision**: Support three modes (Full, Lite, Minimal) with automatic detection and degradation.

**Consequences**: More complex code paths but viable product across hardware range. Users get best available experience for their hardware rather than nothing.

### ADR-003: Template-First Architecture

**Context**: Model generation is expensive (memory, latency) and risky (hallucination). Many requests have predictable responses.

**Decision**: Match requests to templates first. Only generate when no template matches with sufficient confidence.

**Consequences**: Better latency and quality for common cases. Requires investment in template coverage analysis. Reduces the "AI magic" perception but increases reliability.

### ADR-004: No Fine-Tuning

**Context**: Research shows fine-tuning on new knowledge increases hallucinations. Original plan included LoRA fine-tuning.

**Decision**: Use RAG for factual grounding and few-shot prompting for style. No fine-tuning.

**Consequences**: Style matching is less precise but hallucination risk is not increased. Approach is cheaper and fully reversible.

### ADR-005: Gmail API over Local Index

**Context**: Local indexing provides offline search but adds significant complexity.

**Decision**: Use Gmail API directly with working-set approach (30 days recent).

**Consequences**: Requires network for email features. Dramatically simpler implementation. Leverages Gmail's search infrastructure.

### ADR-006: Read-Only iMessage with Schema Detection

**Context**: chat.db access is fragile due to schema changes and concurrent writes.

**Decision**: Read-only access, schema version detection, graceful fallback for unknown schemas.

**Consequences**: Cannot write to iMessage (acceptable for v1). Resilient to Apple's schema changes. Safe from database corruption.

### ADR-007: Self-Critique Checkpoints

**Context**: Agent execution without reflection leads to inefficient approaches and missed optimization opportunities.

**Decision**: Require explicit self-critique at pre-execution, during execution (checkpoints), and completion (verification).

**Consequences**: Slower individual task execution but higher quality output. Catches problems early rather than after hours of wasted work.

---

## 11. Verification Plan

### 11.1 Benchmark Verification

Each benchmark produces machine-readable results (JSON) and human-readable reports (Markdown). The overnight evaluation script runs all benchmarks sequentially (safe for 8GB systems) and generates a consolidated report.

```bash
# Run overnight evaluation
./scripts/overnight_eval.sh

# Check gate status
python scripts/check_gates.py results/latest

# View detailed report
cat docs/BENCHMARKS.md
```

### 11.2 Integration Testing

Integration tests verify behavior under failure conditions.

**Permission revocation**: Simulate TCC permission loss and verify graceful degradation to non-iMessage features.

**Memory pressure**: Simulate low memory conditions and verify model unloading and mode switching.

**Schema change**: Test against chat.db samples from different macOS versions and verify fallback parsing.

**Power state**: Verify reduced activity on battery power.

### 11.3 End-to-End Testing

End-to-end tests verify complete user scenarios.

**Cold start to response**: From application launch to first response, verify total time and quality.

**Template path**: Verify high-confidence template matches return instantly without model loading.

**Generation path**: Verify generation with RAG context produces acceptable HHEM scores.

**Degradation path**: Verify that when generation fails quality checks, fallback to template works correctly.

---

## 12. Glossary

**HHEM**: Hughes Hallucination Evaluation Model. A metric from Vectara that scores factual consistency between source text and generated summary. Range 0-1 where 1 is perfectly grounded.

**MLX**: Apple's machine learning framework optimized for Apple Silicon. Provides efficient inference with unified memory.

**RAG**: Retrieval-Augmented Generation. Technique of injecting retrieved documents into the prompt to ground generation in factual content.

**TCC**: Transparency, Consent, and Control. macOS permission framework that gates access to sensitive data like contacts, calendar, and full disk access.

**chat.db**: SQLite database where macOS stores iMessage history. Located at ~/Library/Messages/chat.db. Requires Full Disk Access permission to read.

**Cold start**: Application state where the model is not loaded in memory. Requires full model load from SSD.

**Warm start**: Application state where the model is already loaded. Only requires prompt processing and generation.

**Circuit breaker**: Pattern for graceful degradation. After N failures, stop trying the failing operation and use fallback behavior. Periodically test if the operation has recovered.

---

## 13. References

**Gekhman et al. (EMNLP 2024)**: "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" Demonstrates that fine-tuning on new factual knowledge increases hallucination rates.

**Vectara HHEM**: Hallucination evaluation model and documentation. Establishes 0.5 as industry threshold for acceptable factual consistency.

**Apple TCC Documentation**: Framework documentation for Transparency, Consent, and Control permission system.

**MLX Documentation**: Apple's machine learning framework documentation and mlx-lm library.

---

## 14. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-20 | Initial v0 design (subsequently rejected) |
| 2.0 | 2026-01-25 | Complete revision addressing adversarial review findings. Validation-first methodology. Self-critique integration. |

---

## Appendix A: Related Documents

**JARVIS v1 Development Guide**: Operational document with workstream definitions, agent prompts, interface contracts, and execution checklists.

**Adversarial Design Review**: Original review that identified the five critical blockers leading to v0 rejection.

**Self-Critique Mechanisms for Agent Execution**: Analysis of agent execution failures and required self-critique checkpoints.

---

## Appendix B: Implementation Status (as of 2026-01-25)

This appendix tracks what has actually been implemented versus what is described as planned in this document.

### Workstream Implementation Status

| Workstream | Description | Status | Notes |
|------------|-------------|--------|-------|
| WS1 | Memory Profiler | NOT IMPLEMENTED | Contract defined, stub only |
| WS2 | HHEM Benchmark | NOT IMPLEMENTED | Contract defined, stub only |
| WS3 | Template Coverage | COMPLETE | Full implementation with 75 templates, 1000 scenarios |
| WS4 | Latency Benchmark | NOT IMPLEMENTED | Contract defined, stub only |
| WS5 | Memory Controller | NOT IMPLEMENTED | Contract defined, stub only |
| WS6 | Degradation Controller | NOT IMPLEMENTED | Contract defined, stub only |
| WS7 | Permission/Schema | NOT IMPLEMENTED | Contract defined, stub only |
| WS8 | Model Generator | COMPLETE | MLX loader, template fallback, RAG support |
| WS9 | Gmail Integration | NOT IMPLEMENTED | Contract defined, stub only |
| WS10 | iMessage Reader | MOSTLY COMPLETE | Has TODOs for attachments/reactions |

### Validation Gates Status

| Gate | Can Be Evaluated? | Notes |
|------|-------------------|-------|
| G1 (Coverage) | YES | `python -m benchmarks.coverage.run` |
| G2 (Memory) | NO | WS1 not implemented |
| G3 (HHEM) | NO | WS2 not implemented |
| G4 (Warm Latency) | NO | WS4 not implemented |
| G5 (Cold Latency) | NO | WS4 not implemented |

### Missing Scripts

The following scripts referenced in this document do not exist:
- `scripts/overnight_eval.sh` - Overnight benchmark runner
- `scripts/generate_report.py` - Report generator

### Key Deviations from Design

1. **Model Size**: Document specifies 3B parameter model; current implementation defaults to Qwen2.5-0.5B-Instruct-4bit (much smaller)
2. **Memory Modes**: Three-tier modes (FULL/LITE/MINIMAL) defined in contract but not implemented
3. **Circuit Breaker**: Pattern described but no implementation exists
4. **HHEM Validation**: Described in data flow but not implemented

See `docs/CODEBASE_AUDIT_REPORT.md` for comprehensive audit details.
