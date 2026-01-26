"""Contract interfaces for JARVIS v1.

This module exports all Protocol interfaces that enable parallel workstream development.
All implementations should code against these contracts, not concrete implementations.
"""

from contracts.hallucination import (
    HallucinationEvaluator,
    HHEMBenchmarkResult,
    HHEMResult,
)
from contracts.health import (
    DegradationController,
    DegradationPolicy,
    FeatureState,
    Permission,
    PermissionMonitor,
    PermissionStatus,
    SchemaDetector,
    SchemaInfo,
)
from contracts.imessage import (
    Conversation,
    Message,
    iMessageReader,
)
from contracts.latency import (
    LatencyBenchmarker,
    LatencyBenchmarkResult,
    LatencyResult,
    Scenario,
)
from contracts.memory import (
    MemoryController,
    MemoryMode,
    MemoryProfile,
    MemoryProfiler,
    MemoryState,
)
from contracts.models import (
    GenerationRequest,
    GenerationResponse,
    Generator,
)

__all__ = [
    # Memory (WS1, WS5)
    "MemoryProfile",
    "MemoryMode",
    "MemoryState",
    "MemoryProfiler",
    "MemoryController",
    # Hallucination (WS2)
    "HHEMResult",
    "HHEMBenchmarkResult",
    "HallucinationEvaluator",
    # Latency (WS4)
    "Scenario",
    "LatencyResult",
    "LatencyBenchmarkResult",
    "LatencyBenchmarker",
    # Health (WS6, WS7)
    "FeatureState",
    "Permission",
    "PermissionStatus",
    "SchemaInfo",
    "DegradationPolicy",
    "DegradationController",
    "PermissionMonitor",
    "SchemaDetector",
    # Models (WS8)
    "GenerationRequest",
    "GenerationResponse",
    "Generator",
    # iMessage (WS10)
    "Message",
    "Conversation",
    "iMessageReader",
]
