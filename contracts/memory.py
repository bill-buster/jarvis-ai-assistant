"""Memory profiling and control interfaces.

Workstreams 1 and 5 implement against these contracts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass
class MemoryProfile:
    """Result of profiling a model's memory usage."""

    model_name: str
    quantization: str
    context_length: int
    rss_mb: float  # Resident Set Size (actual RAM used)
    virtual_mb: float  # Virtual memory allocated
    metal_mb: float  # GPU memory (Apple Metal)
    load_time_seconds: float
    timestamp: str  # ISO format


class MemoryMode(Enum):
    """Operating modes based on available memory."""

    FULL = "full"  # 16GB+ : All features, concurrent models
    LITE = "lite"  # 8-16GB : Sequential loading, reduced context
    MINIMAL = "minimal"  # <8GB : Templates only, cloud fallback


@dataclass
class MemoryState:
    """Current memory status of the system."""

    available_mb: float
    used_mb: float
    model_loaded: bool
    current_mode: MemoryMode
    pressure_level: str  # "green", "yellow", "red", "critical"


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
