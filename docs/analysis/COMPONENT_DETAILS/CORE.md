# Core Subsystem Deep Dive

**Last Updated**: 2026-01-27

---

## Overview

The Core subsystem provides memory management and health monitoring services that other components depend on.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Core Services                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Memory Controller (WS5)                  │   │
│  │  - Three-tier modes: FULL / LITE / MINIMAL           │   │
│  │  - Memory pressure callbacks                          │   │
│  │  - Model loading decisions                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Degradation Controller (WS6)                │   │
│  │  - Circuit breaker pattern                            │   │
│  │  - CLOSED → OPEN → HALF_OPEN state machine           │   │
│  │  - Automatic fallback on failures                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────────────┐  ┌────────────────────────────┐  │
│  │  Permission Monitor   │  │   Schema Detector          │  │
│  │  (WS7)               │  │   (WS7)                    │  │
│  │  - Full Disk Access  │  │   - v14/v15 detection     │  │
│  │  - Contacts          │  │   - Fallback queries      │  │
│  │  - Calendar          │  │                            │  │
│  └───────────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Memory Controller (WS5)

### Purpose
Adaptive memory management for operation across different RAM configurations.

### Implementation

**File**: `core/memory/controller.py` (286 lines)

**Memory Modes**:
| Mode | RAM Available | Features |
|------|---------------|----------|
| FULL | >8GB | All features, concurrent models |
| LITE | 4-8GB | Sequential loading, reduced context |
| MINIMAL | <4GB | Templates only, cloud fallback |

**Thresholds**:
```python
@dataclass
class MemoryThresholds:
    full_mode_mb: float = 8000.0   # 8GB for FULL mode
    lite_mode_mb: float = 4000.0   # 4GB for LITE mode
    memory_buffer_multiplier: float = 1.2  # 20% safety buffer
```

**Key Methods**:
```python
def get_state(self) -> MemoryState:
    """Get current memory state."""

def get_mode(self) -> MemoryMode:
    """Determine appropriate mode based on available memory."""

def can_load_model(self, required_mb: float) -> bool:
    """Check if we have enough memory to load a model."""

def register_pressure_callback(self, callback: Callable[[str], None]) -> None:
    """Register callback for memory pressure events."""
```

**Singleton Access**:
```python
from core.memory import get_memory_controller, reset_memory_controller

controller = get_memory_controller()  # Thread-safe singleton
```

### Memory Monitor

**File**: `core/memory/monitor.py` (92 lines)

**Features**:
- System memory monitoring via psutil
- Pressure level detection: "green", "yellow", "red", "critical"

---

## 2. Degradation Controller (WS6)

### Purpose
Graceful failure handling with automatic fallback.

### Implementation

**File**: `core/health/degradation.py` (418 lines)

**Circuit Breaker States**:
```
    CLOSED (healthy)
        │
        │ failure_threshold reached
        ▼
    OPEN (failed)
        │
        │ recovery_timeout_seconds
        ▼
    HALF_OPEN (testing)
        │
        ├─ success → CLOSED
        └─ failure → OPEN
```

**Degradation Policy**:
```python
@dataclass
class DegradationPolicy:
    feature_name: str
    health_check: Callable[[], bool]
    degraded_behavior: Callable[..., Any]
    fallback_behavior: Callable[..., Any]
    recovery_check: Callable[[], bool]
    max_failures: int = 3
```

**Key Methods**:
```python
def register_feature(self, policy: DegradationPolicy) -> None:
    """Register a feature with its degradation policy."""

def execute(self, feature_name: str, *args, **kwargs) -> Any:
    """Execute feature with automatic fallback on failure."""

def get_health(self) -> dict[str, FeatureState]:
    """Return health status of all features."""

def reset_feature(self, feature_name: str) -> None:
    """Reset failure count and try healthy mode again."""
```

**Singleton Access**:
```python
from core.health import get_degradation_controller, reset_degradation_controller

controller = get_degradation_controller()  # Thread-safe singleton
```

### Circuit Breaker

**File**: `core/health/circuit.py` (302 lines)

**Configuration**:
```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 60.0
    half_open_max_calls: int = 1
```

**Statistics Tracking**:
- Failure count
- Success count
- Total executions
- Last failure/success time

---

## 3. Permission Monitor (WS7)

### Purpose
TCC (Transparency, Consent, Control) permission monitoring.

### Implementation

**File**: `core/health/permissions.py` (336 lines)

**Supported Permissions**:
| Permission | Purpose | Check Method |
|------------|---------|--------------|
| FULL_DISK_ACCESS | iMessage chat.db | File access test |
| CONTACTS | Name resolution | AddressBook access |
| CALENDAR | Calendar integration | Calendar access |
| AUTOMATION | AppleScript | osascript test |

**Key Methods**:
```python
def check_permission(self, permission: Permission) -> PermissionStatus:
    """Check if a specific permission is granted."""

def check_all(self) -> list[PermissionStatus]:
    """Check all required permissions."""

def wait_for_permission(self, permission: Permission, timeout_seconds: int) -> bool:
    """Block until permission granted or timeout."""
```

**Fix Instructions**:
Each `PermissionStatus` includes user-friendly fix instructions:
```python
@dataclass
class PermissionStatus:
    permission: Permission
    granted: bool
    last_checked: str
    fix_instructions: str  # "Grant in System Settings > Privacy & Security > ..."
```

---

## 4. Schema Detector (WS7)

### Purpose
Detect chat.db schema version across macOS releases.

### Implementation

**File**: `core/health/schema.py` (272 lines)

**Schema Versions**:
| Version | macOS | Key Differences |
|---------|-------|-----------------|
| v14 | Sonoma (14.x) | Standard schema |
| v15 | Sequoia (15.x) | Minor column changes |

**Detection Method**:
Delegates to `integrations/imessage/queries.py:detect_schema_version()` for single source of truth.

**Key Methods**:
```python
def detect(self, db_path: str) -> SchemaInfo:
    """Detect schema version and compatibility."""

def get_query(self, query_name: str, schema_version: str) -> str:
    """Get appropriate SQL query for the detected schema."""
```

---

## Test Coverage

| File | Coverage | Notes |
|------|----------|-------|
| `test_memory_controller.py` | 100% | All modes tested |
| `test_degradation.py` | 99% | Circuit breaker states |
| `test_permissions.py` | 100% | All permission types |
| `test_schema.py` | 99% | v14/v15 detection |

---

## Key Files

- `core/memory/controller.py` (286 lines)
- `core/memory/monitor.py` (92 lines)
- `core/health/degradation.py` (418 lines)
- `core/health/circuit.py` (302 lines)
- `core/health/permissions.py` (336 lines)
- `core/health/schema.py` (272 lines)
