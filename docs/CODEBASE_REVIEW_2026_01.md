# Codebase Review - January 2026

Comprehensive review of the JARVIS AI Assistant codebase identifying issues across code quality, architecture, performance, error handling, security, test coverage, and documentation.

## Summary

| Priority | Count | Categories |
|----------|-------|------------|
| **High** | 6 | Architecture, Security, Thread Safety, Performance |
| **Medium** | 8 | Error Handling, Validation, Resource Management |
| **Low** | 6 | Dead Code, Naming, Documentation |

---

## High Priority Issues

### 1. Code Duplication - CLI and API Initialization (Architecture)

**Files**: `jarvis/cli.py:44-128` and `jarvis/api.py:53-176`

**Problem**: Nearly identical initialization code is duplicated between CLI and API modules:
- `initialize_system()` / `_initialize_system()`
- `_check_imessage_access()`
- `_template_only_response()`
- `_fallback_response()`
- `_imessage_degraded()`
- `_imessage_fallback()`

**Impact**: Bug fixes must be applied twice; risk of divergence over time.

**Suggested Fix**: Extract shared initialization logic to a common module.

```python
# jarvis/system.py (new file)
"""System initialization shared by CLI and API."""

from contracts.health import DegradationPolicy
from core.health import get_degradation_controller
from core.memory import get_memory_controller

FEATURE_CHAT = "chat"
FEATURE_IMESSAGE = "imessage"

def check_imessage_access() -> bool:
    """Check if iMessage database is accessible."""
    try:
        from integrations.imessage import ChatDBReader
        with ChatDBReader() as reader:
            return reader.check_access()
    except Exception:
        return False

def template_only_response(prompt: str) -> str:
    """Generate response using only template matching."""
    try:
        from models.templates import TemplateMatcher
        matcher = TemplateMatcher()
        match = matcher.match(prompt)
        if match:
            return match.template.response
    except Exception:
        pass
    return "I'm operating in limited mode. Please try a simpler query."

def fallback_response() -> str:
    """Return a fallback response when chat is unavailable."""
    return "I'm currently unable to process your request. Please check system health."

def imessage_fallback() -> list:
    """Return fallback for iMessage when unavailable."""
    return []

def initialize_system() -> tuple[bool, list[str]]:
    """Initialize JARVIS system components."""
    # ... shared initialization logic
```

---

### 2. Config Singleton Missing Thread Safety (Thread Safety)

**File**: `jarvis/config.py:212-221`

**Problem**: The `get_config()` singleton doesn't use locking, unlike all other singletons in the codebase.

```python
def get_config() -> JarvisConfig:
    global _config
    if _config is None:
        _config = load_config()  # Race condition!
    return _config
```

**Impact**: Race condition when multiple threads call `get_config()` simultaneously during startup.

**Suggested Fix**: Add double-check locking like other singletons.

```python
import threading

_config: JarvisConfig | None = None
_config_lock = threading.Lock()

def get_config() -> JarvisConfig:
    """Get singleton configuration instance (thread-safe)."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config
```

---

### 3. CORS Wildcard Pattern Invalid (Security)

**File**: `jarvis/api.py:203-209`

**Problem**: The CORS configuration uses `http://localhost:*` which is not a valid pattern for FastAPI's CORSMiddleware.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://localhost", "http://localhost:*"],
    # ...
)
```

**Impact**: Localhost ports other than default won't work, breaking development setups.

**Suggested Fix**: Use explicit origins or a regex pattern.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 4. LRUCache Not Thread-Safe (Thread Safety)

**File**: `integrations/imessage/reader.py:32-77`

**Problem**: `LRUCache` is used by `ChatDBReader` but has no thread safety, despite the class docstring noting thread-safety concerns.

```python
class LRUCache(Generic[K, V]):
    def get(self, key: K) -> V | None:
        if key in self._cache:
            self._cache.move_to_end(key)  # Not thread-safe!
            return self._cache[key]
```

**Impact**: Data corruption or KeyError when accessed from multiple threads.

**Suggested Fix**: Add thread-safe access with a lock.

```python
class LRUCache(Generic[K, V]):
    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
```

---

### 5. Circuit Breaker Race Condition (Thread Safety)

**File**: `core/health/circuit.py:218-248`

**Problem**: The `execute` method checks `can_execute()` then later calls `record_success/failure`, but state can change between these calls.

```python
def execute(self, func: Callable[..., object], *args, **kwargs) -> object:
    if not self.can_execute():  # Check happens here
        raise CircuitOpenError(...)

    try:
        result = func(*args, **kwargs)  # Another thread could modify state
        self.record_success()  # State may have changed
        return result
```

**Impact**: Inconsistent state tracking under concurrent load.

**Suggested Fix**: Use the lock for the entire execute operation.

```python
def execute(self, func: Callable[..., object], *args, **kwargs) -> object:
    with self._lock:
        self._check_state_transition()

        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' half-open limit reached")

    # Execute outside lock to avoid blocking
    try:
        result = func(*args, **kwargs)
        self.record_success()
        return result
    except Exception:
        self.record_failure()
        raise
```

---

### 6. Sentence Model Memory Leak (Performance)

**File**: `models/templates.py:23-64`

**Problem**: The sentence transformer model (`_sentence_model`) is lazily loaded but never automatically unloaded. The `unload_sentence_model()` function exists but is never called.

```python
_sentence_model: SentenceTransformer | None = None

def _get_sentence_model() -> Any:
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model
```

**Impact**: ~90MB memory held indefinitely even when template matching isn't needed.

**Suggested Fix**: Integrate unloading into the memory controller's pressure callbacks.

```python
# In jarvis/cli.py or jarvis/system.py during initialization
from models.templates import unload_sentence_model

def handle_memory_pressure(level: str) -> None:
    """Handle memory pressure by unloading non-essential models."""
    if level in ("red", "critical"):
        unload_sentence_model()

# Register with memory controller
mem_controller = get_memory_controller()
mem_controller.register_pressure_callback(handle_memory_pressure)
```

---

## Medium Priority Issues

### 7. AddressBook Connection Not Properly Closed (Resource Management)

**File**: `integrations/imessage/reader.py:320-372`

**Problem**: AddressBook database connection closed without try/finally.

```python
def _load_contacts_from_db(self, db_path: Path) -> None:
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
        # ... operations ...
        conn.close()  # Not in finally block!
    except (sqlite3.Error, OSError) as e:
        logger.debug(f"Error loading contacts: {e}")
        # Connection may be left open on error
```

**Suggested Fix**: Use context manager or try/finally.

```python
def _load_contacts_from_db(self, db_path: Path) -> None:
    if self._contacts_cache is None:
        self._contacts_cache = {}

    cache = self._contacts_cache
    uri = f"file:{db_path}?mode=ro"

    try:
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT_SECONDS)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # ... load contacts ...
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as e:
        logger.debug(f"Error loading contacts: {e}")
```

---

### 8. WS3 Import Always Fails (Dead Code)

**File**: `models/templates.py:935-962`

**Problem**: `_load_templates()` tries to import `benchmarks.coverage.templates` which doesn't exist (WS3 was removed per CLAUDE.md).

```python
def _load_templates() -> list[ResponseTemplate]:
    try:
        from benchmarks.coverage.templates import get_templates_by_category  # Always fails!
        # ...
    except ImportError:
        logger.warning("WS3 templates not available, using minimal fallback set")
        return _get_minimal_fallback_templates()  # Always executes
```

**Impact**: Unnecessary import attempt and misleading warning on every startup.

**Suggested Fix**: Remove dead code branch.

```python
def _load_templates() -> list[ResponseTemplate]:
    """Load templates. Returns comprehensive fallback set."""
    return _get_minimal_fallback_templates()
```

---

### 9. Config Migration Not Persisted (Architecture)

**File**: `jarvis/config.py:147-181`

**Problem**: When config is migrated from v1 to v2, the migrated version isn't saved back to disk.

```python
def load_config(config_path: Path | None = None) -> JarvisConfig:
    # ...
    data = _migrate_config(data)  # Migrates in memory
    return JarvisConfig.model_validate(data)  # Returns migrated config
    # Never calls save_config()!
```

**Impact**: Migration runs on every application start for users with v1 configs.

**Suggested Fix**: Save migrated config.

```python
def load_config(config_path: Path | None = None) -> JarvisConfig:
    path = config_path or CONFIG_PATH
    # ... load and parse ...

    original_version = data.get("config_version", 1)
    data = _migrate_config(data)

    try:
        config = JarvisConfig.model_validate(data)

        # Save if migrated
        if original_version < CONFIG_VERSION:
            save_config(config, path)
            logger.info(f"Config migrated and saved to {path}")

        return config
    except ValidationError as e:
        logger.warning(f"Config validation failed: {e}, using defaults")
        return JarvisConfig()
```

---

### 10. Variable Shadowing in CLI (Code Quality)

**File**: `jarvis/cli.py:349`

**Problem**: Local variable `sender` shadows the parameter from `args.sender`.

```python
def cmd_search_messages(args: argparse.Namespace) -> int:
    sender = args.sender  # Line 298
    # ...
    for msg in messages[:limit]:
        sender = "Me" if msg.is_from_me else (msg.sender or "Unknown")  # Line 349 - shadows!
```

**Impact**: Confusing code, potential bugs if earlier code is modified.

**Suggested Fix**: Rename the loop variable.

```python
for msg in messages[:limit]:
    display_sender = "Me" if msg.is_from_me else (msg.sender or "Unknown")
    text = msg.text[:80] + "..." if len(msg.text) > 80 else msg.text
    table.add_row(date_str, display_sender, text)
```

---

### 11. Dead Code - SearchRequest Model (Dead Code)

**File**: `jarvis/api_models.py:67-77`

**Problem**: `SearchRequest` Pydantic model is defined but never used. The `/search` endpoint uses query parameters directly.

```python
class SearchRequest(BaseModel):  # Never instantiated
    """Request for message search."""
    query: str = Field(...)
    # ...
```

**Impact**: Confusing API surface, maintenance burden.

**Suggested Fix**: Either remove the model or use it in the endpoint.

```python
# Option A: Remove SearchRequest from api_models.py

# Option B: Use it in api.py
@app.post("/search", ...)
async def search_messages(request: SearchRequest) -> SearchResponse:
    # ...
```

---

### 12. ErrorResponse Model Not Used Consistently (Error Handling)

**File**: `jarvis/api.py` (multiple locations)

**Problem**: `ErrorResponse` model is documented in OpenAPI spec but actual errors use FastAPI's default format.

```python
@app.get("/search", responses={403: {"model": ErrorResponse, ...}})
async def search_messages(...):
    # ...
    raise HTTPException(status_code=403, detail="Cannot access...")
    # Returns {"detail": "..."} not {"error": ..., "message": ...}
```

**Impact**: API clients receive different error format than documented.

**Suggested Fix**: Create a custom exception handler or use JSONResponse.

```python
from fastapi.responses import JSONResponse

@app.exception_handler(PermissionError)
async def permission_error_handler(request, exc):
    return JSONResponse(
        status_code=403,
        content={
            "error": "permission_denied",
            "message": str(exc),
            "details": "Grant Full Disk Access in System Settings."
        }
    )
```

---

### 13. Empty Prompt Validation After Model Load (Performance)

**File**: `models/loader.py:201-207`

**Problem**: Empty prompt validation happens after checking if model is loaded, but ideally should happen first.

```python
def generate_sync(self, prompt: str, ...) -> GenerationResult:
    if not self.is_loaded():
        raise RuntimeError("Model not loaded. Call load() first.")

    if not prompt or not prompt.strip():  # Should be first!
        raise ValueError("Prompt cannot be empty")
```

**Impact**: Minor - wastes cycles checking model state for invalid input.

**Suggested Fix**: Validate input first.

```python
def generate_sync(self, prompt: str, ...) -> GenerationResult:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if not self.is_loaded():
        raise RuntimeError("Model not loaded. Call load() first.")
```

---

### 14. Incomplete Directory Iteration Error Handling (Error Handling)

**File**: `integrations/imessage/reader.py:298-306`

**Problem**: Only catches `PermissionError` and `OSError` for iterdir, but other exceptions are possible.

```python
try:
    for source_dir in ADDRESSBOOK_DB_PATH.iterdir():  # Could raise other errors
        ab_db = source_dir / "AddressBook-v22.abcddb"
        if ab_db.exists():
            self._load_contacts_from_db(ab_db)
            return
except (PermissionError, OSError) as e:
    logger.debug(f"Cannot access AddressBook: {e}")
```

**Suggested Fix**: Catch broader exception set or use general Exception.

```python
try:
    for source_dir in ADDRESSBOOK_DB_PATH.iterdir():
        ab_db = source_dir / "AddressBook-v22.abcddb"
        if ab_db.exists():
            self._load_contacts_from_db(ab_db)
            return
except Exception as e:
    logger.debug(f"Cannot access AddressBook: {e}")
```

---

## Low Priority Issues

### 15. Unused `_adapter_path` Field (Dead Code)

**File**: `models/loader.py:63`

**Problem**: `_adapter_path` is initialized and cleared but never used.

```python
class MLXModelLoader:
    def __init__(self, ...):
        self._adapter_path: str | None = None  # Never used

    def unload(self) -> None:
        self._adapter_path = None  # Cleared but never set
```

**Suggested Fix**: Remove unless planned for LoRA adapter support.

---

### 16. Duplicate Memory Mode Enums (Code Quality)

**File**: `contracts/memory.py` and `jarvis/api_models.py`

**Problem**: `MemoryMode` enum exists in contracts, `MemoryModeEnum` duplicates it in API models.

**Suggested Fix**: Import and use the contract enum, or create a mapping function.

```python
# In api_models.py
from contracts.memory import MemoryMode

# Or keep both but document why (API serialization requirements)
```

---

### 17. Function Defined Inside Loop (Code Quality)

**File**: `jarvis/cli.py:258-267`

**Problem**: `generate_response` function is defined inside the `while True` chat loop, creating new function object each iteration.

```python
while True:
    # ...
    def generate_response(prompt: str) -> str:  # Created every loop iteration
        request = GenerationRequest(...)
        response = generator.generate(request)
        return response.text
```

**Suggested Fix**: Define outside the loop.

```python
def _create_response_generator(generator):
    def generate_response(prompt: str) -> str:
        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],
            few_shot_examples=[],
            max_tokens=200,
            temperature=0.7,
        )
        return generator.generate(request).text
    return generate_response

def cmd_chat(args: argparse.Namespace) -> int:
    generator = get_generator()
    generate_response = _create_response_generator(generator)
    # ...
```

---

### 18. Magic Numbers Without Constants (Code Quality)

**File**: Multiple locations

**Problem**: Magic numbers appear without named constants:
- `jarvis/cli.py:263`: `max_tokens=200`
- `jarvis/cli.py:350`: `msg.text[:80]` (truncation length)
- `integrations/imessage/reader.py:129`: `maxsize=10000`

**Suggested Fix**: Extract to named constants.

```python
# At module level
DEFAULT_MAX_TOKENS = 200
MESSAGE_PREVIEW_LENGTH = 80
GUID_CACHE_SIZE = 10_000
```

---

### 19. Inconsistent f-string vs % Formatting (Code Quality)

**File**: Multiple locations

**Problem**: Logger calls mix f-strings and % formatting inconsistently.

```python
# f-string (in reader.py)
logger.debug(f"Detected chat.db schema version: {self._schema_version}")

# % formatting (in loader.py)
logger.info("Loading model: %s", self.config.model_path)
```

**Impact**: Minor inconsistency; % formatting is preferred for logging (avoids string interpolation when logging level disabled).

**Suggested Fix**: Standardize on % formatting for logger calls.

---

### 20. Documentation Drift - WS3 References (Documentation)

**File**: `CLAUDE.md` and code

**Problem**: CLAUDE.md says "Template Coverage (WS3): REMOVED - Functionality moved to models/templates.py" but code still tries to import from `benchmarks.coverage.templates`.

**Suggested Fix**: Update code to remove WS3 import attempt (see issue #8).

---

## Test Coverage Gaps

### Missing Tests

1. **`models/templates.py`**: No dedicated test file for `TemplateMatcher` class
   - Missing: template matching threshold edge cases
   - Missing: sentence model loading failures
   - Missing: cache clearing behavior

2. **`models/prompt_builder.py`**: No test file visible for `PromptBuilder`
   - Missing: RAG context injection
   - Missing: few-shot example formatting

3. **`core/memory/monitor.py`**: Appears undertested
   - Missing: pressure level boundary conditions
   - Missing: psutil error handling

4. **`integrations/imessage/parser.py`**: No direct tests
   - Missing: attributedBody parsing edge cases
   - Missing: phone number normalization international formats

### Brittle Tests

1. **`tests/integration/test_api.py`**: Heavy use of MagicMock can mask interface changes
   - Recommendation: Add contract tests verifying mock interfaces match real implementations

---

## Recommendations Summary

### Immediate Actions (High Priority)

1. Extract shared CLI/API initialization code to `jarvis/system.py`
2. Add thread-safe locking to `get_config()` singleton
3. Fix CORS configuration with explicit origins or proper regex
4. Add thread safety to `LRUCache` class
5. Fix circuit breaker race condition

### Short-term Actions (Medium Priority)

1. Remove dead WS3 import code
2. Save migrated config to disk
3. Fix variable shadowing in CLI
4. Standardize error response format

### Long-term Actions (Low Priority)

1. Clean up dead code (`_adapter_path`, `SearchRequest`)
2. Standardize logging format
3. Extract magic numbers to constants
4. Add missing test coverage for templates and prompt builder
