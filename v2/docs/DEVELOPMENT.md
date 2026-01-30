# Development Guide

This guide covers setting up and developing JARVIS v2.

## Prerequisites

- **macOS 11+** (Big Sur or later)
- **Apple Silicon** (M1/M2/M3) - required for MLX
- **Python 3.11+**
- **Node.js 18+** and **pnpm**
- **uv** (Python package manager)

## Initial Setup

### 1. Clone and Install Python Dependencies

```bash
# From project root
cd jarvis-ai-assistant
uv sync
```

### 2. Install Desktop App Dependencies

```bash
cd v2
make install  # Runs: cd desktop && pnpm install
```

### 3. Grant Permissions

Go to **System Settings → Privacy & Security → Full Disk Access** and add:
- Your terminal app (Terminal, iTerm2, etc.)
- The JARVIS app (after first build)

### 4. Verify Setup

```bash
# Start API server
make api

# In another terminal, test health endpoint
curl http://localhost:8000/health
```

## Development Workflow

### Starting the App

```bash
# Option 1: Full desktop app (API + native window)
make app

# Option 2: API + browser mode (better for debugging)
make dev

# Option 3: API only
make api
```

### Code Changes

**Backend (Python)**:
- Changes to `api/` or `core/` auto-reload with `--reload` flag
- No restart needed for most changes

**Frontend (Svelte)**:
- Changes hot-reload automatically
- State preserved during HMR

### Running Tests

```bash
make test          # Run all tests
make test-gen      # Interactive generation test
```

### Code Quality

```bash
make lint          # Check for issues
make format        # Auto-format code
```

## Project Structure

```
v2/
├── api/                 # FastAPI endpoints
│   ├── main.py         # App setup
│   ├── schemas.py      # Request/response models
│   └── routes/         # Endpoint handlers
│
├── core/               # Business logic
│   ├── models/         # MLX model loading
│   ├── generation/     # Reply generation
│   ├── imessage/       # iMessage access
│   ├── embeddings/     # Vector search
│   └── templates/      # Response templates
│
├── desktop/            # Tauri/Svelte app
│   ├── src/            # Svelte code
│   │   ├── lib/        # Components, stores, API
│   │   └── routes/     # Pages
│   └── src-tauri/      # Native config
│
├── scripts/            # Utilities
├── tests/              # Test suite
└── docs/               # Documentation
```

## Common Tasks

### Adding a New API Endpoint

1. **Add schema** (`api/schemas.py`):
```python
class MyRequest(BaseModel):
    field: str

class MyResponse(BaseModel):
    result: str
```

2. **Create route** (`api/routes/myroute.py`):
```python
from fastapi import APIRouter
from ..schemas import MyRequest, MyResponse

router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint(request: MyRequest) -> MyResponse:
    return MyResponse(result="done")
```

3. **Register route** (`api/main.py`):
```python
from .routes import myroute
app.include_router(myroute.router, tags=["myfeature"])
```

4. **Add TypeScript types** (`desktop/src/lib/api/types.ts`):
```typescript
export interface MyResponse {
  result: string;
}
```

5. **Add client method** (`desktop/src/lib/api/client.ts`):
```typescript
async myEndpoint(field: string): Promise<MyResponse> {
  return this.post('/my-endpoint', { field });
}
```

### Adding a New Svelte Component

1. **Create component** (`desktop/src/lib/components/MyComponent.svelte`):
```svelte
<script lang="ts">
  export let title: string;
</script>

<div class="my-component">
  <h2>{title}</h2>
</div>

<style>
  .my-component {
    padding: 1rem;
  }
</style>
```

2. **Use in page** (`desktop/src/routes/+page.svelte`):
```svelte
<script>
  import MyComponent from '$lib/components/MyComponent.svelte';
</script>

<MyComponent title="Hello" />
```

### Modifying the Generation Pipeline

Key file: `core/generation/reply_generator.py`

**Adding a new analysis stage**:
```python
# In generate_replies() method

# Add timing
t0 = time.time()
my_result = self._my_analysis(messages)
timings["my_analysis"] = (time.time() - t0) * 1000

# Use result in prompt building
prompt = build_reply_prompt(
    ...,
    my_param=my_result,
)
```

**Adding a new prompt parameter**:
1. Update `build_reply_prompt()` in `core/generation/prompts.py`
2. Update the prompt template
3. Pass the new parameter from `reply_generator.py`

### Working with Embeddings

**Index new messages**:
```bash
python scripts/index_messages.py
```

**Test embedding search**:
```bash
python scripts/test_embeddings.py
```

**Programmatic access**:
```python
from core.embeddings import get_embedding_store

store = get_embedding_store()
results = store.find_similar("dinner tomorrow", limit=5)
for msg in results:
    print(f"{msg.text} (similarity: {msg.similarity:.2f})")
```

## Debugging

### Backend Debugging

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check generation timing**:
```
Generation completed in 1250ms -
  template:2ms, style:45ms, past_replies:120ms, LLM:950ms
```

### Frontend Debugging

- **DevTools**: Right-click → Inspect (or Cmd+Option+I in Tauri)
- **Svelte DevTools**: Browser extension for component inspection
- **Network tab**: Monitor API calls
- **Console**: Check for errors

### Common Issues

**"Cannot access iMessage database"**
- Grant Full Disk Access in System Settings
- Restart terminal after granting

**"Model not found"**
- First run downloads model (~500MB)
- Check `~/.cache/` for model files
- Ensure internet connection

**"FAISS not available"**
- Install: `pip install faiss-cpu`
- Falls back to brute-force search if unavailable

**"WebSocket disconnected"**
- Check if API server is running
- Look for CORS errors in console
- Verify port 8000 is free

## Testing

### Unit Tests

```bash
make test
```

### Interactive Testing

```bash
# Test generation with real messages
make test-gen
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Get conversations
curl http://localhost:8000/conversations

# Generate replies
curl -X POST http://localhost:8000/generate/replies \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "iMessage;+;chat123", "num_replies": 3}'
```

## Performance Profiling

### Generation Timing

Every generation logs a timing breakdown:
```
Generation completed in 1250ms -
  template:2ms, coherence:5ms, style:45ms, context:30ms,
  past_replies:120ms, profile:25ms, refresh:50ms,
  LLM:950ms, parse:3ms
```

### Memory Profiling

```python
import tracemalloc
tracemalloc.start()

# ... your code ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f}MB")
print(f"Peak: {peak / 1024 / 1024:.1f}MB")
```

### Model Benchmarking

```bash
python scripts/benchmark_models.py
```

## Building for Production

### Desktop App

```bash
cd desktop
pnpm exec tauri build
```

Output: `desktop/src-tauri/target/release/bundle/`

### API Server

```bash
# Production server (no reload)
uvicorn v2.api.main:app --host 0.0.0.0 --port 8000
```

## Code Style

- **Python**: Ruff (line-length=100)
- **TypeScript/Svelte**: Prettier (default settings)
- **Formatting**: Run `make format` before committing

## Useful Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_generation.py` | Interactive generation testing |
| `scripts/benchmark_models.py` | Compare model performance |
| `scripts/index_messages.py` | Build embedding indices |
| `scripts/test_embeddings.py` | Test embedding system |
| `scripts/build_response_templates.py` | Learn response patterns |
| `scripts/extract_training_pairs.py` | Extract training data |
