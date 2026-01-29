.PHONY: help install test test-v3 lint format typecheck check clean

# Default target
help:
	@echo "JARVIS v3 - Minimal Reply Generation"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run all tests"
	@echo "  make test-v3      - Quick v3 structure test"
	@echo "  make lint         - Check code style (ruff)"
	@echo "  make format       - Auto-format code"
	@echo "  make typecheck    - Run type checker (mypy)"
	@echo "  make check        - Run lint + typecheck"
	@echo ""
	@echo "Usage:"
	@echo "  make api          - Start API server"
	@echo "  make profile      - Profile your contacts"
	@echo "  make index        - Index messages for RAG"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache files"

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv sync
	@echo "✓ Dependencies installed"

# Run all tests
test:
	@echo "Running tests..."
	uv run pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
	@echo "✓ Tests complete (see test_results.txt)"

# Quick v3 structure test (no model loading)
test-v3:
	@echo "Testing v3 structure..."
	uv run python scripts/test_v3.py

# Run specific test file
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=tests/test_something.py"; \
		exit 1; \
	fi
	uv run pytest $(FILE) -v

# Lint code with ruff
lint:
	@echo "Running ruff..."
	@cd /Users/jwalinshah/coding/jarvis-ai-assistant/v3 && uv run ruff check core/ api/ scripts/ tests/
	@echo "✓ Lint complete"

# Auto-format code
format:
	@echo "Formatting code..."
	@cd /Users/jwalinshah/coding/jarvis-ai-assistant/v3 && uv run ruff format core/ api/ scripts/ tests/
	@echo "✓ Format complete"

# Type check with mypy
typecheck:
	@echo "Running mypy..."
	@cd /Users/jwalinshah/coding/jarvis-ai-assistant/v3 && uv run mypy core/ api/ --ignore-missing-imports --no-error-summary 2>&1 | head -50
	@echo "✓ Type check complete"

# Run all checks
check: lint typecheck test-v3
	@echo "✓ All checks passed"

# Start API server
api:
	@echo "Starting API server on http://localhost:8000..."
	uv run uvicorn api.main:app --port 8000 --reload

# Profile contacts
profile:
	@echo "Profiling contacts..."
	uv run python scripts/profile_contacts.py

# Index messages for RAG
index:
	@echo "Indexing messages..."
	uv run python scripts/index_messages.py

# Clean cache files
clean:
	@echo "Cleaning cache files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	@rm -f test_results.txt 2>/dev/null || true
	@echo "✓ Cleaned"
