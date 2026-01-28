.PHONY: install dev test lint serve clean

# Install dependencies
install:
	cd .. && uv pip install -e ./v2

# Run development server
dev:
	cd .. && uv run uvicorn v2.api.main:app --reload --port 8000

# Run server (production)
serve:
	cd .. && uv run uvicorn v2.api.main:app --port 8000

# Run tests
test:
	cd .. && uv run pytest v2/tests -v

# Lint code
lint:
	cd .. && uv run ruff check v2/

# Format code
format:
	cd .. && uv run ruff format v2/

# Test generation manually
test-gen:
	cd .. && uv run python -m v2.scripts.test_generation

# Clean cache
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# Show help
help:
	@echo "JARVIS v2 Commands:"
	@echo "  make dev      - Run development server with auto-reload"
	@echo "  make serve    - Run production server"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Check code style"
	@echo "  make test-gen - Test generation manually"
