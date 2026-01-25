# Makefile for JARVIS AI Assistant
# All commands use uv for package management

.PHONY: help install test test-verbose test-fail-fast lint clean

# Default target
help:
	@echo "JARVIS AI Assistant - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install        Install dependencies via uv sync"
	@echo "  test           Run all tests with output captured to test_results.txt"
	@echo "  test-verbose   Run tests with extra verbosity (-vvv)"
	@echo "  test-fail-fast Run tests, stop at first failure (--maxfail=1)"
	@echo "  lint           Run linters (ruff check and mypy)"
	@echo "  clean          Remove test artifacts, caches, and temp files"
	@echo "  help           Show this help message"

# Install dependencies
install:
	uv sync --extra dev

# Run all tests with output capture
# Output goes to both terminal (via tee) and test_results.txt
# JUnit XML also generated for CI integration
test:
	uv run pytest tests/ --tb=long -v --junit-xml=test_results.xml 2>&1 | tee test_results.txt

# Run tests with extra verbosity
test-verbose:
	uv run pytest tests/ --tb=long -vvv --junit-xml=test_results.xml 2>&1 | tee test_results.txt

# Run tests but stop at first failure (useful for debugging)
test-fail-fast:
	uv run pytest tests/ --tb=long -v --maxfail=1 --junit-xml=test_results.xml 2>&1 | tee test_results.txt

# Run linters
lint:
	uv run ruff check .
	uv run mypy core/ models/ integrations/ benchmarks/

# Clean up build artifacts and caches
clean:
	rm -f test_results.txt
	rm -f test_results.xml
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
