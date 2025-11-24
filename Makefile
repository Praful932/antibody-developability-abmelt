# Makefile for AbMelt Inference Pipeline
# Cross-platform compatible (Unix-like + Windows via Git Bash/WSL)

.PHONY: all install format lint test clean help

# Detect OS
OS := $(shell uname -s 2>/dev/null || echo Windows)

# Python command (assumes uv is installed)
UV := uv

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install    Sync dependencies with uv"
	@echo "  format     Format code with ruff"
	@echo "  lint       Lint code with ruff and mypy"
	@echo "  test       Run tests with pytest"
	@echo "  clean      Remove build artifacts and cache"
	@echo "  all        Run format, lint, and test"

install:  ## Sync dependencies
	$(UV) sync

format:  ## Format code
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:  ## Run static analysis
	$(UV) run ruff check .
	$(UV) run mypy abmelt_infer_pipeline/src/

test:  ## Run tests
	$(UV) run pytest

all: format lint test  ## Run all checks

clean:  ## Clean up
	@echo "Cleaning up..."
	@rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
