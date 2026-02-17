# SpecSplit — Disaggregated Speculative Decoding
# ==============================================================================

.PHONY: help install proto test lint typecheck format clean all

PROTO_DIR   := specsplit/proto
PROTO_SRC   := $(PROTO_DIR)/spec_decoding.proto

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

proto: ## Generate Python stubs from protobuf definitions
	python -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(PROTO_DIR) \
		--grpc_python_out=$(PROTO_DIR) \
		--mypy_out=$(PROTO_DIR) \
		$(PROTO_SRC)
	@echo "✓ Proto stubs generated in $(PROTO_DIR)/"

test: ## Run unit tests (excludes integration)
	pytest tests/unit/ -v --tb=short

test-all: ## Run all tests including integration
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/unit/ -v --cov=specsplit --cov-report=term-missing --cov-report=html

lint: ## Run linter (ruff)
	ruff check specsplit/ tests/ scripts/

typecheck: ## Run static type checking (mypy)
	mypy specsplit/

format: ## Auto-format code (ruff)
	ruff format specsplit/ tests/ scripts/
	ruff check --fix specsplit/ tests/ scripts/

clean: ## Remove generated files, caches, and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f $(PROTO_DIR)/*_pb2.py $(PROTO_DIR)/*_pb2_grpc.py $(PROTO_DIR)/*_pb2.pyi
	@echo "✓ Cleaned."

all: install proto lint typecheck test ## Full setup: install, generate protos, lint, typecheck, test
