# SpecSplit — Project Guide

## What Is SpecSplit?

SpecSplit is a research framework for **disaggregated speculative decoding** — an
approach to accelerating large language model (LLM) inference by splitting the
workload across two networked GPUs:

| Worker | Role | Hardware |
|--------|------|----------|
| **Draft Worker** ("The Hare") | Generates speculative token trees with a small, fast LLM | Cheap GPU |
| **Target Worker** ("The Tortoise") | Verifies drafts via tree-attention with a large, accurate LLM | Expensive GPU |

An **Orchestrator** coordinates the draft→verify ping-pong loop and presents a
single user-facing interface.

> **Key insight:** The draft model is cheap to run, so speculating several tokens
> ahead is fast.  The target model verifies the entire tree in *one* batched
> forward pass, amortizing its high per-token cost.

---

## Repository Layout

```
specsplit/
├── proto/                  # gRPC protobuf definitions
│   └── spec_decoding.proto
├── core/                   # Shared utilities
│   ├── config.py           #   Pydantic settings (env-overridable)
│   ├── serialization.py    #   Tensor ↔ list conversion
│   ├── telemetry.py        #   High-precision timing + JSON spans
│   └── verification.py     #   Greedy tree verification math
├── workers/
│   ├── draft/              # Draft Worker microservice
│   │   ├── engine.py       #   Autoregressive generation + KV cache
│   │   └── service.py      #   gRPC server/client bindings
│   ├── target/             # Target Worker microservice
│   │   ├── engine.py       #   Session-based KV-cached verification
│   │   ├── service.py      #   gRPC server bindings
│   │   ├── tree_attn.py    #   Custom tree attention masking
│   │   └── kv_cache.py     #   Pre-allocated static KV cache
│   └── orchestrator/       # Pipeline coordinator
│       ├── client.py       #   User-facing entry point
│       └── pipeline.py     #   Async overlapped draft→verify loop
tests/
├── unit/                   # Fast, no-model tests
└── integration/            # Tests requiring model downloads
scripts/
└── benchmark_run.py        # Load-testing harness with CSV output
docs/                       # You are here
```

---

## Installation

```bash
# Clone
git clone https://github.com/<your-org>/SpecSplit.git
cd SpecSplit

# Create a virtual environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Generate gRPC stubs from proto
make proto
```

### Environment Variables

All configuration is driven by Pydantic settings with a `SPECSPLIT_` prefix.
Override any default via the environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPECSPLIT_DRAFT_MODEL_NAME` | `gpt2` | HuggingFace draft model ID |
| `SPECSPLIT_DRAFT_DEVICE` | `cuda:0` | Draft worker torch device |
| `SPECSPLIT_DRAFT_MAX_DRAFT_TOKENS` | `5` | Gamma — speculation depth |
| `SPECSPLIT_TARGET_MODEL_NAME` | `meta-llama/Llama-2-7b-hf` | Target model ID |
| `SPECSPLIT_TARGET_DEVICE` | `cuda:0` | Target worker torch device |
| `SPECSPLIT_TARGET_MAX_SESSIONS` | `16` | Max concurrent KV cache sessions |
| `SPECSPLIT_ORCH_DRAFT_ADDRESS` | `localhost:50051` | gRPC address of draft worker |
| `SPECSPLIT_ORCH_TARGET_ADDRESS` | `localhost:50052` | gRPC address of target worker |
| `SPECSPLIT_ORCH_MAX_OUTPUT_TOKENS` | `256` | Max tokens to generate per prompt |

---

## Quick Start

### 1. Start the Target Worker

```bash
SPECSPLIT_TARGET_MODEL_NAME=meta-llama/Llama-3.1-70B \
SPECSPLIT_TARGET_DEVICE=cuda:0 \
    python -m specsplit.workers.target.service
```

### 2. Start the Draft Worker

```bash
SPECSPLIT_DRAFT_MODEL_NAME=meta-llama/Llama-3.1-8B \
SPECSPLIT_DRAFT_DEVICE=cuda:1 \
    python -m specsplit.workers.draft.service
```

### 3. Run the Orchestrator

```bash
python -m specsplit.workers.orchestrator.client \
    --prompt "Explain the CAP theorem in distributed systems."
```

---

## Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install package in editable mode |
| `make proto` | Regenerate gRPC Python stubs |
| `make test` | Run unit tests only |
| `make test-all` | Run full test suite (unit + integration) |
| `make lint` | Lint with ruff |
| `make typecheck` | Static analysis with mypy |
| `make format` | Auto-format with ruff |
| `make clean` | Remove build artifacts and caches |

---

## Contributing

1. Create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure `make lint test typecheck` passes.
4. Open a PR with a clear description of the change.
