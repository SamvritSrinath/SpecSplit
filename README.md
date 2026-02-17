# SpecSplit

**Disaggregated Speculative Decoding** — a distributed systems research framework for accelerating LLM inference by splitting draft and target model computation across networked GPUs.

---

## Architecture

```
┌─────────────────┐        gRPC         ┌─────────────────┐
│   Draft Worker   │◄──────────────────►│  Target Worker   │
│  (Small LLM)     │   token trees +    │  (Large LLM)     │
│  cheap GPU        │   verifications    │  expensive GPU    │
└────────┬────────┘                     └────────┬────────┘
         │                                        │
         └──────────────┐  ┌─────────────────────┘
                        ▼  ▼
                  ┌──────────────┐
                  │ Orchestrator  │
                  │ (async loop)  │
                  └──────────────┘
```

**Draft Worker** generates speculative token trees using a small, fast model.  
**Target Worker** verifies those trees using the full-size model with tree attention.  
**Orchestrator** manages the asynchronous ping-pong pipeline and aggregates results.

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/<your-org>/SpecSplit.git
cd SpecSplit
make install            # pip install -e ".[dev]"

# 2. Generate gRPC stubs from protobuf
make proto

# 3. Run tests
make test               # unit tests only
make test-all           # unit + integration

# 4. Lint & typecheck
make lint
make typecheck
```

## Development Commands

| Command          | Description                                     |
|------------------|-------------------------------------------------|
| `make install`   | Editable install with dev dependencies          |
| `make proto`     | Generate Python stubs from `.proto` definitions |
| `make test`      | Run unit tests                                  |
| `make test-all`  | Run all tests (unit + integration)              |
| `make test-cov`  | Run tests with coverage report                  |
| `make lint`      | Run ruff linter                                 |
| `make typecheck` | Run mypy type checker                           |
| `make format`    | Auto-format with ruff                           |
| `make clean`     | Remove caches and generated files               |

## Project Structure

```
specsplit/
├── proto/              # gRPC protobuf definitions
├── core/               # Shared utilities (serialization, telemetry, config)
├── workers/
│   ├── draft/          # Draft Worker — speculative generation
│   ├── target/         # Target Worker — tree-attention verification
│   └── orchestrator/   # Pipeline coordinator
tests/
├── unit/               # Fast, mocked tests
└── integration/        # Network / model tests
scripts/                # Benchmarking & utilities
docs/                   # Architecture documentation
```

## License

MIT — see [LICENSE](./LICENSE).
