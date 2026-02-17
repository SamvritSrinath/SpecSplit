# Testing Guide

SpecSplit uses **pytest** with a clear separation between fast unit tests and
heavier integration tests that require model downloads or GPU access.

---

## Test Layout

```
tests/
├── conftest.py                     # Shared fixtures (configs, tmp dirs)
├── unit/                           # Fast tests — no models, no network
│   ├── test_serialization.py       #   Tensor ↔ list round-trips
│   ├── test_telemetry.py           #   Stopwatch + TelemetryLogger
│   ├── test_config.py              #   Pydantic config validation
│   ├── test_draft_engine.py        #   DraftEngine init + stub generation
│   └── test_target_engine.py       #   Session caching, rollback, verification
└── integration/                    # Requires transformers + torch
    ├── test_grpc_roundtrip.py      #   End-to-end gRPC smoke test (stubbed)
    └── test_exact_match.py         #   Speculative vs standard generation
```

---

## Running Tests

### All Tests

```bash
make test
# or equivalently:
pytest -v
```

### Unit Tests Only (fast, no GPU)

```bash
pytest tests/unit/ -v
```

### Integration Tests Only

```bash
pytest tests/integration/ -v -s --timeout=300
```

> Integration tests download models on first run (~1 GB for Qwen2.5-0.5B).
> Subsequent runs use the HuggingFace cache.

### Single Test File

```bash
pytest tests/unit/test_target_engine.py -v
```

### Single Test by Name

```bash
pytest -k "test_rollback_crops_tensors" -v
```

---

## Test Categories

### Unit Tests (`tests/unit/`)

**Goal:** Validate business logic in isolation.  No real models, no network, no
GPU.  These must run in < 5 seconds total.

| File | What It Tests | Key Fixtures |
|------|---------------|-------------|
| `test_serialization.py` | `tensor_to_token_ids` / `token_ids_to_tensor` round-trips, `softmax_with_temperature` | — |
| `test_telemetry.py` | `Stopwatch` precision, `TelemetryLogger` span collection + JSON export | `tmp_path` |
| `test_config.py` | Pydantic defaults, env var override, field validation | Monkeypatch |
| `test_draft_engine.py` | `DraftEngine` init, stub tree generation, `TokenNode.to_dict()` | `draft_config` |
| `test_target_engine.py` | Session create/reuse/evict, `rollback_cache` tensor cropping, verify with sessions | `target_engine`, `fake_kv_cache` |

### Integration Tests (`tests/integration/`)

**Goal:** End-to-end correctness with real model inference.  Marked with
`@pytest.mark.integration` so they can be selectively skipped in CI.

| File | What It Tests |
|------|---------------|
| `test_exact_match.py` | Loads Qwen2.5-0.5B as both draft and target. Asserts speculative decoding output is **byte-identical** to `model.generate()`. Tests multiple prompts, varying gamma (K=1,3,5,10), and edge cases. Uses mock gRPC stubs (no ports). |
| `test_grpc_roundtrip.py` | Smoke test for the gRPC service bindings (currently stubbed). |

---

## Writing New Tests

### Conventions

1. **File naming:** `test_<module_under_test>.py`
2. **Class naming:** `class Test<Feature>:` — groups related assertions.
3. **Fixtures over setup:** Use `conftest.py` fixtures, not `setUp/tearDown`.
4. **Docstrings on every test:** One line describing the assertion.
5. **Determinism:** Use `torch.manual_seed()` and `do_sample=False` for
   reproducible model outputs.

### Adding a Unit Test

```python
# tests/unit/test_my_module.py

from specsplit.core.my_module import my_function

class TestMyFunction:
    def test_basic_case(self):
        """my_function should return 42 for input 'hello'."""
        assert my_function("hello") == 42

    def test_edge_case(self):
        """my_function should raise ValueError on empty input."""
        with pytest.raises(ValueError):
            my_function("")
```

### Adding an Integration Test

```python
# tests/integration/test_new_feature.py

import pytest

try:
    from transformers import AutoModelForCausalLM
    _SKIP = False
except ImportError:
    _SKIP = True

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_SKIP, reason="transformers not installed"),
]

@pytest.fixture(scope="module")
def model():
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").eval()

class TestNewFeature:
    def test_something(self, model):
        """Feature X should produce Y when given Z."""
        ...
```

### Shared Fixtures (`conftest.py`)

The root `conftest.py` provides pre-built config fixtures:

| Fixture | Type | Description |
|---------|------|-------------|
| `draft_config` | `DraftWorkerConfig` | CPU-based draft config for testing |
| `target_config` | `TargetWorkerConfig` | CPU-based target config |
| `tmp_path` | `Path` | pytest built-in temp directory |

---

## Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.integration` | Requires model download / GPU |
| `@pytest.mark.slow` | Takes > 10 seconds |

Skip integration tests in CI without GPU:

```bash
pytest -m "not integration"
```

---

## Coverage

Generate a coverage report:

```bash
pytest --cov=specsplit --cov-report=html tests/
open htmlcov/index.html
```

---

## CI Integration

The test suite is designed to run in two stages:

1. **Fast gate** (`pytest tests/unit/ -x`) — runs on every push, < 10s.
2. **Full validation** (`pytest -v`) — runs on PR merge or nightly, includes
   model download + integration tests.

The `Makefile` target `make test` runs both stages sequentially.
