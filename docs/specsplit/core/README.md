# Core modules

Everything shared across workers lives in `specsplit/core/`. This includes:

- configuration (env-driven, Pydantic validated)
- serialization helpers at the gRPC boundary
- telemetry/spans for latency breakdowns
- verification logic (greedy and stochastic tree acceptance)
- small utilities like KV-cache helpers and model loading glue

## Main modules

- `specsplit/core/config.py`
  - Pydantic settings with the `SPECSPLIT_` environment variable prefix
- `specsplit/core/serialization.py`
  - tensor ↔ list conversions for gRPC message payloads
- `specsplit/core/telemetry.py`
  - high-resolution timing + span export (JSON)
- `specsplit/core/verification.py`
  - greedy and stochastic verification over token trees
- `specsplit/core/model_loading.py`
  - model/tokenizer load and device placement helpers
- `specsplit/core/cache_utils.py`
  - small cache manipulation utilities used by the target engine

## Tests

- `tests/unit/test_config.py`
- `tests/unit/test_serialization.py`
- `tests/unit/test_telemetry.py`
- `tests/unit/test_verification.py`
- (indirect coverage) `tests/unit/test_target_engine.py`, `tests/unit/test_draft_engine.py`

