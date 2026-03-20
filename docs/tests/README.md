# Tests

This repo uses `pytest` with a deliberate split between fast unit tests and
heavier integration tests.

## Layout

- `tests/unit/`: CPU-friendly, no real model downloads (CI unit gate)
- `tests/integration/`: end-to-end checks that may download models / require GPU

## Running

- Unit tests:
  - `pytest tests/unit/ -v`
- Full suite:
  - `pytest tests/ -v`

## CI coverage

- CI always runs `pytest tests/unit/ -v --tb=short -x` (fast gate).
- Integration tests can be marked with `@pytest.mark.integration` and skipped
  when resources are limited.

## Canonical guide

For detailed conventions (docstrings on every test, determinism guidance, etc.),
see `docs/testing.md`.

