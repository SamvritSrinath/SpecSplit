# Orchestrator Worker (`specsplit.workers.orchestrator`)

The Orchestrator is the user-facing entry point. It coordinates the draft→verify
ping-pong loop between:

- the Draft Worker (`DraftService.GenerateDrafts`)
- the Target Worker (`TargetService.VerifyDrafts`)

and iterates until generation finishes.

## Responsibilities

- Parse user input (CLI) and load runtime config.
- Tokenize prompt text and manage iteration state across rounds.
- Call Draft + Target RPCs in the right order.
- Optionally run an overlapped async mode to hide network round-trip latency.

## Where the code lives

- `specsplit/workers/orchestrator/client.py`
  - CLI entry point + per-run reporting
- `specsplit/workers/orchestrator/pipeline.py`
  - `run_speculative_loop_async` and pipeline helpers
- `specsplit/workers/orchestrator/vocab_bridge.py`
  - `VocabBridge` for heterogeneous draft/target vocab ID alignment

## Entry points

- Run the orchestrator client:
  - `python -m specsplit.workers.orchestrator.client`

## Relevant tests

- `tests/unit/test_pipeline.py` (pipeline helpers + path utilities)
- `tests/unit/test_orchestrator_logging.py` (telemetry/log report behavior)

