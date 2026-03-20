# Draft Worker (`specsplit.workers.draft`)

The Draft Worker ("The Hare") runs a small LLM and generates speculative token
trees of depth `K` (aka `gamma`).

Those candidates are sent to the Target Worker for verification.

## Responsibilities

- Load the configured draft model and tokenizer.
- Maintain per-session KV-cache state so concurrent RPC requests don't share
  mutable cache data.
- Generate a speculative tree and return it as protobuf `TokenNode`s.

## Where the code lives

- `specsplit/workers/draft/engine.py`
  - `DraftEngine`: model/KV cache management + tree generation
  - `TokenNode`: in-memory representation of a draft tree node
- `specsplit/workers/draft/service.py`
  - `DraftServiceServicer`: gRPC handlers for `DraftService`

## Session KV caching

When `session_id` is provided in `DraftRequest`, the engine keeps an isolated
cache state per session and uses it to extend generation across rounds.

When the request sets `reset_cache=True`, the draft cache is cleared for the
session (used after speculation misses).

## Entry points

- Run the worker:
  - `python -m specsplit.workers.draft.service`

## Relevant tests

- `tests/unit/test_draft_engine.py` (draft initialization + tree generation basics)

