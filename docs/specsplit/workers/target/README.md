# Target Worker (`specsplit.workers.target`)

The Target Worker ("The Tortoise") runs the large, accurate model and verifies
draft speculative token trees using tree-attention.

It returns the longest accepted token prefix (and an optional correction token
when the draft diverges from the target distribution).

## Responsibilities

- Load the configured target model and tokenizer.
- Maintain per-session KV-cache state (`session_id`) to avoid prompt
  recomputation across verification rounds.
- Perform batched tree-attention to score the draft tree efficiently.
- Roll back/crop cached KV state to the accepted prefix.

## Where the code lives

- `specsplit/workers/target/engine.py`
  - `TargetEngine`: verification + session cache orchestration
  - `VerificationResult`: accepted/correction outputs + acceptance rate
- `specsplit/workers/target/tree_attn.py`
  - tree-attention mask construction and position/id logic
- `specsplit/workers/target/kv_cache.py`
  - `StaticKVCache` and rollback/compaction operations
- `specsplit/workers/target/service.py`
  - gRPC handlers for `TargetService`

## Entry points

- Run the worker:
  - `python -m specsplit.workers.target.service`

## Relevant tests

- `tests/unit/test_target_engine.py` (session caching + rollback)
- `tests/unit/test_tree_attn.py` (mask + tree attention helpers)
- `tests/unit/test_verification.py` (greedy/stochastic verification math)

