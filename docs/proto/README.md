# Protocol: `spec_decoding.proto`

The entire draft→verify loop is defined by the gRPC services and message schema in
`specsplit/proto/spec_decoding.proto`.

## Where it lives

- `specsplit/proto/spec_decoding.proto`

## Why this protocol exists

It lets the Draft Worker and Target Worker exchange a compact representation of
speculation candidates:

- the prompt/context is represented as `repeated int32` token IDs
- the speculative candidates are represented as a tree of `TokenNode`s
- the Target Worker can optionally reuse per-session KV cache state via
  `session_id`

This keeps the network payload small and makes verification latency dominated
by the Target model forward pass (rather than serialization cost).

## Services

### `DraftService` (Draft Worker)

- `GenerateDrafts (DraftRequest) returns (DraftResponse)`
- `Ping (PingRequest) returns (PingResponse)` (health/readiness)

### `TargetService` (Target Worker)

- `VerifyDrafts (VerifyRequest) returns (VerifyResponse)`
  - supports per-session KV reuse when `session_id` is provided
- `EndSession (EndSessionRequest) returns (EndSessionResponse)`
  - releases GPU KV cache for a session
- `Ping (PingRequest) returns (PingResponse)` (health/readiness)

## Key messages

### `TokenNode`

Single node in the speculative tree:

- `token_id`: vocabulary index of the candidate token
- `log_prob`: log-probability assigned by the draft model
- `children`: child candidate nodes (branching)
- `top_k_token_ids` / `top_k_probs`: optional Top-K distribution data used for
  full-vocabulary residual computations.

### `DraftRequest` / `DraftResponse`

- `DraftRequest`
  - `prompt_token_ids`: current prompt/context token IDs
  - `max_draft_len`: tree depth (K / gamma)
  - `num_beams`: branching factor per level
  - `temperature`: `0` means greedy; `>0` enables sampling
  - `reset_cache`: clear draft KV cache before generating (used after misses)
  - `session_id`: session identifier for thread-safe KV reuse
- `DraftResponse`
  - `draft_tree`: generated tree candidates (forest at root-level)
  - `telemetry`: server-side timing metadata

### `VerifyRequest` / `VerifyResponse`

- `VerifyRequest`
  - `draft_tree`: draft candidates to verify
  - `session_id`: KV cache reuse key (empty means stateless verification)
  - `temperature`: `0` for greedy verification; `>0` for stochastic verification
  - `expected_prefix_length`: orchestrator’s expected accepted prefix length
- `VerifyResponse`
  - `accepted_token_ids`: longest accepted prefix from the tree
  - `correction_token_id` + `has_correction`: correction token when draft
    rejection occurs
  - `cache_hit`: whether session KV cache was reused
  - `telemetry`: server-side timing metadata

## Tests / CI hooks

- The CI workflow job `proto-check` compiles `spec_decoding.proto` and verifies the
  generated stub files exist.
- Unit tests validate key building blocks around token-tree transformations and
  verification math (see `tests/unit/`).

