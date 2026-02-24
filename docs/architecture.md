# SpecSplit Architecture

## Overview

SpecSplit is a disaggregated speculative decoding system that splits LLM inference
across two networked GPU workers:

- **Draft Worker** ("The Hare") — a small, fast model that speculatively generates
  token trees.
- **Target Worker** ("The Tortoise") — a large, accurate model that verifies draft
  trees using tree-attention.

The system is coordinated by an **Orchestrator** that manages the asynchronous
ping-pong loop between workers.

## System Diagram

```
                          ┌─────────────────────────┐
                          │      Orchestrator        │
                          │  (Pipeline Coordinator)  │
                          └────┬───────────────┬────┘
                     prompt +  │               │  accepted tokens
                     context   │               │  + correction
                               ▼               ▼
                ┌──────────────────┐   ┌──────────────────┐
                │   Draft Worker    │   │  Target Worker    │
                │                  │   │                  │
                │  ┌────────────┐  │   │  ┌────────────┐  │
                │  │ Small LLM  │  │   │  │ Large LLM  │  │
                │  │ (e.g. GPT-2)│  │   │  │(e.g. Llama)│  │
                │  └────────────┘  │   │  └────────────┘  │
                │                  │   │                  │
                │  KV Cache ✓      │   │  Session KV ✓    │
                │  Cheap GPU       │   │  Expensive GPU   │
                └──────────────────┘   └──────────────────┘
                         ▲                       ▲
                         │     gRPC (proto3)      │
                         └───────────────────────┘
```

## Data Flow

1. **User prompt** → Orchestrator tokenizes and sends to Draft Worker.
2. **Draft Worker** generates a speculative token tree of depth K using
   autoregressive sampling with a local KV cache.
3. **Draft tree + session_id** → sent to Target Worker via gRPC.
4. **Target Worker** performs a single batched forward pass with tree attention
   to score all candidate paths simultaneously.  If a `session_id` is provided,
   the existing KV cache for that session is reused and rolled back to the
   accepted prefix after verification.
5. **Verification** determines the longest accepted path:
   - **Greedy** (temperature = 0): accept when `argmax(p_target) == draft token`.
   - **Stochastic** (temperature > 0): rejection sampling over all branches via DFS;
     at each node accept if `p_target ≥ p_draft`, else accept with probability
     `p_target / p_draft`; the longest accepted path across the tree is chosen.
6. **Accepted tokens + optional correction** → returned to Orchestrator.
7. Orchestrator appends accepted tokens and loops back to step 2.
8. When generation completes, the Orchestrator calls `EndSession` to free
   the Target Worker's KV cache.

## Protocol Design

The gRPC protocol (`spec_decoding.proto`) defines:

| Service          | RPC               | Purpose                                |
|------------------|--------------------|----------------------------------------|
| `DraftService`   | `GenerateDrafts`   | Generate speculative token trees       |
| `DraftService`   | `Ping`             | Health check                           |
| `TargetService`  | `VerifyDrafts`     | Verify draft trees (with session KV cache) |
| `TargetService`  | `EndSession`       | Release a session's KV cache           |
| `TargetService`  | `Ping`             | Health check                           |

### Key Messages

- **`TokenNode`** — recursive tree node: `{token_id, log_prob, children[]}`
- **`TelemetryMetadata`** — per-RPC timing: `{span_id, wall_time_ms, model_time_ms, ...}`

## Component Responsibilities

| Component           | Responsibility                                          |
|---------------------|---------------------------------------------------------|
| `core/config.py`    | Pydantic settings with env var override (`SPECSPLIT_*`) |
| `core/serialization.py` | Tensor ↔ list conversion at gRPC boundary          |
| `core/telemetry.py` | Nanosecond-precision timing + JSON span export          |
| `core/verification.py` | Greedy and stochastic tree verification (argmax or rejection sampling + DFS) |
| `workers/draft/`    | Stateful draft generation with KV cache management      |
| `workers/target/engine.py` | Session-based KV-cached tree-attention verification |
| `workers/target/tree_attn.py` | Custom tree attention mask + position ID construction |
| `workers/target/kv_cache.py` | Pre-allocated static KV cache (O(1) rollback)     |
| `workers/orchestrator/client.py` | CLI entry point + synchronous pipeline         |
| `workers/orchestrator/pipeline.py` | Async overlapped draft→verify with speculation |

## Design Decisions

1. **Disaggregated architecture** — Draft and Target workers run on separate
   machines/GPUs, connected over the network. This allows independent scaling
   and heterogeneous hardware.

2. **Session-based KV caching** — The Target Worker maintains a per-session
   KV cache (`session_id → KVCacheState`) to avoid prompt recomputation across
   verification rounds. Sessions are LRU-evicted at `max_sessions`, and caches
   are freed explicitly via the `EndSession` RPC.

3. **Pre-allocated static KV cache** — `StaticKVCache` in `kv_cache.py` avoids
   `torch.cat` reallocation by pre-allocating key/value buffers and using slice
   assignment. Rollback is O(1) — a single pointer update.

4. **Tree-structured speculation** — Instead of linear draft sequences, we
   generate trees (branching factor > 1) to explore multiple hypotheses in
   parallel, increasing acceptance rates. Custom tree attention masks
   (`tree_attn.py`) ensure each node only attends to its ancestors.

5. **Greedy verification math** — `verification.py` runs `torch.argmax` and
   `torch.eq` on-device, then uses iterative DFS on a small boolean mask to
   find the longest accepted path. Only the final result is synced to CPU.

6. **Async overlapped pipeline** — `pipeline.py` uses `asyncio.gather` to
   speculatively draft round N+1 while verifying round N. On speculation hit,
   a full gRPC round-trip is saved.

7. **Pydantic configuration** — All settings are type-safe, validated, and
   overridable via environment variables for easy deployment configuration.

8. **Structured telemetry** — Every RPC call generates a span with nanosecond
   timing, enabling distributed tracing and performance analysis.
