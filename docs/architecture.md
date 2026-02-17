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
                │  KV Cache ✓      │   │  Stateless ✗     │
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
3. **Draft tree** → sent to Target Worker via gRPC.
4. **Target Worker** performs a single batched forward pass with tree attention
   to score all candidate paths simultaneously.
5. **Rejection sampling** determines the longest accepted prefix:
   - If `p_target(x) ≥ p_draft(x)`: token accepted.
   - Otherwise: accepted with probability `p_target(x) / p_draft(x)`.
6. **Accepted tokens + optional correction** → returned to Orchestrator.
7. Orchestrator appends accepted tokens and loops back to step 2.

## Protocol Design

The gRPC protocol (`spec_decoding.proto`) defines:

| Service          | RPC               | Purpose                                |
|------------------|--------------------|----------------------------------------|
| `DraftService`   | `GenerateDrafts`   | Generate speculative token trees       |
| `DraftService`   | `Ping`             | Health check                           |
| `TargetService`  | `VerifyDrafts`     | Verify draft trees via tree attention   |
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
| `workers/draft/`    | Stateful draft generation with KV cache management      |
| `workers/target/`   | Stateless tree-attention verification                   |
| `workers/orchestrator/` | Async pipeline control + CLI entry point            |

## Design Decisions

1. **Disaggregated architecture** — Draft and Target workers run on separate
   machines/GPUs, connected over the network. This allows independent scaling
   and heterogeneous hardware.

2. **Stateless Target Worker** — No KV cache is maintained between calls,
   making the Target Worker horizontally scalable and failure-resilient.

3. **Tree-structured speculation** — Instead of linear draft sequences, we
   generate trees (branching factor > 1) to explore multiple hypotheses in
   parallel, increasing acceptance rates.

4. **Pydantic configuration** — All settings are type-safe, validated, and
   overridable via environment variables for easy deployment configuration.

5. **Structured telemetry** — Every RPC call generates a span with nanosecond
   timing, enabling distributed tracing and performance analysis.
