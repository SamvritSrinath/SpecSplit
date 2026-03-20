# SpecSplit

Disaggregated speculative decoding (draft + verification) over gRPC.

## Getting Started

- Project guide: [`project_guide.md`](project_guide.md)
- Architecture overview: [`architecture.md`](architecture.md)

## Main Components

- Orchestrator (pipeline coordinator): [`specsplit/workers/orchestrator/README.md`](specsplit/workers/orchestrator/README.md)
- Draft Worker (generates speculative token trees): [`specsplit/workers/draft/README.md`](specsplit/workers/draft/README.md)
- Target Worker (verifies/accepts token trees): [`specsplit/workers/target/README.md`](specsplit/workers/target/README.md)

## Protocol

- `spec_decoding.proto`: [`proto/README.md`](proto/README.md)

## What to Read Next

- Core modules: [`specsplit/core/README.md`](specsplit/core/README.md)
- Benchmarks & experiments: [`benchmarks/README.md`](benchmarks/README.md)
- Tests & CI: [`tests/README.md`](tests/README.md)
- API reference: [`api/index.md`](api/index.md)

## GitHub Wiki

If you prefer a smaller “single page per topic” format, see the GitHub Wiki:
[SpecSplit.wiki](https://github.com/SamvritSrinath/SpecSplit.wiki.git).

