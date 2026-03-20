# Benchmarks

The benchmark harness runs the full SpecSplit loop over a prompt dataset and
emits per-round and aggregated metrics.

## Where the code lives

- `benchmarks/runner.py`: main benchmark entry point (writes CSV metrics)
- `benchmarks/analyze_results.py`: post-processing for plots/tables
- `benchmarks/run.sh`: convenience wrapper (optional)

## What it produces

- `benchmarks/results/<run>/summary.csv`: per-gamma (or per-config) aggregation
- `benchmarks/results/<run>/per_round.csv`: per-request, per-round details

## Run it

See `docs/experiments.md` for the canonical command templates. A typical run is:

```bash
python benchmarks/runner.py \
  --prompts data/prompts.jsonl \
  --results-dir benchmarks/results/custom \
  --draft-addr localhost:50051 \
  --target-addr localhost:50052 \
  --tokenizer gpt2 \
  --gamma 5
```

## Related docs

- `docs/experiments.md`: parameters, gamma sweep semantics, and metrics reference

