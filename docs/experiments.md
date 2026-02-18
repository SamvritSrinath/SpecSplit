# Running Experiments

This guide explains how to use the SpecSplit benchmarking harness to run
reproducible experiments, sweep hyper-parameters, and collect per-request
telemetry.

---

## Prerequisites

1. **Both workers running** — see [Project Guide](project_guide.md) for startup.
2. **A prompt dataset** — JSONL file, one JSON object per line.
3. **Python environment** — `pip install -e ".[dev]"` completed.

---

## Dataset Format

The benchmark script accepts JSONL.  Each line needs at minimum a `"prompt"` field:

```jsonl
{"prompt": "Explain quantum computing to a 5-year-old."}
{"prompt": "Write a Python function that merges two sorted lists.", "id": "code-01"}
{"prompt": "What is the difference between TCP and UDP?"}
```

**ShareGPT format** is also supported — the script auto-extracts the first human
turn from `"conversations"`:

```jsonl
{"conversations": [{"from": "human", "value": "What is RLHF?"}]}
```

> **Tip:** For quick smoke tests, create a 5-line JSONL.  For publication-grade
> results, use a 500+ prompt slice of ShareGPT or LMSYS-Chat-1M.

---

## Basic Run

```bash
python scripts/benchmark_run.py \
    --dataset data/prompts.jsonl \
    --output results/baseline.csv
```

This runs every prompt with the default Gamma (K=5) and writes per-request
metrics to CSV.

---

## Gamma Sweep

Gamma (K) is the draft tree depth — the number of tokens the draft model
speculates per round.  It directly controls the throughput/acceptance-rate
trade-off.  This corresponds to `DraftWorkerConfig.max_draft_tokens` and
interacts with the tree attention mask (`tree_attn.py`) and the static KV
cache rollback depth (`kv_cache.py`).

Sweep multiple values in a single invocation:

```bash
python scripts/benchmark_run.py \
    --dataset data/prompts.jsonl \
    --output results/gamma_sweep.csv \
    --gamma 1 3 5 8 12
```

The script runs the full dataset once per gamma value and writes all results to
a single CSV.  The summary table printed at the end is grouped by gamma:

```
==========================================================================================
BENCHMARK SUMMARY
==========================================================================================
 Gamma   Reqs    Avg TTFT    Avg TPOT   Avg Accept   Avg NetIdle   Avg Latency
------------------------------------------------------------------------------------------
     1     50     12.34ms      3.21ms       100.00%       4.12ms       161.23ms
     5     50     14.56ms      2.01ms        87.30%       5.67ms       102.45ms
    12     50     18.90ms      1.45ms        72.10%       7.89ms        74.32ms
==========================================================================================
```

---

## Metrics Reference

Each row in the output CSV contains:

| Column | Unit | Description |
|--------|------|-------------|
| `request_id` | — | Unique identifier (from dataset `"id"` or auto-generated) |
| `gamma` | int | Draft depth (K) for this run |
| `prompt_length` | tokens | Estimated prompt token count |
| `generated_tokens` | tokens | Number of output tokens produced |
| `ttft_ms` | ms | **Time-to-First-Token** — latency from request start to the first token |
| `tpot_ms` | ms | **Time-Per-Output-Token** — average inter-token latency |
| `average_acceptance_rate` | 0–1 | Mean fraction of draft tokens accepted per round |
| `total_network_idle_ms` | ms | Cumulative gRPC round-trip overhead |
| `total_latency_ms` | ms | End-to-end wall-clock time for the request |
| `num_rounds` | int | Number of draft→verify iterations |

---

## Overriding Configuration

```bash
# Limit generation length
python scripts/benchmark_run.py \
    --dataset data/prompts.jsonl \
    --max-output-tokens 128 \
    --max-rounds 10

# Point at custom worker addresses
SPECSPLIT_ORCH_DRAFT_ADDRESS=gpu1:50051 \
SPECSPLIT_ORCH_TARGET_ADDRESS=gpu2:50052 \
    python scripts/benchmark_run.py --dataset data/prompts.jsonl
```

---

## Analyzing Results

The CSV can be loaded directly into pandas, a Jupyter notebook, or any
spreadsheet:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/gamma_sweep.csv")

# TPOT vs Gamma
df.groupby("gamma")["tpot_ms"].mean().plot(kind="bar", title="Avg TPOT by Gamma")
plt.ylabel("ms / token")
plt.tight_layout()
plt.savefig("results/tpot_vs_gamma.png")
```

### Key Plots to Produce

1. **TPOT vs Gamma** — shows where throughput gains saturate.
2. **Acceptance Rate vs Gamma** — reveals the quality ceiling of the draft model.
3. **TTFT vs Gamma** — shows first-token latency cost of deeper speculation.
4. **Network Idle Fraction** — `total_network_idle_ms / total_latency_ms` highlights
   whether the bottleneck is compute or network.

---

## Reproducibility Checklist

- [ ] Pin model versions in your experiment log (e.g. `Qwen/Qwen2.5-0.5B`)
- [ ] Record GPU type and driver version (`nvidia-smi`)
- [ ] Use the same dataset JSONL across all runs
- [ ] Set `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for determinism
- [ ] Log the exact command used in each CSV filename or beside it

---

## Advanced: Async Overlapped Pipeline

The `pipeline.py` module implements an **async overlapped** execution mode
where draft round N+1 is speculatively started while round N is being verified:

```python
from specsplit.workers.orchestrator.pipeline import run_speculative_loop_async

result = await run_speculative_loop_async(
    draft_stub, target_stub, prompt_ids, config
)
print(result.speculation_hits, result.speculation_misses)
```

The benchmark script does not yet use this mode, but it can be integrated
for latency-focused experiments.  Track `speculation_hits / total_rounds`
to quantify the pipeline overlap benefit.
