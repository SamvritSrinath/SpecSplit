#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SpecSplit Benchmark Wrapper
# =============================================================================
# Canonical benchmark entrypoint.
# Usage:
#   bash benchmarks/run.sh [mode]
#
# Modes:
#   smoke       Quick 5-prompt sanity check (default gamma=5)
#   single      Single gamma run over full dataset (GAMMA env var)
#   sweep       Full gamma sweep: 1 3 5 8 12
#   category    Sweep one category (CATEGORY env var)
#   case_study  One fixed case-study prompt across gamma sweep
#   ablation    Vary max-output-tokens at fixed gamma=5
# =============================================================================

# Configuration (set as env vars to override)
DRAFT_ADDR="${SPECSPLIT_ORCH_DRAFT_ADDRESS:-}"
TARGET_ADDR="${SPECSPLIT_ORCH_TARGET_ADDRESS:-}"
TOKENIZER="${SPECSPLIT_ORCH_TOKENIZER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

PROMPTS_FILE="${PROMPTS_FILE:-benchmarks/specsplit_bench.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-benchmarks/results}"

MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128}"
DRAFT_TEMP="${DRAFT_TEMP:-0.0}"
VERIFY_TEMP="${VERIFY_TEMP:-0.0}"
REQUEST_DELAY="${REQUEST_DELAY:-1.0}"
TIMEOUT="${TIMEOUT:-300}"
# Subsample size used by default for non-smoke modes.
# Set BENCH_LIMIT=0 to run full dataset.
BENCH_LIMIT="${BENCH_LIMIT:-0}"
# Balanced subsample: prompts per category (0 disables).
BENCH_PER_CATEGORY="${BENCH_PER_CATEGORY:-2}"

MODE="${1:-smoke}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

validate_env() {
  log "Validating environment..."
  [[ -n "$DRAFT_ADDR" ]] || die "SPECSPLIT_ORCH_DRAFT_ADDRESS is required"
  [[ -n "$TARGET_ADDR" ]] || die "SPECSPLIT_ORCH_TARGET_ADDRESS is required"
  [[ -n "$TOKENIZER" ]] || die "SPECSPLIT_ORCH_TOKENIZER_MODEL is required"
  [[ -f "$PROMPTS_FILE" ]] || die "Prompts file not found: $PROMPTS_FILE"
  python3 -c "import specsplit" 2>/dev/null || die "specsplit package not importable. Activate your venv?"
  log "  Draft  : $DRAFT_ADDR"
  log "  Target : $TARGET_ADDR"
  log "  Model  : $TOKENIZER"
  log "  Prompts: $PROMPTS_FILE"
}

ping_workers() {
  log "Pinging workers..."
  local ping_script
  ping_script=$(cat <<'PYEOF'
import asyncio, os, sys

async def ping():
    import grpc
    from specsplit.proto import spec_decoding_pb2 as pb2
    from specsplit.proto import spec_decoding_pb2_grpc as stubs

    # Use service-specific stubs when available; fall back to discovered stubs.
    draft_stub_cls = getattr(stubs, "DraftServiceStub", None)
    target_stub_cls = getattr(stubs, "TargetServiceStub", None)
    fallback = [getattr(stubs, n) for n in dir(stubs) if n.endswith("Stub")]

    if draft_stub_cls is None and fallback:
        draft_stub_cls = fallback[0]
    if target_stub_cls is None and fallback:
        target_stub_cls = fallback[-1]
    if draft_stub_cls is None or target_stub_cls is None:
        raise RuntimeError("No usable gRPC Stub classes found in spec_decoding_pb2_grpc")

    addrs = [os.environ["SPECSPLIT_ORCH_DRAFT_ADDRESS"], os.environ["SPECSPLIT_ORCH_TARGET_ADDRESS"]]
    names = ["draft", "target"]
    stub_classes = [draft_stub_cls, target_stub_cls]
    ok = True
    for addr, name, stub_cls in zip(addrs, names, stub_classes):
        try:
            async with grpc.aio.insecure_channel(addr) as ch:
                stub = stub_cls(ch)
                resp = await asyncio.wait_for(stub.Ping(pb2.PingRequest()), timeout=10)
                print(f"  [OK] {name} @ {addr} -> status={resp.status}")
        except Exception as e:
            print(f"  [FAIL] {name} @ {addr}: {e}", file=sys.stderr)
            ok = False
    sys.exit(0 if ok else 1)

asyncio.run(ping())
PYEOF
  )
  if python3 -c "$ping_script"; then
    log "Both workers reachable."
  else
    die "Worker connectivity check failed. Check addresses and worker status."
  fi
}

common_args=(
  --prompts "$PROMPTS_FILE"
  --results-dir "$RESULTS_DIR"
  --draft-addr "$DRAFT_ADDR"
  --target-addr "$TARGET_ADDR"
  --tokenizer "$TOKENIZER"
  --max-rounds "$MAX_ROUNDS"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --draft-temperature "$DRAFT_TEMP"
  --verify-temperature "$VERIFY_TEMP"
  --delay "$REQUEST_DELAY"
  --timeout "$TIMEOUT"
)

run_smoke() {
  log "MODE: smoke (5 prompts, gamma=5)"
  python3 benchmarks/runner.py "${common_args[@]}" --gamma 5 --limit 5 --verbose
}

run_single() {
  local gamma="${GAMMA:-5}"
  if [[ "$BENCH_PER_CATEGORY" -gt 0 ]]; then
    log "MODE: single (gamma=$gamma, per-category=$BENCH_PER_CATEGORY)"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma "$gamma" --per-category-limit "$BENCH_PER_CATEGORY" --resume
  elif [[ "$BENCH_LIMIT" -gt 0 ]]; then
    log "MODE: single (gamma=$gamma, limit=$BENCH_LIMIT)"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma "$gamma" --limit "$BENCH_LIMIT" --resume
  else
    log "MODE: single (gamma=$gamma, full dataset)"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma "$gamma" --resume
  fi
}

run_sweep() {
  if [[ "$BENCH_PER_CATEGORY" -gt 0 ]]; then
    local category_count
    category_count=$(python3 - <<'PY'
import json
from pathlib import Path
path = Path("benchmarks/specsplit_bench.jsonl")
cats = set()
for line in path.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        cats.add(json.loads(line).get("category", "unknown"))
    except Exception:
        pass
print(len(cats))
PY
)
    log "MODE: sweep (gamma=1,3,5,8,12, per-category=$BENCH_PER_CATEGORY)"
    log "Estimated runs: $(( category_count * BENCH_PER_CATEGORY )) prompts x 5 gammas"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma 1 3 5 8 12 --per-category-limit "$BENCH_PER_CATEGORY" --resume
  elif [[ "$BENCH_LIMIT" -gt 0 ]]; then
    log "MODE: sweep (gamma=1,3,5,8,12, limit=$BENCH_LIMIT)"
    log "Estimated runs: ${BENCH_LIMIT} prompts x 5 gammas"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma 1 3 5 8 12 --limit "$BENCH_LIMIT" --resume
  else
    log "MODE: sweep (gamma=1,3,5,8,12, full dataset)"
    log "Estimated runs: $(wc -l < "$PROMPTS_FILE") prompts x 5 gammas"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma 1 3 5 8 12 --resume
  fi
}

run_category() {
  local cat="${CATEGORY:-factual_qa}"
  if [[ "$BENCH_PER_CATEGORY" -gt 0 ]]; then
    log "MODE: category (category=$cat, per-category=$BENCH_PER_CATEGORY)"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma 1 3 5 8 12 --categories "$cat" --per-category-limit "$BENCH_PER_CATEGORY" --resume
  elif [[ "$BENCH_LIMIT" -gt 0 ]]; then
    log "MODE: category (category=$cat, limit=$BENCH_LIMIT)"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma 1 3 5 8 12 --categories "$cat" --limit "$BENCH_LIMIT" --resume
  else
    log "MODE: category (category=$cat, full dataset)"
    python3 benchmarks/runner.py "${common_args[@]}" --gamma 1 3 5 8 12 --categories "$cat" --resume
  fi
}

run_case_study() {
  log "MODE: case_study (fixed capitals prompt, gamma=1,3,5,8,12)"
  local tmp_prompts
  tmp_prompts=$(mktemp /tmp/case_study_XXXXXX.jsonl)
  echo '{"id":"cs-capitals","category":"case_study","prompt":"What is the Capital of France?"}' > "$tmp_prompts"
  python3 benchmarks/runner.py "${common_args[@]}" \
    --prompts "$tmp_prompts" \
    --gamma 1 3 5 8 12 \
    --max-rounds 20 \
    --max-output-tokens 96 \
    --results-dir "${RESULTS_DIR}/case_study"
  rm -f "$tmp_prompts"
}

run_ablation() {
  log "MODE: ablation (vary max-output-tokens with gamma=5)"
  for tokens in 64 128 256 512; do
    local out_dir="${RESULTS_DIR}/ablation_tokens_${tokens}"
    log "  Running max-output-tokens=$tokens"
    python3 benchmarks/runner.py "${common_args[@]}" \
      --results-dir "$out_dir" \
      --gamma 5 \
      --max-output-tokens "$tokens" \
      --categories factual_qa \
      --limit 10
  done
}

validate_env

export SPECSPLIT_ORCH_DRAFT_ADDRESS="$DRAFT_ADDR"
export SPECSPLIT_ORCH_TARGET_ADDRESS="$TARGET_ADDR"
export SPECSPLIT_ORCH_TOKENIZER_MODEL="$TOKENIZER"

if [[ "${SKIP_PING:-0}" != "1" ]]; then
  ping_workers
fi

RUN_START=$(date +%s)
case "$MODE" in
  smoke) run_smoke ;;
  single) run_single ;;
  sweep) run_sweep ;;
  category) run_category ;;
  case_study) run_case_study ;;
  ablation) run_ablation ;;
  *) die "Unknown mode '$MODE' (smoke|single|sweep|category|case_study|ablation)" ;;
esac
RUN_END=$(date +%s)
log "Benchmark complete. Total elapsed: $(( RUN_END - RUN_START ))s"

log "Analyze with: python3 benchmarks/analyze_results.py --results-dir $RESULTS_DIR"
