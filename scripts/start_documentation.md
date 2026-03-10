# Start the Target Worker (large model, expensive GPU)
```
SPECSPLIT_TARGET_MODEL_NAME=/home/shared/models/Meta-Llama-3.1-70B-BNB-NF4-BF16 \
SPECSPLIT_TARGET_GRPC_PORT=50052 \
    python -m specsplit.workers.target.service
```
# Start the Orchestrator
```
# Run a single prompt through the pipeline
SPECSPLIT_ORCH_DRAFT_ADDRESS=0.tcp.us-cal-1.ngrok.io:14091 \
SPECSPLIT_ORCH_TARGET_ADDRESS=8.tcp.us-cal-1.ngrok.io:12462 \
    python -m specsplit.workers.orchestrator.client \
        --prompt "If I could" \
        --max-rounds 5

export SPECSPLIT_ORCH_DRAFT_ADDRESS=0.tcp.us-cal-1.ngrok.io:14091
export SPECSPLIT_ORCH_TARGET_ADDRESS=8.tcp.us-cal-1.ngrok.io:12462
export SPECSPLIT_ORCH_TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
python -m specsplit.workers.orchestrator.client --prompt $"Q: What can I do when I am struggling to find the right words?\nA:" --max-rounds 20 \
  --max-output-tokens 96 \
  --max-draft-tokens 3 \
  --draft-temperature 0.15 \
  --verify-temperature 0.15 \
  --use-target-cache
```

# Start the target worker without interrupt
```
scripts/manage_remote_worker.sh start ~/specsplit-target.env
exit
```
# Manage the target worker
```
scripts/manage_remote_worker.sh status ~/specsplit-target.env
scripts/manage_remote_worker.sh logs ~/specsplit-target.env
scripts/manage_remote_worker.sh stop ~/specsplit-target.env
scripts/manage_remote_worker.sh update ~/specsplit-target.env
```
