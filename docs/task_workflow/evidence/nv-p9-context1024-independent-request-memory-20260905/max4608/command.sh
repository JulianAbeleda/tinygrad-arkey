#!/bin/bash
set -euo pipefail
exec flock -n /tmp/gpu-bench.lock timeout 1200 env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/decode/decode_runtime_overhead.py \
  --ckpts 1024 --max-context 4608 --nmeas 10 --reps 3 --warmup-decode 3 \
  --skip-dispatch-diagnostic --gpu-state --request-scoped-prewarm \
  --checkpoint-dir /tmp/nv-p9-context1024-r3-checkpoint \
  --graph-census-dir /tmp/nv-p9-context1024-r3-census \
  --out /tmp/nv-p9-context1024-r3.json
