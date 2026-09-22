#!/bin/bash
set -euo pipefail
exec flock -n /tmp/gpu-bench.lock timeout 1800 env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/decode/decode_runtime_overhead.py \
  --ckpts 128,256,512,1024,2048,4096 --max-context 4608 --nmeas 10 --reps 3 --warmup-decode 3 \
  --skip-dispatch-diagnostic --gpu-state \
  --checkpoint-dir /tmp/nv-p9-context-sweep-r1-checkpoints \
  --graph-census-dir /tmp/nv-p9-context-sweep-r1-census \
  --out /tmp/nv-p9-context-sweep-r1.json
