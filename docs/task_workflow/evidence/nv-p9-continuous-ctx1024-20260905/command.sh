#!/bin/bash
set -euo pipefail
exec flock -n /tmp/gpu-bench.lock timeout 1200 env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/decode/decode_continuous_context_coverage.py \
  --depth 1024 --max-context 1536 --nmeas 10 --reps 3 --warmup-decode 3 --gpu-state --request-scoped-prewarm \
  --census-out /tmp/nv-p9-cont1024-m1536-r5-census.json --out /tmp/nv-p9-cont1024-m1536-r5.json
