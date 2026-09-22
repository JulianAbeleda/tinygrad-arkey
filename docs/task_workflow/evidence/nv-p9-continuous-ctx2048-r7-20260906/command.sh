#!/bin/bash
set -euo pipefail
exec flock -n /tmp/gpu-bench.lock timeout 1200 env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/decode/decode_continuous_context_coverage.py \
  --depth 2048 --max-context 2560 --nmeas 10 --reps 3 --warmup-decode 3 --gpu-state --request-scoped-prewarm \
  --census-out /tmp/nv-p9-cont2048-m2560-r7-census.json --out /tmp/nv-p9-cont2048-m2560-r7.json
