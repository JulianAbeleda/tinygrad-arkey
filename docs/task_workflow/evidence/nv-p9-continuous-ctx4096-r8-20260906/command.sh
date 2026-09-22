#!/bin/bash
set -euo pipefail
exec flock -n /tmp/gpu-bench.lock timeout 1200 env DEV=NV PYTHONPATH=. .venv/bin/python extra/llm_research/decode/decode_continuous_context_coverage.py \
  --depth 4096 --max-context 4608 --nmeas 10 --reps 3 --warmup-decode 3 --gpu-state --request-scoped-prewarm \
  --census-out /tmp/nv-p9-cont4096-m4608-r8-census.json --out /tmp/nv-p9-cont4096-m4608-r8.json
