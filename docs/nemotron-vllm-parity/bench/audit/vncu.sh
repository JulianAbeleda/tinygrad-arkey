#!/bin/bash
# usage (inside gpu-run time): vncu.sh OUTPREFIX ROLES MS   ncu of the cuBLAS kernels vLLM runs, one profiled
# F.linear per (role, M), NVTX-labelled. Thin caller of BoltBeam's collector (see oncu.sh).
D=$(cd "$(dirname "$0")" && pwd); R=$(cd "$D/../../../.." && pwd); BB=${BOLTBEAM_ROOT:-$(cd "$R/.." && pwd)/BoltBeam}
out=$1; shift
cd "$BB" && python3 -m boltbeam.cli ncu-collect --side reference --nvtx --report "$out" --out "$out.json" \
  --env "PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH" --env "HOME=$HOME" \
  -- /home/ubuntu/vllm-bench/.venv/bin/python "$D/vllm_gemm.py" --ncu "$@"
