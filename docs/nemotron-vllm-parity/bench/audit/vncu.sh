#!/bin/bash
# usage (inside gpu-run time): vncu.sh OUTPREFIX ROLES MS   ncu of the cuBLAS kernels vLLM runs, one
# profiled F.linear per (role, M). sudo only for admin counters, memory-capped (a root profiler once
# OOM-crashed this box). Thin caller of ncu_kernel_counters.py build-cmd (shared with oncu.sh).
D=$(cd "$(dirname "$0")" && pwd)
out=$1; shift
cmd=$(python3 "$D/ncu_kernel_counters.py" build-cmd --out "$out" --sudo-scope --mem-max 20G --nvtx \
  --env "PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/cuda/bin:$PATH" --env "HOME=$HOME" \
  -- /home/ubuntu/vllm-bench/.venv/bin/python "$D/vllm_gemm.py" --ncu "$@")
eval "$cmd"
rc=$?; sudo chown ubuntu:ubuntu "$out".ncu-rep; exit $rc
