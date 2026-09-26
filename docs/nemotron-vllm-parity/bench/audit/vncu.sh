#!/bin/bash
# usage (inside gpu-run time): vncu.sh OUTPREFIX ROLES MS
# ncu of the cuBLAS kernels vLLM runs, one profiled F.linear per (role, M). sudo only for admin counters,
# memory-capped (a root profiler once OOM-crashed this box).
D=$(cd "$(dirname "$0")" && pwd)
out=$1; shift
sudo systemd-run --scope -q -p MemoryMax=20G env PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/cuda/bin:$PATH HOME=$HOME \
  ncu --profile-from-start off --target-processes all --nvtx \
  --section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis --cache-control all --clock-control none -f -o "$out" \
  /home/ubuntu/vllm-bench/.venv/bin/python "$D/vllm_gemm.py" --ncu "$@"
rc=$?; sudo chown ubuntu:ubuntu "$out".ncu-rep; exit $rc
