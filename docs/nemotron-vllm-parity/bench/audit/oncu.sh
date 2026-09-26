#!/bin/bash
# usage (inside gpu-run time): oncu.sh OUTPREFIX ROLES MS [mode]   ncu of BoltBeam kernels via the DEV=CUDA proxy
# Thin caller of ncu_kernel_counters.py build-cmd: sections/cache/clock-control and the memory-capped
# sudo scope live there once, shared with vncu.sh.
D=$(cd "$(dirname "$0")" && pwd); R=$(cd "$D/../../../.." && pwd)
out=$1; shift
cmd=$(cd "$R" && python3 "$D/ncu_kernel_counters.py" build-cmd --out "$out" --sudo-scope --mem-max 20G \
  --env "PATH=/usr/local/cuda/bin:$PATH" --env "HOME=$HOME" --env "DEV=CUDA" --env "PYTHONPATH=$R" \
  -- /usr/bin/python3 "$D/ours_ncu.py" "$@")
cd "$R" && eval "$cmd"
rc=$?; sudo chown ubuntu:ubuntu "$out".ncu-rep; exit $rc
