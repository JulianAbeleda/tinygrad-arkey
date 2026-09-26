#!/bin/bash
# usage (inside gpu-run time): oncu.sh OUTPREFIX ROLES MS [mode]   ncu of BoltBeam kernels via the DEV=CUDA proxy
D=$(cd "$(dirname "$0")" && pwd); R=$(cd "$D/../../../.." && pwd)
out=$1; shift
cd $R && sudo systemd-run --scope -q -p MemoryMax=20G env PATH=/usr/local/cuda/bin:$PATH HOME=$HOME DEV=CUDA PYTHONPATH=$R \
  ncu --profile-from-start off --target-processes all \
  --section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis --cache-control all --clock-control none -f -o "$out" \
  /usr/bin/python3 "$D/ours_ncu.py" "$@"
rc=$?; sudo chown ubuntu:ubuntu "$out".ncu-rep; exit $rc
