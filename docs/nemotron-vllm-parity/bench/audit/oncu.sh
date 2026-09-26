#!/bin/bash
# usage (inside gpu-run time): oncu.sh OUTPREFIX ROLES MS [mode]   ncu of our GEMM routes via the DEV=CUDA proxy.
# Thin caller of BoltBeam's collector (ncu command line, memory-capped sudo scope, raw export, import):
# writes OUTPREFIX.ncu-rep, OUTPREFIX.raw.csv, OUTPREFIX.stdout.log (the SHAPE labels) and OUTPREFIX.json.
D=$(cd "$(dirname "$0")" && pwd); R=$(cd "$D/../../../.." && pwd); BB=${BOLTBEAM_ROOT:-$(cd "$R/.." && pwd)/BoltBeam}
out=$1; shift
cd "$BB" && python3 -m boltbeam.cli ncu-collect --side ours --report "$out" --out "$out.json" \
  --env "PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH" --env "HOME=$HOME" --env "DEV=CUDA" --env "PYTHONPATH=$R" \
  -- /usr/bin/python3 "$D/ours_ncu.py" "$@"
