#!/bin/bash
# usage (inside gpu-run time): ours_sweep.sh OUT.json [modes]   one process per (mode, role)
# HCQ_NV_READY_PLACEMENT=0: one compute queue, so the independent weight copies run serially like the dependent layer
# chain of a real step (with multi-queue placement they overlap and per-kernel durations become noisy)
D=$(cd "$(dirname "$0")" && pwd); R=$(cd "$D/../../../.." && pwd)
out=$1; modes=${2:-"prod bf16"}
for mode in $modes; do for role in ssm_in ssm_out attn_q attn_kv attn_o ffn_up ffn_down output; do
  pj=$(mktemp --suffix .jsonl); rm -f $pj
  (cd $R && HCQ_NV_READY_PLACEMENT=0 DEV=NV PROFILE=1 HCQ_GRAPH_PROFILE_JSON=$pj PYTHONPATH=. python3 $D/ours_gemm.py $out $role "" $mode 2>&1 | grep -E "^[a-z_]+ +M=")
  rm -f $pj
done; done
