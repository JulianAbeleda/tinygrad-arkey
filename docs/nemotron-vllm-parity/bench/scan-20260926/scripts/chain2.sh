#!/bin/bash
O=/home/ubuntu/storage/scan-20260926/climb; mkdir -p $O
cd /home/ubuntu/tinygrad-self-training
for s in ssm_in:256 ffn_down:256 ffn_up:256 ssm_out:256 ssm_in:2048; do
  r=${s%%:*}; m=${s##*:}
  echo "== $s $(date)"
  DEV=NV PYTHONPATH=. /home/ubuntu/scratchpad/bin/gpu-run time python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --scan --roles $r --rows $m --out $O/$r-$m.jsonl 2>&1 | tail -3
  echo "== done $s $(date)"
done
echo CHAIN2 DONE
