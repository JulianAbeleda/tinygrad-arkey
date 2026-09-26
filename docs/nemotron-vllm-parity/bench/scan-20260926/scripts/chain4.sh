#!/bin/bash
O=/home/ubuntu/storage/scan-20260926/climb; mkdir -p $O
cd /home/ubuntu/tinygrad-self-training
for s in ssm_out:2048 attn_q:2048 attn_o:2048 attn_kv:2048 ffn_up:1024 ffn_down:1024 ssm_out:1024 ffn_up:128 ffn_down:128 ssm_out:128 output:256 output:128; do
  r=${s%%:*}; m=${s##*:}
  echo "== $s $(date)"
  DEV=NV PYTHONPATH=. /home/ubuntu/scratchpad/bin/gpu-run time python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --scan --roles $r --rows $m --out $O/$r-$m.jsonl 2>&1 | tail -3
  echo "== done $s $(date)"
done
echo CHAIN4 DONE
