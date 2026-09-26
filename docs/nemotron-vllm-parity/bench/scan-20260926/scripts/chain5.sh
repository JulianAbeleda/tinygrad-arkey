#!/bin/bash
O=/home/ubuntu/storage/scan-20260926/ncu2; mkdir -p $O
T=/home/ubuntu/tinygrad-self-training; A=$T/docs/nemotron-vllm-parity/bench/audit
G=/home/ubuntu/scratchpad/bin/gpu-run
C1='[[128,128,64,2,2],1,[3,true,true,true]]'
cd $T && $G time env DEV=CUDA PYTHONPATH=$T python3 /home/ubuntu/storage/scan-20260926/cand_ncu.py ssm_in 2048 "$C1" > $O/try.log 2>&1; tail -3 $O/try.log
cd /home/ubuntu/BoltBeam && $G time python3 -m boltbeam.cli ncu-collect --side ours --report $O/cand_ssm_in_2048 --out $O/cand_ssm_in_2048.json \
  --env "PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH" --env "HOME=$HOME" --env "DEV=CUDA" --env "PYTHONPATH=$T" \
  -- /usr/bin/python3 /home/ubuntu/storage/scan-20260926/cand_ncu.py ssm_in 2048 "$C1"
$G time bash $A/vncu.sh $O/vncu_2048 ssm_in,ffn_up,ffn_down 2048
echo CHAIN5 DONE
