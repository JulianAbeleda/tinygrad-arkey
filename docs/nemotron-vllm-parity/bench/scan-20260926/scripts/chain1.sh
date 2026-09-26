#!/bin/bash
set -x
O=/home/ubuntu/storage/scan-20260926
A=/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/audit
cd /home/ubuntu/vllm-bench && /home/ubuntu/scratchpad/bin/gpu-run time env PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/cuda/bin:$PATH .venv/bin/python $A/vllm_gemm.py $O/vllm_gemm.json > $O/vllm_gemm.log 2>&1
df -h /home/ubuntu/storage /
cd /home/ubuntu/tinygrad-self-training && MS=8,16,32,64,128,512,1024 NCU=1 bash docs/nemotron-vllm-parity/bench/scan.sh $O/scan > $O/scan.log 2>&1
echo CHAIN1 DONE
