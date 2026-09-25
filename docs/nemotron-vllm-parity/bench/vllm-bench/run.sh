#!/bin/bash
# usage: run.sh <logname> <cmd...>   -- runs under the shared GPU lock
export PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
# FlashInfer sampler JIT needs curand.h (absent in system CUDA). Our workload (top_k=0, top_p=1) takes
# forward_native regardless (topk_topp_sampler.py:185), so disabling it changes no measured path.
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /home/ubuntu/vllm-bench
log=runs/$1.log; shift
/tmp/claude-1000/-home-ubuntu/d46d3054-5e4f-4756-9122-360207142d7f/scratchpad/gpu-run time 1800 "$@" > $log 2>&1
echo "exit $?" >> $log
