#!/bin/bash
export PATH=/home/ubuntu/vllm-bench/.venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
cd /home/ubuntu/vllm-bench
NS="nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true"
$NS -o runs/prof_spec64 env GPU_UTIL=0.85 EXTRA='{"disable_log_stats": false}' .venv/bin/python prof.py 64:2000 300
