#!/bin/bash
cd /home/ubuntu/vllm-bench
NS="nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true"
# default engine config (prefix caching on -> mamba 'align')
$NS -o runs/prof_default env GPU_UTIL=0.85 EXTRA='{"disable_log_stats": false}' .venv/bin/python prof.py 8:200,128:200,128:2000 300 > runs/prof_default.log 2>&1
# prefix caching off -> mamba 'none' (1 state block/seq): true B=128 residency
$NS -o runs/prof_nopc env GPU_UTIL=0.85 EXTRA='{"disable_log_stats": false, "enable_prefix_caching": false}' .venv/bin/python prof.py 128:200,128:2000 300 > runs/prof_nopc.log 2>&1
