#!/bin/bash
# ReplaySSM (Triton) requires Model Runner V1 -> A/B both on runner V1
cd /home/ubuntu/vllm-bench
export VLLM_USE_V2_MODEL_RUNNER=0
.venv/bin/python bench.py --prompt-lens 2000 --bs 8,64,128 --max-tokens 1024 --out runs/cfg_v1runner.json --extra '{"disable_log_stats": false}' > runs/cfg_v1runner.log 2>&1
.venv/bin/python bench.py --prompt-lens 2000 --bs 8,64,128 --max-tokens 1024 --out runs/cfg_replayssm.json --extra '{"disable_log_stats": false, "use_replayssm": true}' > runs/cfg_replayssm.log 2>&1
NS="nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node --trace=cuda,nvtx --sample=none --cpuctxsw=none -f true"
$NS -o runs/prof_replay env GPU_UTIL=0.85 EXTRA='{"disable_log_stats": false, "use_replayssm": true}' .venv/bin/python prof.py 128:2000 300 > runs/prof_replay.log 2>&1
