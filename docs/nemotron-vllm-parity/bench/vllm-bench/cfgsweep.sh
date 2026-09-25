#!/bin/bash
# three engine configs at P=2000, N=1024, B=64/128/256, with scheduler stats logging on
cd /home/ubuntu/vllm-bench
P=.venv/bin/python
$P bench.py --prompt-lens 2000 --bs 64,128,256 --max-tokens 1024 --out runs/cfg_default.json --extra '{"disable_log_stats": false}' > runs/cfg_default.log 2>&1
$P bench.py --prompt-lens 2000 --bs 64,128,256 --max-tokens 1024 --out runs/cfg_nopc_fp32.json --extra '{"disable_log_stats": false, "enable_prefix_caching": false}' > runs/cfg_nopc_fp32.log 2>&1
$P bench.py --prompt-lens 2000 --bs 64,128,256 --max-tokens 1024 --mamba-dtype bfloat16 --out runs/cfg_nopc_bf16.json --extra '{"disable_log_stats": false, "enable_prefix_caching": false}' > runs/cfg_nopc_bf16.log 2>&1
