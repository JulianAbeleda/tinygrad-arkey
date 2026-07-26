#!/usr/bin/env bash
set -eu
export PATH=/opt/rocm/bin:$PATH
exec /opt/rocm/bin/rocprofv3 --kernel-trace --hip-runtime-trace --stats --output-config on \
  --output-directory bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/rocprofv3-raw \
  --output-format json -- \
  /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 128 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json
