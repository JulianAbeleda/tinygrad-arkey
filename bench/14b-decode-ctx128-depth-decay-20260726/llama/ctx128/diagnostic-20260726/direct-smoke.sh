#!/usr/bin/env bash
set -u
cd /home/ubuntu/worktrees/luna-profiler-llama128
export PATH=/opt/rocm/bin:$PATH
out=bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/diagnostic-20260726
{
  printf 'kind=direct-unprofiled-smoke\n'
  printf 'base_commit='; git rev-parse HEAD
  printf 'boot_id='; cat /proc/sys/kernel/random/boot_id
  printf 'PATH=%s\n' "$PATH"
  printf 'pid=%s\n' "$$"
  printf 'argv=/home/ubuntu/env/llama.cpp/build/bin/llama-bench -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 -p 0 -n 1 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json\n'
} > "$out/direct-smoke-manifest.txt"
/home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 1 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json \
  > "$out/direct-smoke-stdout.json" 2> "$out/direct-smoke-stderr.txt"
status=$?
printf '%s\n' "$status" > "$out/direct-smoke-exit-status.txt"
exit "$status"
