#!/usr/bin/env bash
set -u
cd /home/ubuntu/worktrees/luna-profiler-llama128
export PATH=/opt/rocm/bin:$PATH
out=bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/diagnostic-20260726
{
  printf 'kind=rocprofv3-simple-kernel-trace\n'
  printf 'base_commit='; git rev-parse HEAD
  printf 'boot_id='; cat /proc/sys/kernel/random/boot_id
  printf 'PATH=%s\n' "$PATH"
  printf 'pid=%s\n' "$$"
  printf 'argv=/opt/rocm/bin/rocprofv3 --kernel-trace --output-directory %s/simple-kernel-trace --output-format json -- /home/ubuntu/env/llama.cpp/build/bin/llama-bench -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 -p 0 -n 1 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json\n' "$out"
} > "$out/simple-kernel-trace-manifest.txt"
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --output-directory "$out/simple-kernel-trace" --output-format json -- \
  /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 1 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json \
  > "$out/simple-kernel-trace-stdout.json" 2> "$out/simple-kernel-trace-stderr.txt"
status=$?
printf '%s\n' "$status" > "$out/simple-kernel-trace-exit-status.txt"
exit "$status"
