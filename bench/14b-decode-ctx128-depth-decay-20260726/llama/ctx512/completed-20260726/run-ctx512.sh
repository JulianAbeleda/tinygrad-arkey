#!/usr/bin/env bash
set -u
cd /home/ubuntu/worktrees/luna-profiler-llama128
export PATH=/opt/rocm/bin:$PATH
out=/home/ubuntu/worktrees/luna-profiler-llama128/bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx512/completed-20260726
{
  printf 'task=LUNA-022\n'
  printf 'base_commit='; git rev-parse HEAD
  printf 'boot_id='; cat /proc/sys/kernel/random/boot_id
  printf 'timestamp_utc='; date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'PATH=%s\n' "$PATH"
  printf 'pid=%s\n' "$$"
  printf 'profiler=/opt/rocm/bin/rocprofv3 --kernel-trace --output-directory %s/rocprofv3 --output-format json\n' "$out"
  printf 'argv=/home/ubuntu/env/llama.cpp/build/bin/llama-bench -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 -p 0 -n 128 -d 512 -b 512 -ub 512 -fa 1 -r 1 -o json\n'
  printf 'GGML_HIP_FORCE_MMQ=%s\n' "${GGML_HIP_FORCE_MMQ-unset}"
  printf 'GGML_HIP_USE_FLASH_ATTN=%s\n' "${GGML_HIP_USE_FLASH_ATTN-unset}"
} > "$out/manifest.txt"
/opt/rocm/bin/rocprofv3 --kernel-trace --output-directory "$out/rocprofv3" --output-format json -- \
  /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 128 -d 512 -b 512 -ub 512 -fa 1 -r 1 -o json \
  > "$out/llama-bench-stdout.json" 2> "$out/rocprofv3-stderr.txt"
status=$?
printf '%s\n' "$status" > "$out/exit-status.txt"
exit "$status"
