# LUNA-021: llama context-128 bounded trace

Verdict: `TOOL_FAILURE`.

The prerequisite tool census on this exact host records no `rocprofv3`, `rocprof`, `rocprof-compute`, or `omniperf`. `ROCtx` headers and library exist but are not a collector, so it cannot prove HIP dispatch identity. Per the task stop rule, no untraceable llama throughput/smoke workload was substituted for the required trace.

## Retained execution

Exclusive ownership was obtained non-blockingly with `flock -n /tmp/gpu-bench.lock`. The command held the lock through a separate FD 9 lock; its PID, boot ID, and `/proc/<pid>/fdinfo/9` are retained in the artifact.

```bash
flock -n /tmp/gpu-bench.lock bash -c 'set -eu; exec 9>/tmp/gpu-bench.lock; flock -n 9; ...; rocm-smi --json --showperflevel --showclocks --showcomputepartition --showpids; rocminfo'
```

Positive controls passed: lock ownership recording, `rocm-smi` device/power/clock query, and `rocminfo` gfx1100 identification. These are device-query controls only, not evidence of a llama kernel launch.

## Required commands deliberately not run

```bash
/home/ubuntu/env/llama.cpp/build/bin/llama-bench -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 -p 0 -n 128 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json
```

This is the retained LUNA-020 ordinary geometry command, but it was not run because it cannot provide the required exact kernel order, grid/block, duration, HIP correlation, or source join keys without a dispatch collector. A source-predicted `mmq-instance-q4_k.cu`/`mmq-instance-q6_k.cu` positive control therefore remains unobserved.

No AMD kernel was launched or interrupted. No timeout, `pkill`, `kill -9`, installation, or system change was used. The process exited normally and released the lock.

Artifacts: `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/`.

Next card recommendation: restore an approved ROCm dispatch collector, rerun `LUNA-001`, then rerun LUNA-021 from a fresh process before LUNA-022; do not advance trace-dependent llama cards on throughput-only evidence.
