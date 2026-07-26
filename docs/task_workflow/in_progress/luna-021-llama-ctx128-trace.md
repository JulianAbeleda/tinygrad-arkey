# LUNA-021: llama context-128 bounded trace

Verdict: `TOOL_FAILURE`.

`LUNA-001` is reopened and passes under the corrected ROCm environment: `/opt/rocm/bin/rocprofv3` version 1.1.0 on ROCm 7.2.4 exposes kernel and HIP runtime tracing. `rocminfo` positively identifies the intended RX 7900 XTX (`gfx1100`, wave32).

The sole bounded capture invocation used `/opt/rocm/bin/rocprofv3 --kernel-trace --hip-runtime-trace --stats --output-config on --output-format json` around the retained llama command:

```bash
/home/ubuntu/env/llama.cpp/build/bin/llama-bench -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 -p 0 -n 128 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json
```

It was launched once through a non-blocking `/tmp/gpu-bench.lock` wrapper and returned status zero in 4.5 seconds. The configured controls were full GPU offload (`-ngl 99`), flash attention (`-fa 1`), and the ambient `GGML_HIP_FORCE_MMQ` and `GGML_HIP_USE_FLASH_ATTN` values recorded as unset. The llama binary reports HIP/ROCm device `gfx1100`; its commit and SHA256, model SHA256, base commit, boot ID, PATH, and exact argv are retained in `manifest-before-trace.txt`.

This does not satisfy the trace acceptance criteria. No benchmark stdout, profiler stderr, generated profiler configuration, or dispatch JSON appeared. Consequently there are no observed kernel names, source join keys, grids, blocks, or durations. The exact failure is a zero-status invocation with all required trace positive controls absent, which is `TOOL_FAILURE` under scope section 14.2. No profiler error text exists because no stderr file was created.

Artifacts: `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/`, especially `trace-artifact-index.json`, `trace-command.sh`, and the positive-control files.

No timeout, signal, `pkill`, `kill -9`, or live AMD-kernel interruption was used. The empty `rocprofv3-raw/` directory is retained. LUNA-022 and LUNA-023 remain blocked because LUNA-021 did not produce a passing bounded dispatch trace.
