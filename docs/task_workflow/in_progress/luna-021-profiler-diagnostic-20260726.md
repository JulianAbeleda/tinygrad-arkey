# LUNA-021 profiler diagnostic

Verdict: `SUPPORTED_WORKING_TRACE_RECIPE`.

The direct unprofiled control completed with exit status zero and emitted valid JSON identifying backend `ROCm`, `gfx1100`, `n_depth=128`, `n_gen=1`, full GPU offload, and flash attention. This rules out the llama binary, model, and child-output path as the reason for the prior empty capture.

The first simplified profiler control also completed with exit status zero:

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --output-directory <absolute-output-directory> --output-format json -- \
  /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 1 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json
```

It retained `simple-kernel-trace/ubuntu-dualboot/143113_results.json`, containing 923 kernel-name records. Positive kernel names include `quantize_mmq_q8_1`, `mul_mat_vec_q`, and RMS-norm variants. This proves that rocprofv3 observes the llama HIP/KFD dispatches on this device.

The earlier nonworking invocation combined `--kernel-trace --hip-runtime-trace --stats --output-config on` with a relative output directory. It returned zero but retained neither child stdout nor profiler output. The cause is therefore the additional option/output configuration combination, not command syntax before the separator, child-process visibility, or direct KFD/HIP tracing. This bounded diagnostic intentionally stops at the first working configuration and does not assign failure to one particular added flag.

Both controls held `/tmp/gpu-bench.lock` for their entire process lifetime. They recorded the task commit, boot ID, PATH with `/opt/rocm/bin` first, argv, stdout, stderr, and exit status. No timeout, `pkill`, `kill -9`, or AMD-kernel interruption was used.

`LUNA-021` itself remains `TOOL_FAILURE`: the required full bounded `-n 128` trace has not been rerun using the known-good recipe. Do not advance LUNA-022 or LUNA-023 on this diagnostic alone.
