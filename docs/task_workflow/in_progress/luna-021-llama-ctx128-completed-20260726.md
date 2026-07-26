# LUNA-021: llama context-128 bounded trace

Verdict: `PASS`.

The accepted ctx128 workload completed once under `/tmp/gpu-bench.lock` using the known-good minimal ROCm profiler configuration:

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --output-directory /home/ubuntu/worktrees/luna-profiler-llama128/bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/completed-20260726/rocprofv3 \
  --output-format json -- \
  /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 128 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json
```

The child JSON is nonempty and reports ROCm on the RX 7900 XTX, `n_gen=128`, `n_depth=128`, `n_gpu_layers=99`, flash attention enabled, 2.148676792 s average measurement time, and 59.571547 token/s. The raw profiler JSON is nonempty (48,991,461 bytes) and contains 110,905 dispatch records and 923 kernel metadata records.

Observed material families, with timestamp-derived aggregate duration and one observed grid/block, are retained in `kernel-duration-ledger.json`:

- `mul_mat_vec_q` Q4-family: 999,174,394 ns; grid `163840x1x1`; block `32x1x1`.
- `mul_mat_vec_q` Q6-family: 227,356,728 ns; grid `163840x2x1`; block `32x2x1`.
- `mul_mat_q` Q4-family: 72,165,023 ns; grid `1280x8x1`; block `32x8x1`.
- `quantize_q8_1`: 42,651,696 ns; grid `5120x1x1`; block `256x1x1`.
- `rms_norm_f32`: 40,095,349 ns; grid `1024x1x1`; block `1024x1x1`.
- `flash_attn_tile`: 36,933,474 ns; grid `32x16x40`; block `32x2x1`.
- `flash_attn_combine_results`: 10,666,004 ns; grid `128x40x1`; block `128x1x1`.

The run records the worktree commit, boot ID, PATH beginning `/opt/rocm/bin`, exact argv, output paths, and the MMQ/attention controls. CLI flash attention was `-fa 1`; `GGML_HIP_FORCE_MMQ` and `GGML_HIP_USE_FLASH_ATTN` were unset. Existing rocprofv3 and gfx1100 device positive controls remain retained alongside this capture.

No timeout, signal, `pkill`, `kill -9`, or live AMD-kernel interruption was used. This completes the ctx128 trace prerequisite; do not infer ctx512 or ctx4096 behavior from it.
