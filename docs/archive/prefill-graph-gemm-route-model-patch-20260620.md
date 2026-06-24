# Prefill Graph GEMM Route Model Patch - 2026-06-20

Adds `PREFILL_GRAPH_GEMM=1`, a default-off research route that tries to replace eligible `PREFILL_V2` fp16
matmuls with the dependency-free graph-capturable RDNA3 GEMM.

Fallback behavior:

- flag off: unchanged;
- unsupported shape, missing realized fp16 weight, or bias: falls back to normal `PREFILL_V2`;
- decode: unchanged.

Measurement command:

```bash
DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 PYTHONPATH=. python3 extra/qk_prefill_v2_measure.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```
