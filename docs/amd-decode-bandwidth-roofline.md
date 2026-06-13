# AMD Decode Bandwidth Roofline

Date: 2026-06-13

Status: current model-scope roofline for Qwen3 Q4_K_M decode on the local
gfx1100 path.

## Summary

The current QK generated shared-storage path is no longer bottlenecked by missing
fusion, missing fp16 dequant, or an isolated packed-dot instruction. The
model-scope roofline points at memory-load efficiency.

Canonical artifact:

- `bench/qk-bandwidth-roofline-20260613/roofline.md`
- `bench/qk-bandwidth-roofline-20260613/roofline.json`

Regenerate:

```sh
PYTHONPATH=. .venv/bin/python extra/qk_bandwidth_roofline.py \
  bench/qk-shared-storage-20260612/8b \
  bench/qk-shared-storage-20260612/14b \
  bench/qk-shared-storage-20260612/32b \
  --json bench/qk-bandwidth-roofline-20260613/roofline.json \
  --md bench/qk-bandwidth-roofline-20260613/roofline.md
```

## Current Numbers

The report uses a logical full-file bandwidth proxy:

```text
GGUF file bytes * decode tok/s
```

This is not a hardware-counter HBM-read measurement. It is a stable
model-scope proxy for comparing tinygrad and llama.cpp on the same model bytes.

| model | tinygrad generated | llama.cpp ref | tinygrad file GB/s | llama file GB/s | tinygrad % peak | llama % peak | tinygrad % llama |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8B | `52.07 tok/s` | `101.20 tok/s` | `261.82` | `508.81` | `27.27%` | `53.00%` | `51.46%` |
| 14B | `40.55 tok/s` | `65.80 tok/s` | `365.04` | `592.32` | `38.03%` | `61.70%` | `61.63%` |
| 32B | `17.23 tok/s` | `30.80 tok/s` | `340.47` | `608.67` | `35.47%` | `63.40%` | `55.94%` |

Peak assumption: RX 7900 XTX `960 GB/s`.

## Interpretation

The important comparison is same-model byte throughput:

- tinygrad generated: `27-38%` of theoretical peak by full-file proxy.
- llama.cpp: `53-63%` of theoretical peak by the same proxy.
- gap to llama.cpp: up to `268.21 GB/s`.

Batch-1 quant GEMV has low arithmetic intensity. The exact Q4_K intensity
depends on what bytes are counted, but the local per-kernel roofline already
measured accepted Q4/Q6 kernels at only `2.4-3.6 ops/packed-byte`, far below
the RX 7900 XTX FP32 ridge of roughly `64 ops/byte`. That makes schedule-only
knobs a weak lever unless they also improve memory transactions.

The committed negative surfaces match that model:

- descriptor `parts`/`LOCAL` search: exhausted;
- semantic schedule v0: rejected by full decode;
- semantic codegen v1 direct output: no microbench accepts;
- semantic codegen v2 row grouping: rejected badly;
- q8/vdot variants: correct, but not enough without better memory lowering.

## Decision

The next decode research surface is packed-weight memory-access codegen:

- make packed Q4_K/Q6_K loads compiler-visible;
- prefer wide/coalesced packed loads over scalar byte/nibble gathers;
- preserve the q8_1/packed-dot route as a helper only if it keeps ALU under the
  memory roofline;
- do not add another local schedule family without a concrete memory-traffic
  mechanism and a confirmation gate.

WMMA remains a prefill/GEMM topic unless a batch-1 decode source inspection
proves otherwise.

## Next Measurement

A future counter pass should replace this proxy with hardware-counter or
profiler evidence:

- measured copy/stream peak on the same gfx1100 box;
- tinygrad Q4/Q6 primitive HBM bytes and achieved GB/s;
- llama.cpp MMVQ HBM bytes and achieved GB/s;
- per-kernel load width/coalescing evidence from generated source or profiler.

Until that exists, the logical model-scope roofline is strong enough to stop
schedule-knob exploration and aim the next implementation at bytes.
