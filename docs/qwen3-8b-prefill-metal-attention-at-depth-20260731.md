# Metal prefill attention at depth

Date: 2026-07-31

Qwen3-8B-Q4_K_M, Apple M4 10-core / Metal, current production path
(`prefill_route = DIRECT_PACKED_FALLBACK`). Per-kernel timings from `JIT=0 DEBUG=2` at four start
positions, one 512-token chunk each. Probe: `scratchpad/metal_prefill_attn_depth_probe.py`; raw logs
`/tmp/metal_prefill_attn_depth/depth_*.log`.

**This is the measurement that sizes every prefill decision after today.**

---

## 1. Result

| depth | KV-varying (attention) | shared (GEMM + elementwise) | total | attention share |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 336.7 ms | 316.3 ms | 652.9 ms | **51.6%** |
| 1024 | 156.1 ms | 315.4 ms | 471.5 ms | **33.1%** |
| 2048 | 321.1 ms | 317.8 ms | 638.9 ms | **50.3%** |
| 4096 | 291.3 ms | 316.4 ms | 607.7 ms | **47.9%** |

**Attention is already about half of per-chunk prefill time on Metal, at every depth measured.**

**The shared half is flat to 0.8%** — 315.4–317.8 ms across a full 8× range of context. That
independently confirms, at the whole-model level, what the stress test found at the kernel level: GEMM
does not decay with depth.

### 1.1 How kernels were classified

Not by name-guessing. The attention kernels are **recompiled per depth with the KV length baked into the
kernel name** (`r_16_32_32_4_2_4_4_512_…` at depth 4096, `…_256_…` at 2048, `…_128_…` and
`r_32_16_8_32_4_2_4_4_16_…` at 1024). So they appear at exactly one depth each, while the 24 kernels
present at *all four* depths are the shape-invariant GEMM and elementwise work.

The split is therefore **KV-varying vs depth-invariant**, which falls out of the data rather than from a
naming heuristic. An earlier attempt to classify by dimension signature failed — it put every `r_`
kernel in one bucket — and is not the basis for the table above.

---

## 2. This is the opposite of AMD, and the cause is known

AMD 8B/gfx1100 (`docs/prefill-current-state.md`, `PROFILE=1` per-kernel trace):

> the per-chunk decomposition is **GEMM-bound and flat with context** (QKV/O + FFN ≈ 470 ms/chunk,
> 93%→74% of per-chunk time) while **fused attention is the ONLY component that grows with KV** — 4%
> (pp512) → 23% (pp4096), ~8×.

| | attention share, pp512 | attention share, pp4096 |
| --- | ---: | ---: |
| AMD (fused attention) | 4% | 23% |
| **Metal (no fused attention)** | **51.6%** | **47.9%** |

**Metal's attention share at pp512 is ~13× AMD's.**

**Cause, confirmed:** `bench/prefill-whole-synced/t2-metal-pp512.json` reports
`custom_kernel_attention_trace: {"dispatches": 0}`. **Metal dispatches no fused attention kernel.** AMD's
flash-prefill attention work does not apply here, so Metal runs the unfused path and pays for it at every
depth — not only at long context.

---

## 3. What this does to the precontract win

The precontract kernel measures **3610 GFLOPS against a 1063 control — 3.4×**
(`docs/qwen3-8b-prefill-metal-precontract-campaign-20260731.md`). That speedup applies only to the
**shared** half of the table above.

**Projection — arithmetic on the measured split, not a measurement.** Holding attention fixed and
dividing the shared half by 3.4:

| depth | attention share now | → after | per-chunk total | speedup |
| ---: | ---: | ---: | --- | ---: |
| 512 | 51.6% | 78.4% | 652.9 → 429.7 ms | **1.52×** |
| 1024 | 33.1% | 62.7% | 471.5 → 248.8 ms | **1.89×** |
| 2048 | 50.3% | 77.5% | 638.9 → 414.6 ms | **1.54×** |
| 4096 | 47.9% | 75.8% | 607.7 → 384.4 ms | **1.58×** |

**A 3.4× kernel win is worth roughly 1.5–1.9× end-to-end**, and it leaves attention owning **76–78%** of
prefill.

This is Amdahl's law with a measured split, and it caps what the remaining lifecycle work (QUALIFY,
POLICY) can deliver. It does not argue against finishing that work — 1.5× is real — but it settles what
comes *after*: **fused attention on Metal is the larger remaining lever, and it becomes dominant the
moment the GEMM path lands.**

---

## 4. Caveats

- **Depth 1024 is an outlier and is not smoothed here.** Attention lands at 156.1 ms against ~290–337 ms
  at the other three depths. The KV-varying kernels differ in *shape*, not only in size
  (`r_32_16_8_32_4_2_4_4_16` + `r_16_32_32_4_2_4_4_128` at 1024, versus a single
  `r_16_32_32_4_2_4_4_512` at 4096), so kernel *selection* changes with depth and its cost does not scale
  smoothly. The share is therefore not monotonic in depth, and no clean quadratic should be read into it.
- **One chunk per depth, one run each.** These are per-chunk kernel timings, not whole-model tok/s with
  repetitions. The flatness of the shared half across four independent runs (0.8% spread) is the internal
  consistency check.
- **`JIT=0 DEBUG=2`** is the campaign's established per-kernel basis; absolute times are higher than a
  batched run and are not comparable to tok/s figures.
- The §3 table is **arithmetic on a measured split**, not an executed configuration. Nothing has been run
  with the precontract kernel wired into whole-model prefill — that requires QUALIFY and POLICY, both
  blocked.

---

## 5. Consequence

Before today the assumption was that Metal prefill was GEMM-bound, as AMD's is. **It is not.** Attention
is the larger half already, and the reason is structural rather than tuning: there is no fused attention
path on this target.

Priority after the lifecycle work should be fused prefill attention on Metal, not further GEMM geometry
search — the campaign already showed geometry is flat within 1.26% across the legal space, so there is
little left there, while attention is ~50% of the time and entirely unoptimized.
