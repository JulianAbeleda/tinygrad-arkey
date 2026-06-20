# Prefill — TRUE throughput (bench under-measures 2.3×) + real ~2.7× in-model matmul penalty

Date: 2026-06-20

## Headline (two banked numbers corrected)

The prefill benchmark (`qk_prefill_v2_measure.py`, "2797 tok/s") **under-measures by ~2.3×**: it times
`fn().realize()` in a loop **without `dev.synchronize()`**, so the host returns before the GPU finishes
(host-dispatch time, not GPU time), and the baseline-first sequencing masks it. The **true** prefill is
**~1200 tok/s**, confirmed by three independent methods.

| measurement | ms/forward | tok/s |
|---|---:|---:|
| bench (baseline-first, **nosync**) | 184 | 2797 |
| v2-only, nosync | 410 | 1249 |
| v2-only, **sync** each call | 430 | 1190 |
| **arbiter: K=16 forwards, ONE sync, total/K** | **414** | **1236** |

The arbiter is the trustworthy GPU-throughput method (same one that resolved the GEMM batch-isolate, validated
against `wait=True`). Three methods agree ~414–433 ms; only the bench's 184 ms is the outlier.

## Two corrections

1. **"Prefill = 2797 tok/s = ~93% of llama" → RETRACTED.** True prefill ≈ **1200 tok/s ≈ 40% of llama**
   (llama pp512 3020 is a synced benchmark; the 93% compared tinygrad's nosync-184 ms to llama's synced number
   — apples-to-oranges).
2. **"Prefill matmul integrates fine / ~9% upside" → RETRACTED.** Using the true 414 ms forward, the in-model
   gate/up GEMM runs at **~22 TFLOPS** (39.5% share × 414 ms / 72 launches) vs **~60 isolated** (authority,
   same kernel) → a **real ~2.7× in-model graph penalty.** The penalty probe's 0.36 ratio was **correct**; it
   was wrongly rejected as a warmstart artifact (warmstart *did* apply, apply=5 — the forward is genuinely
   ~430 ms, not the bench's 184).

## What it means for integration

- The in-model integration gap is **large and recoverable**: the matmuls (~70% of prefill) run at **~38% of
  their isolated speed** inside the graph. This is the real lever — far bigger than the ~9% earlier estimate.
- A faster *standalone* kernel (our GEMM, 74 vs authority 60) won't help much **if the 2.7× penalty is
  graph-level** (surrounding ops / fusion boundaries / scheduling / dequant), because the in-model kernel
  already runs at 22 regardless of its isolated ceiling. **Root-causing the 2.7× is the prerequisite.**
- Candidate causes to attribute: (a) the warmstart TC schedule being graph-suboptimal, (b) per-matmul
  surrounding ops (`.contiguous()`, fp16 cast, dequant) not fusing, (c) memory traffic between kernels,
  (d) sustained-clock vs isolated-burst (partial — 2.7× is too large for clock alone).

## Measurement hygiene (banked lesson)

- **Always `dev.synchronize()` around in-model forward timing**, or use the **batched arbiter** (K forwards,
  one sync, total/K). A nosync `realize()` loop measures host dispatch, not GPU time — here it inflated prefill
  throughput 2.3×. This is the prefill analogue of the GEMM host-overhead trap.
- Cross-framework ratios must compare like timing: tinygrad-synced vs llama-synced (→ 40%), never
  tinygrad-nosync vs llama-synced (→ the spurious 93%).

## Status / next

- True prefill throughput established: **~1200 tok/s (~40% of llama)**, ~2.7× in-model matmul penalty.
- Next: root-cause the 2.7× (attribute graph vs surrounding-ops vs clock) on a clean synced harness, then
  decide whether the lever is graph-scheduling, op-fusion, or wiring our kernel.
- Promotion quality gate for any route change must be **VRAM-safe** (sampled/chunked NLL + greedy smoke), not
  the full (512×vocab) logits NLL (that is the source of the eval OOM, not the route).
