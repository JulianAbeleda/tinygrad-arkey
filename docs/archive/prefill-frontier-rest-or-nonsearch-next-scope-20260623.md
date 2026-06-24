# Prefill Frontier — Rest / Non-Search Next Scope (2026-06-23)

## Why machine search is NOT ready
The prefill audit (`docs/prefill-post-decode-parity-frontier-result-20260623.md`) found the dependency-free LDS GEMM
kernel is **at GPU-level parity-to-+10 % with vendored Tensile** on the actual prefill shape. A GEMM machine search
fails 4 of 6 readiness criteria: not materially bottlenecked at the kernel, **local timing does not transfer** to
whole-prefill, **tuning knobs exhausted** (BK64 overflows VGPR; PAD16/PLR/occupancy done), and expected whole-prefill
gain ≈ 0. Searching a solved, non-transferring metric would burn compute for nothing.

`PREFILL_MACHINE_SEARCH_NOT_READY` → `PREFILL_NEEDS_NONSEARCH_FIX_FIRST`.

## The actual frontier: in-model integration penalty (non-search)
The whole-prefill gap is **66 % (graph-GEMM) → 87 % (Tensile)** *with a kernel already at isolation parity*. That gap
is **integration, not kernel** — the universal "isolated wins don't transfer" lesson (in-model gate/up runs ~22 TFLOPS
vs ~75 isolated; concrete-chunk 3436 vs whole-prefill 1983 tok/s). This is the dominant lever and is **not** a search.

### Lever 1 (dominant) — attribute and close the in-model integration penalty
**First step is measurement, not a kernel.** Build a **synced per-role in-model prefill time-tax** (the audit deferred
this — the prefill_v2 path is intricate, not ad-hoc-measurable):
- Use the synced arbiter pattern (`qk_prefill_tc_attn_concrete_gate.py` `burst(K)` + `dev.synchronize()`), per
  Measurement-Authority SOP. NOT nosync `qk_prefill_v2_measure`.
- Attribute whole-prefill GPU time per role (ffn gate/up/down, q/o/k/v proj, attention QK/PV, norm/rope, copies) for
  **graph-GEMM** vs **Tensile** routes (both byte-identical) on the same harness.
- Answer: which roles/shapes fall back to slow WMMA under graph-GEMM but route through Tensile? Is the delta per-kernel
  launch/fusion overhead, role coverage, or attention? **That attribution decides the next bounded fix** (extend
  dependency-free coverage to the lagging roles, or reduce per-kernel integration overhead).
- Gate: a fix only counts if it moves **whole-prefill synced tok/s** (not a single concrete chunk, not isolated TFLOPS).

### Lever 2 (bounded micro-lever) — VALU address-leanness
The one ISA-confirmed kernel residual is **+23 % VALU** (8.66M vs 7.04M; `v_add_nc_u32`/`v_lshlrev`/`v_mul_lo` per-
iteration index math). A **deterministic** hand-codegen fix in `build_gemm_lds2`: hoist loop-invariant address math,
strength-reduce per-iteration increments into running pointers. Gates: ISA audit shows the VALU count drop (no
spill/VGPR regression), rel_rmse ≤ 2e-4 preserved, **and** it must show whole-prefill synced transfer (kernel is
already at parity, so expect small). Bounded; not a search.

### Lever 3 (policy, not perf) — defaults
`PREFILL_GRAPH_GEMM` is engineering-ready (4 readiness gates pass, byte-identical greedy) but default-on is
**owner-pending** on an absolute-parity drift call (`max_abs_dNLL 0.0176` report-only; `max_positive_dNLL 0.0094 ≤
0.01` gate passes). `PREFILL_V2` stays **off** by owner decision (+14 GB resident during decode for no decode benefit).
These are owner policy calls, not optimization work.

## Is prefill at rest?
**Kernel-wise: yes** — the GEMM is at Tensile parity; no material kernel headroom remains, and the dependency story is
a tier choice (dependency-free ~66 % vs vendored-Tensile ~87 %). **In-model-wise: no** — the integration penalty is a
real, unattributed ~+20 pp lever, but it is an attribution-then-engineering task, not a search and not a kernel.

## Recommended next action
Run **Lever 1's synced per-role in-model prefill time-tax** as the next bounded task. Only after it attributes the
66 %→87 % gap does a concrete fix (coverage extension or integration-overhead reduction) become scopeable. Do **not**
start a GEMM machine search. Decode stays at parity and untouched.

## Boundaries (unchanged)
No decode changes; no kernels during attribution; no machine search; no default flips; no 14B/32B; whole-prefill
synced transfer is the only authority (not isolated TFLOPS, not the concrete single chunk).
