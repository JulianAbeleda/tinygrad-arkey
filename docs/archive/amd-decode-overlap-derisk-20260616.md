# Decode overlap de-risk — 2026-06-16

Gate before committing to the second-queue overlap infra (the +38% spike lever). Also records
that **norm-fusion (the intended warm-up increment) was refuted** and dropped.

## Norm-fusion: refuted (dropped)

Folding RMSNorm into the quantized GEMV is **blocked, non-exact, and low-ROI**:
- the GEMV kernels are **single-accumulator** (linearizer rejects multi-accumulator), so the RMS
  `mean(x²)` reduction can't live in the GEMV (`extra/q4_k_gemv_primitive.py`,
  `extra/q6_k_gemv_primitive.py`);
- folding only the gain `g` pre-quantization is **non-exact** (`quantize(x·g) ≠ quantize(x)·g`,
  `extra/qk_layout.py q8_1_quantize`) — changes output bytes;
- the norm is **already lazily fused** into the primitive's `.contiguous()`
  (`tinygrad/nn/__init__.py:300`), so the standalone win is ~0.5–1 ms/tok batched, not the
  named-mode 11%.

Refusing it (wrong abstraction). The real lever is overlap.

## Gate 1 — idle-capacity proxy (zero code): **PASS**

Necessary condition for overlap: batch-1 decode must leave GPU capacity idle. Ran 2 concurrent
`cli --benchmark 40` decodes (8B, ~6.3 GB each, 12.6 GB < 24 GB), full-clock, steady median:

| config | tok/s |
|---|---:|
| single process | 55.0 |
| concurrent proc A | 39.8 |
| concurrent proc B | 32.7 |
| **aggregate** | **72.5 = 1.32× single** |

Two streams share HBM, yet aggregate throughput **rises +32%** — so batch-1 decode does **not**
saturate the GPU; there is real idle capacity, and the AMD hardware/driver **does run concurrent
compute** (the two streams genuinely overlapped). It is not 2× because two *full* decodes contend
for the memory controller — so +32% is a reasonable **floor** on what in-model overlap (shared
weights, no process context-switch, no 2× VRAM) can reclaim.

## Gate 2 — concurrency feasibility: resolved without a full prototype

The intra-process two-queue micro-prototype (manual HCQ program/kernarg/timeline-signal handling
below `HCQProgram.__call__`) is real low-level work — but the question it answers is already
settled by cheaper evidence:
- **Hardware concurrent compute is proven** (Gate 1: two streams overlapped, +32%).
- **tinygrad exposes multiple HW queues**: `dev.hw_compute_queue_t()` is a callable factory and
  the copy path already instantiates several (`runtime/graph/hcq.py:51-54`,
  `runtime/ops_amd.py AMDComputeQueue`). The graph layer simply uses **one** compute queue today.

So the residual risk is **build complexity** (cross-layer scheduling), not feasibility. The
two-queue prototype is therefore folded into the build as **Milestone 0** (its first killable
step), reusing the same `.exec/.signal/.wait/.timestamp/.submit` HCQ work the build needs anyway.

## Verdict: **GREENLIGHT** the second-queue overlap infra

Two independent estimates corroborate:

| source | reclaimable |
|---|---:|
| analytical bandwidth slack (spike) | +38% (ceiling) |
| concurrent-capacity (Gate 1, measured floor) | +32% |
| **realizable estimate** | **~+25–32%  →  ~69–72 tok/s, ~65–68% of llama** |

The bandwidth-slack and concurrent-capacity numbers agreeing at ~+30% is a strong, cross-checked
signal. Overlap is the largest decode lever left and it is reclaimable on this stack.

### Next plan: scope the overlap build, sequenced to stay killable
1. **Milestone 0 — two-queue overlap micro-prototype** (`extra/`, standalone): launch a
   bandwidth kernel + an independent compute kernel on two `hw_compute_queue_t()` queues; confirm
   wall(concurrent) < sum and that it meets/beats the +32% floor. KILL here if it doesn't.
2. **Milestone 1 — cross-layer scheduler**: issue layer-(N+1) weight-GEMV to a second compute
   queue concurrent with layer-N non-GEMV, with cross-queue signal deps (`runtime/graph/hcq.py`
   `ji_schedule`/`_resolve_deps`). Gated, exact-output (greedy token parity), AMD-validated,
   `[runtime]`/`[codegen]`.

### Caveat
~+30% is the estimate, not a guarantee; memory-controller contention and scheduler/signal
overhead eat into it. Milestone 0 exists to measure the real fraction before the full build.

Anchors: `amd-decode-overlap-feasibility-spike-20260616.md` (+38% ceiling),
`amd-decode-sequential-tax-profile-20260616.md` (72/28 split),
`amd-decode-beyond-llama-roadmap.md` (B2).
