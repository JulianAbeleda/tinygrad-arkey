# Next plan — prefill fp16 WMMA LDS-tiling (the surviving high-EV arc) + the BLAS-boundary decision

Scopes the next plan after the decode space closed (q8 side-channel = Q8L-2 codegen-walled, `q8-mmvq-lifecycle-deep-result-20260619.md`).
Authority: warm pp throughput (PREFILL_V2 = 8B baseline) + dNLL for fp16. Map-first; no kernel built or routed until
a phase gate passes. This is a **decision-first** scope: Phase PWLT-0 is a strategic authority decision that is the
user's call.

## Why this is the next plan

- **Decode is exhausted** — every bounded lever shipped/refuted; the only reopening (q8 side-channel) is now
  deferred behind a codegen capability (Q8L-2). The llama-residual audit (`llama-kernel-residual-primitive-audit-20260619.md`)
  independently confirms decode headroom collapses to the same MMVQ/q8 wall and that **prefill residual needs a
  separate trace** — i.e. prefill is where headroom remains.
- **PWR-0/1 [M]:** PREFILL_V2 is the 8B authority baseline (2085 tok/s pp512, 11.35× over default). The forward is
  **~74% fp16 WMMA matmul, ~24% attention, ~2% norm** — and the WMMA matmul emits with **LDS=0** (re-reads operands
  per WMMA op), so it sits well below rocBLAS/Tensile's ~80% peak. **Quant-weight reuse is closed for 8B**
  (`qk-prefill-weight-reuse-result-20260618.md`). The lever is **fp16 WMMA operand LDS/cache-blocking**.
- **Amdahl [I]:** matmul ~74% share; a 2× matmul → **~1.6× full pp** (clears the ≥1.2× gate). The single highest-EV
  open arc in the whole project.

## The unifying insight (why this arc matters beyond prefill)

**LDS-tiled custom-kernel codegen is the shared capability behind ALL THREE remaining deep arcs:**

| deep arc | what it needs | shared capability |
|---|---|---|
| prefill fp16 WMMA matmul (~74% pp) | stage operand tiles in LDS, reuse across WMMA macro-tiles | **hand-written LDS tiling + barrier, no BEAM** |
| q8 side-channel producer (Q8L-2) | stage per-row reduce in LDS + barrier, then per-32 pass + multi-output | **same** (LDS-staged multi-granularity reduction) |
| flash-prefill attention (~24% pp at long ctx) | K/V tiles in LDS, register-resident online softmax | **same** (LDS K/V tiling) |

The BEAM-hang on gfx1100 is the *automated* path to LDS tiling. But the **hand-written** path is already proven
expressible in-repo: `AddrSpace.LOCAL→DEFINE_LOCAL` (renders `__attribute__((shared))`), `UOp.barrier→BARRIER`
(`s_barrier`), REG accumulators, WR1-3 warp/LDS reductions — `extra/gemm/amd_uop_matmul.py`,
`extra/gemm/amd_flash_attention.py`, `extra/amd_warp_reduce.py`, with passing gfx1100 tests
(`test_lds_custom_kernel_bridge.py`, `test_amd_warp_reduce.py`). So **Branch A below has a triple payoff** — if the
hand-LDS-tiled WMMA matmul works, the same capability unblocks the q8 producer and flash-prefill attention.

## Primitive

`prefill_fp16_wmma_lds_tiling` (phase: prefill). Boundary: fp16 realized weights + WMMA macro-tiles + **LDS/shared
operand tiles reused across macro-tiles** + double-buffer + warm pp + fp16 dNLL gate (already passed for PREFILL_V2).

## Phase PWLT-0 — authority-boundary decision (USER'S CALL, gates everything)

> **DECISION (2026-06-19): Branch A first** — tinygrad-internal hand-LDS WMMA, for the triple payoff and to keep the
> authority boundary pure. Fall back to Branch B (external BLAS) only if PWLT-A2 fails the isolated ≥1.5× gate.
> Next action: PWLT-A1 expressibility spike.

Two branches reach ~80% peak; they differ in **who owns the kernel**. Decide before building.

| | Branch A — tinygrad-internal hand-LDS WMMA | Branch B — external BLAS (hipBLASLt/rocBLAS) |
|---|---|---|
| approach | hand-written LDS-tiled WMMA custom_kernel (no BEAM) | call hipBLASLt/rocBLAS for the fp16 prefill tiles |
| feasibility [M] | assets + passing gfx1100 tests exist | `librocblas.so` + `libhipblaslt.so` present in `/opt/rocm-7.2.4/lib` |
| payoff | **triple** (also unblocks q8 producer + flash-prefill attention) | prefill matmul only |
| risk | deep codegen; transfer-to-model historically fragile | integration/fallback/portability; opaque to tinygrad sched |
| authority | stays pure tinygrad | crosses to external kernel boundary (portability, artifact, fallback policy) |
| reversibility | high (custom_kernel behind a flag) | medium (a BLAS dependency + fallback path) |

**Decision gate:** pick A, B, or A-with-B-as-control. **Recommendation: Branch A first** for the triple payoff and
to keep the authority boundary pure — fall back to Branch B only if PWLT-A2 fails the isolated gate. **Kill PWLT-0
to "deferred" if neither branch is funded** (the project accepts PREFILL_V2 at ~70-83% llama as the resting point).

## Branch A — tinygrad-internal hand-LDS WMMA

### PWLT-A1 — expressibility spike (no route)
Build a hand-written LDS-tiled WMMA fp16 matmul custom_kernel for ONE prefill shape (ffn_gate 512×4096→12288),
reusing `amd_uop_matmul.py` (c_regs + LDS copy + barrier) / `amd_flash_attention.py` patterns. Gate: one kernel,
correct vs fp16 matmul oracle, `__attribute__((shared))` + `s_barrier` emitted, compile sane. Kill: LDS staging /
multi-buffer / WMMA-fragment plumbing fails to compile (→ Branch B).

### PWLT-A2 — isolated gate (DEBUG=2 device time, the authority for kernels)
Compare hand-LDS WMMA vs the current PREFILL_V2 WMMA (LDS=0) on the ffn + attn-QKVO shapes, T=512. Gate: **≥1.5×**
the current WMMA, correct (fp16 rel ~2%). Kill: <1.5× isolated → the LDS-staging overhead doesn't beat the cache;
bank as a tinygrad-codegen wall → Branch B.

### PWLT-A3 — in-model warm pp (authority for prefill)
Route the winning matmul behind `PREFILL_WMMA_LDS=1` (no default flip, no decode change). Measure warm pp512 (and
pp1024 if VRAM allows) vs PREFILL_V2. Gate: **≥1.5× full pp candidate** (≥3× strong), no decode regression, fp16
dNLL ≤0.01 (already passes), fallback intact. Kill: isolated win doesn't transfer (classify: scheduling, layout
boundary, occupancy).

### PWLT-A4 — default-candidate + capability harvest
Only if A3 passes: bank the route candidate; then test whether the same LDS-tiling capability unblocks the q8
producer (Q8L-2 reopen) and flash-prefill attention — the triple-payoff harvest.

## Branch B — external BLAS boundary

### PWLT-B1 — hipBLASLt/rocBLAS bridge spike
Bridge a single fp16 GEMM call (the ffn shape) into the model path via `Ops.PROGRAM`/custom_kernel or a Device hook,
with a clean fallback when the lib is absent. Gate: correct, fallback works, no global state leak. Kill: bridge
can't be contained behind a small reviewed boundary (env-ordering, JIT capture, artifact portability all clean).

### PWLT-B2 — isolated GEMM gate
hipBLASLt/rocBLAS vs current WMMA on the prefill shapes, DEBUG=2/device timer. Gate: ≥1.5×. Kill: external GEMM not
materially faster on these shapes (small-K prefill) → external boundary not worth the dependency.

### PWLT-B3 — in-model warm pp + policy
Route behind `PREFILL_EXTERNAL_GEMM=1`. Gate: ≥1.5× full pp, no decode regression, dNLL ok, fallback tested, an
explicit portability/artifact policy recorded (the change crosses the authority boundary — `tinygrad-coding-overrides`
must bless it). Kill: integration/portability cost exceeds the pp win.

## Non-negotiable gates (both branches)
- correctness: fp16 dNLL ≤0.01 multi-window (PREFILL_V2 reference); no decode regression at any ctx.
- performance: isolated ≥1.5× current WMMA AND in-model ≥1.5× warm pp before any candidate route.
- principles: diagnostic≠shipped; no default flip; opt-in flag; fallback tested; DEBUG=2 device time for kernels,
  warm pp for the model; document refutations.

## Expected outcomes
- **Best:** hand-LDS WMMA ≥1.5× pp → prefill candidate route + the LDS-tiling capability harvested for q8 + flash.
- **Most likely:** isolated LDS win is real but in-model transfer is partial (<1.5× pp) → a tinygrad-scheduler/
  codegen diagnosis; Branch B becomes the pragmatic route.
- **Worst:** neither branch beats the current WMMA enough in-model → prefill rests at PREFILL_V2 (~70-83% llama),
  and the project's performance frontier is **closed** pending a tinygrad LDS-tiling-codegen capability (BEAM-hang
  fix) — bank as a durable D.

## Main risk
The prefill plan's history is in-model transfer failure: isolated matmuls hit good %peak but the real forward loses
locality/scheduling. The gates are deliberately **in-model warm pp**, not isolated TFLOPS, to catch this early
(PWLT-A2 isolated is necessary-not-sufficient; PWLT-A3 in-model is the real gate).

## Files (planned)
`extra/qk_prefill_wmma_lds_probe.py` (PWLT-A1/A2 + B1/B2), `extra/qk_prefill_component_breakdown.py` (reuse for
A3/B3 in-model), `bench/qk-prefill-wmma-lds-20260619/`, `docs/prefill-wmma-lds-tiling-result-20260619.md`.
Commit shape: `[test]` probes, `[codegen]`/`[runtime]` if it touches the renderer or a BLAS bridge, `[nn]` for the
flag route, `[docs]` verdict. Provenance: `qk-prefill-weight-reuse-result-20260618.md`, `amd-decode-prefill-plan.md`,
`amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`, the LDS-tiling assets above.

## The decision needed before execution
Phase PWLT-0 is the user's: **Branch A (tinygrad hand-LDS, triple payoff, deep) / Branch B (external BLAS, prefill-
only, lower-risk integration) / A-then-B / defer (rest at PREFILL_V2).** Recommendation: **A first, B as the fallback
control.**
