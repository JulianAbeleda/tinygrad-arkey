# Tensile Roadmap Scope

Date: 2026-06-20

Artifact:
`bench/qk-tensile-primitive-transfer/roadmap.json`

Command:

```bash
python3 extra/qk_tensile_roadmap_scope.py
```

Verdict:
`PASS_TENSILE_ROADMAP_SCOPED`

## Purpose

Pause kernel work and scope **what we actually need from Tensile** before any more implementation. This
builds on PTM-0 (`tensile-primitive-transfer-matrix-scope-20260620.md`, which froze the 10-row transfer
matrix) and splits the work into **three tracks** plus phases **PTM-1..PTM-5**. It does not run a kernel,
extract an artifact, or claim performance.

The discipline this enforces: **every future Tensile experiment must name the matrix row it is proving,
the local artifact it uses, and the pass/fail criterion** that would let the row transfer to prefill or
decode. No more looping on "Tensile" as one opaque object.

## Track 1 — Tensile Explanation (why the selected kernel works)

- **Official parameter map** (each knob → primitive row): WorkGroup/WGM=8, ThreadTile=`4_64`,
  MacroTile=`128x128x16`, MatrixInstruction=`16x16x16x1`, PGR=1, PLR=1, DepthU=16, GLVWA/GRVW=4,
  1LDSB (double LDS buffer). Sources: the four official Tensile docs already cited in `scope.json`.
- **Selected kernel metadata:** rocBLAS symbol
  `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_...PGR1_PLR1_...WGM8`, code object
  `Kernels.so-000-gfx1100.hsaco`. Per-role isolated TFLOPS: ffn_gate_up 65.6/73.0, ffn_down 69.8/71.7
  (StreamK, 132-byte kernarg), attn_q_o 59.3/67.2. No workspace, no layout copies. Cumulative
  `full_pp_speedup` = **1.393×** if all three roles are replaced.
- **Timing-authority rule:** per-primitive throughput attribution and absolute TFLOPS are **not validated
  across measurement methods**; the robust prefill number is the **e2e 1.84× byte-identical** (47%→87%
  llama). Do not cite the per-primitive "3.3×/6.6×" scoreboard as settled.

### Two corrections this track MUST state (cross-file inconsistencies)

1. **The captured 43.026 TFLOPS authority kernel is NOT the Tensile LDS kernel.** It is tinygrad's own
   global-direct WMMA authority (`r_16_192_32_...`, `ds_load_b128=0`, `v_wmma=64`). Do not conflate
   "43 TFLOPS captured authority" with the Tensile schedule.
2. **v_wmma count is scope-dependent:** 13810 (whole `.so` disasm) vs **80** (isolated selected-function
   body). Cite the per-body **80 v_wmma + 256 v_fma_mix**; the selected function uses **`ds_store_b64`**
   (not `ds_store_b128`) for global→LDS stores.

## Track 2 — Prefill Transfer (native reproduction vs external artifact)

- **PTM-1 same-harness authority bridge** is the gating first step. Time the captured 43.026 authority
  kernel **and** the current P8 LDS (18.4) / no-LDS (17.9) candidates under **one** synchronized or
  device-timestamp harness. This resolves whether the 43↔18-21 TFLOPS gap is real kernel quality or a
  harness/identity mismatch. No mixed-kernel/mixed-harness comparison.
- **Forced single native candidate (decided in PTM-2):** exactly one of
  `software_pipelined_k_loop`, `spill_free_accumulator_resource_policy`, or `timing_launch_correction`.
  The codegen oracle names the missing capability as the **software-pipelined K-loop** with
  double-buffered global→LDS→reg prefetch.
- **Standalone LDS is CLOSED** unless paired with K-loop overlap + waits + resource scheduling
  (A3 P2/P3 refuted naive LDS at ~6 vs ~32 TFLOPS; net-negative on IC-served global reads).
- **Gates:** correctness (RMSE vs authority), resource (scratch/private 0, acceptable VGPR/occupancy),
  performance (same-harness TFLOPS vs the PTM-1-bridged baseline only).
- **Artifact/dependency policy:** native codegen transfer and the external rocBLAS `.co` route are
  **separate projects**. The `.co` route is policy-gated (vendoring vs no-deps), default
  `PREFILL_TENSILE_GEMM=0`, ~87% llama byte-identical. Dependency-free fallback rests at WMMA ~47% +
  shipped concrete-KV 1.24×.

## Track 3 — Decode Applicability (does any Tensile primitive touch q8 decode)

Decode does **not** match Tensile's dense-fp16 domain — it is q8/MMVQ, batch-1, quantized,
lifecycle/consumer-bound. Current readiness is `ROADMAP_ONLY`: N2 candidate count `0`, max timing-grade
movement `14.087µs` (< the 30µs gate), oracle gap `73.109µs`. Before **any** transfer claim, a q8 row
must clear all of:

- q8 role-joined gate/up evidence
- same-binary primitive ablation
- `>=30µs` timing-grade movement
- W==D quality unchanged
- packed q8 format preserved
- **no dense GEMM substitution from prefill-only evidence**

## Phases

| Phase | Name | Status | Minimum pass |
|---|---|---|---|
| PTM-1 | same-harness authority bridge | **next** | 43.026 authority + current P8 candidates timed under one common harness |
| PTM-2 | prefill primitive decision | blocked on PTM-1 | choose exactly one native row; standalone LDS disallowed |
| PTM-3 | native candidate scope | blocked on PTM-2 | scope the single chosen row only (dataflow + resource + gates) |
| PTM-4 | external artifact policy scope | parallel/policy | decide vendored `.co` vs no-deps; fallback, default, provenance, quality gates |
| PTM-5 | decode transfer gate | blocked until a q8 row clears 30µs | q8 row with `>=30µs` movement, W==D, role-joined gate/up evidence |

## Stop Rules

- Do not run another P8 kernel variant unless it names a matrix row.
- No standalone LDS work; LDS only with overlapped movement + resource control.
- No mixed-kernel / mixed-harness TFLOPS comparison.
- Do not treat the external `.co` artifact route and native tinygrad codegen transfer as one project.
- Do not reopen q8 transfer yet.
- Every future experiment must name the primitive row it is proving.

## Next

Run **PTM-1**: same-harness authority bridge.
