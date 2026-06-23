# Prefill K-Loop Schedule-Template Microkernel — Result (2026-06-23)

## Verdict: `KLOOP_SCHEDULE_TEMPLATE_MICROKERNEL_PASS` + `PREFILL_SCHEDULE_TEMPLATE_REPRESENTATION_EMITTABLE` + `PREFILL_FULL_SPEED_SEARCH_STILL_DEFERRED` (register-pool gated)
The `schedule_template` representation **is emittable**: a tiny K-loop microkernel that the detector classifies as
**PIPELINED**, is numerically correct, and does not spill — proving the new representation can generate the missing
software-pipeline primitive. But the **Tensile-class depth** (full A+B prefetch) hits the static VGPR wall, so the
whole-prefill *speed* match still needs a **register-pool** representation. No `tinygrad/` source, no model route, no
default, no whole-prefill speed claim.

## 1. Detector baseline reproduced — YES
`build_gemm_lds2(down)` → **PHASED** (0/8 global loads in the wmma span); Tensile → **PIPELINED** (3/4 + 76/76).
`KLOOP_TEMPLATE_AUTHORITY_LOCKED`.

## 2. The microkernel / template
`extra/qk_prefill_kloop_template_microkernel.py`, template `kloop_pipeline_v0`. **Emitter = `build_gemm_lds2`'s
`DBUF=1`** (double-buffer = software pipeline: prefetch the next K-block's global/LDS loads while computing the
current block's WMMA, removing the inner barrier). Shape 128×128×256 (8 K-tiles, RDNA3 WMMA, fp16 in / fp32 accum).
The representation already existed — DBUF *is* the schedule-template; this task proved it satisfies the gates.

## 3. Did it compile? — YES (`build` OK, n_insts emitted).

## 4. Numerically correct? — YES
rel_rmse **2.05e-4** ≤ 3e-4 → `KLOOP_TEMPLATE_CORRECTNESS_PASS`.

## 5. Detector classifies it PIPELINED? — YES
`build_gemm_lds2(DBUF=1)`: **PIPELINED**, **16/32 global loads + 136/160 ds-loads inside the wmma span** (vs the
DBUF=0 baseline's PHASED 0/8). The interleaving is genuine next-tile prefetch work (the double-buffer), not dead code.
`KLOOP_TEMPLATE_INTERLEAVE_PASS`.

## 6. ISA / resource envelope
`build_gemm_lds2` asserts `SCR+2 ≤ 256` (its VGPR-envelope check); the kernel is hand-asm with explicit register
allocation (no compiler spill). For this config VGPR ≈ **236 (< 256), 0 spill**, WMMA + LDS present (LDS doubled by
DBUF). `KLOOP_TEMPLATE_ISA_RESOURCE_PASS`.

## 7. Did it hit the VGPR / register wall? — Only at Tensile-class depth
The **shallow** DBUF pipeline fits (236 VGPR). The **full A+B prefetch (`PLRAB`)** — which matches Tensile's deeper
software pipeline — **overflows: VGPR 300 > 256** (`PLRAB VGPR overflow 300, needs smaller tile`). So the depth that
would actually close the ~4–5 % gap is **register-pool-blocked** under static allocation.

## 8. Does this make prefill machine-searchable now?
**Partially, and honestly:** the `schedule_template` level is now **emittable + gate-checkable** (a bounded search over
DBUF/prefetch/group-size knobs on the microkernel is now possible — the representation exists). **But the whole-prefill
*speed* search stays deferred:** (a) the shallow DBUF pipeline **regresses in-model** (prior Phase B: −0.7…−2.7 %, the
L2-contention / occupancy cost of doubling LDS), and (b) the Tensile-class deep pipeline needs a **register pool**
(dynamic VGPR lifetime) the static allocator can't express (`PLRAB` wall). So the next representation level —
`register_lifetime` — is the real gate to a whole-prefill win. This matches the prior analysis exactly:
`PREFILL_FULL_SPEED_SEARCH_STILL_DEFERRED`.

## 9. Next step
1. **Register-lifetime representation** (the actual unlock): a liveness/pool allocator so deep A+B prefetch fits ≤ 256
   VGPR — only then does a schedule-template search have a path to the ~4–5 %.
2. Until then, a *learning-only* bounded search over the **shallow** schedule-template knobs (DBUF/group-size) is
   valid for the explorer (proves search can drive the emittable representation), with **no whole-prefill speed claim**.
3. No hand-asm full prefill kernel; no model route.

## 10. Defaults / routes changed? — NONE
No `tinygrad/` source, no model route, no default flip, no whole-prefill speed claim. The emitter is an isolated tool;
`build_gemm_lds2` defaults and the shipped graph-GEMM route are untouched.

## Files changed
New: `extra/qk_prefill_kloop_template_microkernel.py` + this doc + 7 artifacts under
`bench/qk-prefill-kloop-schedule-template/` (authority, template_spec, correctness, interleave_gate,
isa_resource_gate, decision, ledger_entry) + 1 project-ledger entry.

## Git status
Clean before; adds 1 tool + 1 doc + 7 artifacts + 1 ledger line. Defaults unchanged.
