# Scope - CG-W1.5: diagnose the DOMINANT warmstarted in-model PREFILL_V2 matmul (validate the address-lowering premise)

Gating step before any renderer work (the correction in `prefill-address-lowering-renderer-arc-plan-20260619.md`):
the CG-W "address-lowering" diagnosis was on the `amd_copy` proxy (no opts); the in-model kernels I captured via
`m.logits` had no WMMA + offset-clean addressing — but that was likely the **non-warmstarted fallback**, not the real
`prefill_v2_jit` path. CG-W1.5 captures + diagnoses the **actual dominant warmstarted in-model matmul** to answer:
**(a) does it use WMMA tensor cores? (b) is it address-ALU-bound (like the proxy) or something else?** The answer
retargets (or confirms) the entire prefill-codegen plan.

## Tooling assessment (do we have it? — YES for the core questions)
| need | tool | status |
|---|---|---|
| capture the warmstarted kernels | `get_runtime` hook during `model(v2_chunk, sp, temp)` (the measure path: concrete-512 chunk → `prefill_v2_jit` + warmstart in `__call__`) | YES (the fix: use `model(...)`, not `m.logits`) |
| per-kernel TIME (find the dominant matmul) | `DEBUG=2` per-kernel ms on the first/non-replayed run; `rocprofv3 --kernel-trace` backup | YES |
| ISA: WMMA presence, instruction mix, addressing | `llvm-objdump` on the captured lib | YES (used throughout) |
| cycle/stall/occupancy attribution | rocprofv3 PMU counters | LIMITED on gfx1100 — **not needed**; WMMA-presence + instruction-class + time answers the question |

The only tooling gap (deep PMU stall counters) does not block CG-W1.5: the binding constraint is read off the
instruction mix + whether WMMA fires, exactly as the amd_copy/decode diagnoses were done.

## Method
1. Mirror the measure's warmstart setup (`vsp`/`vtk`/`t`/`temp`/`sp`, `v2_chunk = t[:, sp:sp+512]`), warm it once
   (JIT capture), then run `model(v2_chunk, sp, temp)` under (a) a `get_runtime` hook capturing `(function_name,
   lib, global_size)` and (b) `DEBUG=2` for per-kernel ms.
2. Rank kernels by total device time (the per-layer ffn/attn matmuls × 36 dominate). Identify the dominant matmul(s).
3. `llvm-objdump` the dominant matmul lib → answer:
   - **WMMA?** count `v_wmma`. If 0 → the warmstarted matmul is *not* using tensor cores (the gap is "TC not firing",
     a different lever than address-lowering).
   - **binding instruction class** — `v_wmma`/`v_fma` vs `v_mov`/`v_add` (addressing) vs `v_cvt`/`v_and` (dequant) vs
     `global_load`/`ds_load` (memory). Compare per-WMMA (or per-FMA) ALU overhead to the amd_copy proxy.
   - **addressing** — per-load `off` (computed) vs `offset:` immediates.

## Decision tree
- **WMMA + address-ALU-bound (like proxy)** → the address-lowering plan is VALIDATED in-model → proceed with the
  renderer arc (CG-W2…).
- **WMMA + offset-clean (NOT address-bound)** → address-lowering is NOT the in-model lever → retarget to the actual
  binding class (WMMA-issue density / occupancy / epilogue) and re-scope.
- **No WMMA (scalar v_fma matmul)** → the warmstarted matmul isn't firing tensor cores → the in-model 80%-llama gap is
  "TC not firing for these shapes" → the lever is the TC-firing condition (warmstart opt / shape), NOT address-lowering
  and NOT WMMA-issue. Biggest retarget.

## Gates / constraints
- read the binding constraint from the dominant-by-TIME kernel (not by lib size — last time's mistake was ranking by
  lib size and grabbing the wrong/fallback kernels).
- confirm it is the warmstarted path (warmstart `apply` > 0 in the run).
- no model/default change; diagnosis only.

## Deliverable
`extra/qk_prefill_inmodel_matmul_diag.py` (capture + per-kernel-time + disasm of the dominant matmul),
`bench/qk-codegen-wmma/inmodel_matmul.json`, result appended to `prefill-codegen-wmma-issue-result-20260619.md` (or a
new CG-W1.5 result) with the verdict on the decision tree → which prefill-codegen plan (if any) is the real one.
