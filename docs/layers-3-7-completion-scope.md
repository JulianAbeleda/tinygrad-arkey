# Layers 3-7 completion: pure-codegen emitter to replace the hand asm builder

GOAL (provenance, NOT performance): make tinygrad's ISA-renderer layers 3-7 emit the SAME instruction stream the
hand `build_gemm_pipe`/`build_gemm_lds2` emits by hand, so `extra/qk/prefill/wmma.py` can be DELETED. The performance
target (5167 tok/s unpinned / 4413 pinned / ~58-68 eff TFLOPS) already exists via the gen_sched asm substrate; this
work reproduces it through pure codegen (no raw `Ops.INS`).

## The convergence target (reverse-engineered, attn_qo 512x4096x4096, per 128x128 tile-block)
246 instructions, REGISTER-BUFFERED (lds_bytes=1, NO LDS): 32x `global_load_b128`, 16x `v_wmma` (4x4 tile),
6x `s_waitcnt` incl. `vmcnt(8)` (0x23F7), 32x `v_cvt_f16_f32` + 32x `global_store_b16` epilogue, rest = index/loop.
COMPLETE for the whole substrate = the generated kernel's instruction histogram matches this blueprint AND hits
TFLOPS parity at the same clock policy.

## Reuse / centralize / modularize discipline (binding)
- EXTEND the existing ISA renderer in place (isel_load/store, _schedule, _insert_waitcnt, _frag_base) -- do NOT fork a
  parallel path. Layer work is width/mode branches inside existing dispatch, not new machinery.
- `extra/qk/prefill/wmma.py` is a SPEC REFERENCE ONLY (the file we delete) -- mirror its instruction structure, never
  import/call it.
- Single-source-of-truth helpers already centralized (reuse, do not duplicate): `_frag_base`/`_acc_base` (fragment +
  accumulator VGPR ranges), `_waitcnt_simm16` (waitcnt packing), the itemsize width-selection (Phase-1a),
  `operand_staging_policy` (REGISTER-vs-LDS; for 8B stays REGISTER -- no LDS path needed).
- Each task states: (a) existing code it extends (file:line), (b) the centralized helper it uses, (c) no-dup confirmation.

---

## LAYER 7 — tensor-core emit — COMPLETE (100%)
COMPLETE = `Ops.WMMA` lowers to `v_wmma_f32_16x16x16_f16` with correct fragment operands. DONE, AMD bit-exact
(16x16x16 rmse 8.4e-4; rolled any-K rmse 1.6e-3). No remaining work. (`isel_wmma`, `lower_inst` V_WMMA.)

## LAYER 5 — register allocation — COMPLETE (100%)
COMPLETE = the blueprint's 4x4 register tile (16 accumulators = 128 VGPRs + A/B frags) allocates within the 256-VGPR
file with NO spill, INCLUDING the epilogue store-address pressure.
- DONE: LOW per-subtile accumulator model (`_acc_base`, `WMMA_ACC_BASE=8`, per-subtile key `(id(dreg), idx.arg//8)`),
  `_vpool` exclusion. Single/rolled/chain paths unchanged. (`amd.py` accumulator region + `_frag_base` split.)
- DONE: 4x4 terminal blocker fixed in `11396c605`. The generated post-loop epilogue was reusing high WMMA scratch
  `v201/v202`; `_vpool` now reclaims the low `v1..v7` alignment pad as scalar scratch for multi-output WMMA, so epilogue
  temps avoid the high WMMA scratch band. The I0 generated 4x4 harness remu-passes and GPU-passes with no env flags.
- REUSE: `_acc_base`/`_frag_base` (no new allocator); the immediate-offset already threaded in GLOBAL_STORE lowering.
- GATE: `extra/qk/prefill/gen4x4_i0_harness.py --remu/--gpu`, plus `test/unit/test_amd_isa_wmma.py`.

## LAYER 3 — instruction selection — COMPLETE for native-ISA prefill (100%)
COMPLETE = fragment loads emit `global_load_b128` (one 128-bit load per 8-VGPR half-fragment pair), matching the
blueprint's 32 b128 loads -- NOT the current scalarized `global_load_u16` (16 narrow loads per fragment = ~16x the
memory instructions, the single biggest gap to the blueprint).
- DONE (default-on, rollback `AMD_ISA_WMMA_B128_FRAG=0`): recognizes WMMA operand carriers whose 16 fp16 lanes are two contiguous
  8-half spans, then emits two `global_load_b128` instructions directly into the pinned 8-VGPR fragment instead of
  scalar half loads plus `v_pack`.
- DONE: the route-shaped native-ISA prefill form (`a @ b.transpose()`) folds both operands: the 4x4 generated stream has
  16 `global_load_b128`, 0 `v_pack_b32_f16`, 0 `global_load_u16`, and 16 `v_wmma`; the GPU custom-kernel route-shaped
  run passes (`rmse=0.001664`, `nan=0`). The plain `a @ b` unit intentionally keeps B column-strided and still packs,
  because that is not the route layout.
- NOTE: cooperative-B ownership remains a HIP/postrange medium-stage issue, not a native-ISA b128 blocker.
- REUSE: extend the EXISTING `isel_load`/`lower_inst` GLOBAL_LOAD dispatch as a width branch (same entry point as the
  scalarized path + the Phase-1a u16 branch); the 4-VGPR range comes from `_frag_base`. No new load family.
- GATE: default `python3 extra/qk/prefill/gen4x4_i0_harness.py --gpu` passes on AMD (`nan=0`, `rmse=0.00156`);
  `test_amd_isa_wmma.py` asserts default b128, rollback, and route-shaped full-b128 behavior.

## LAYER 6 — waitcnt — ~40% (full-drain works; targeted vmcnt(n) missing)
COMPLETE = emits targeted `vmcnt(n)` (the blueprint's `vmcnt(8)`) so next-tile loads stay in flight during compute --
NOT a full-drain `s_waitcnt(0)` after every load.
- DONE (correctness prototype, opt-in): `AMD_ISA_WAITCNT_TARGETED=1` changes `_insert_waitcnt` to track pending VMEM/LGKM
  loads as ordered span lists and emit partial waits for the newest dependent load. Exact 4x4 GPU harness passes with
  targeted waits alone and with b128+targeted waits.
- DONE (regression fix, still opt-in): scalarized `v_pack_b32_f16` consumers now coalesce pending VMEM to a single
  default-equivalent drain instead of one partial wait per pack; the prior 31-wait plain 4x4 targeted stream drops to
  10 waits and the schedule-table AMD gate returns to the normal band.
- REMAINING: performance-valid promotion. Targeted waitcnt is correct and no longer catastrophically regresses, but it
  still does not beat the default/full-drain route or create hand-class overlap. Keep default full-drain until L4 proves
  a real load/compute cadence.
- REUSE: EXTEND `_insert_waitcnt` in place (reuse its pend/hazard tracking); emit through the centralized
  `_waitcnt_simm16`. The span-aware `_inst_regs` (R1 fix, done) already makes fragment-range hazards correct -> this is
  unblocked. No parallel waitcnt pass.
- GATE: bit-exact vs full-drain first; promotion requires DEV=AMD TFLOPS lift and disasm showing fewer/coalesced
  targeted waits near WMMAs.

## LAYER 4 — instruction scheduling / DBUF — ~30% (list-sched + span-aware done; no overlap yet)
COMPLETE = the loop overlaps next-tile `global_load_b128` with current-tile `v_wmma` (software-pipelined /
double-buffered), matching the blueprint's load-ahead + `vmcnt(8)` cadence.
- CURRENT BLOCKER (2026-07-07): the generated native-ISA route has one live resident A/B fragment bank. Structurally it
  is `load all fragments -> wait -> v_wmma all subtiles`, so `_schedule` and targeted waitcnt have no next-K fragment
  bank to overlap with current-K WMMAs. The hand `build_gemm_pipe` shape is different: `load F1 -> wait/use F0 -> load
  F0 -> wait/use F1`, with explicit F0/F1 fragment banks plus prologue/tail.
- `PREFILL_DBUF=1` is not a current escape hatch for the direct native-ISA route: forcing it on the route-shaped 4x4
  AST fails in `isel_wmma` (`C init lane 0 is Ops.LOAD, expected CONST`) because the peeled second K-copy presents the
  rolled accumulator as a load-headed chain that the native-ISA WMMA lowering does not accept today.
- NEXT IMPLEMENTATION PATH: first make unroll-by-2 rolled-accumulator WMMA chains lower correctly; then add phase-aware
  A/B fragment allocation or a lower-footprint software-pipeline representation. A literal second 4x4 resident A/B bank
  costs another 64 VGPRs on top of 128 C + 64 current A/B, leaving no room for scratch/address registers, so this needs a
  constrained design rather than a naive duplicate of the hand layout.
- SCOPE: `docs/native-isa-l4-software-pipeline-scope.md` is the exhaustive L4 task list and candidate matrix.
- REUSE: existing `_schedule` + existing `_prefill_dbuf_peel` (codegen owns the shape, renderer owns sched+wait). No dup.
- GATE: disasm shows next-iter loads above current WMMAs, each WMMA preceded by targeted `vmcnt`; TFLOPS -> hand class.

---

## Ordered task list (dependency order; each gated bit-exact-first, same-clock TFLOPS)
1. **DONE: L5 epilogue pressure** -> generated 4x4 now runs on GPU; yesterday's NaN roadblock is closed.
2. **DONE: L3 b128 loads** -> route-shaped native-ISA prefill folds both A and B.T into direct b128 fragment loads.
3. **PROTOTYPE: L6 targeted vmcnt** -> correctness passes, perf regresses; needs coalescing before promotion.
4. **L4 DBUF overlap** -> next-tile loads hide behind compute -> converge on the 246-instruction blueprint / hand TFLOPS.
5. **Delete** `extra/qk/prefill/wmma.py` + the raw-`Ops.INS` route; confirm `PURE_MACHINE_SEARCH_ONLY=1`.

## Completion definition (the whole substrate)
COMPLETE = a generated (no `Ops.INS`) prefill GEMM whose per-block instruction histogram matches the hand blueprint
(32 b128 loads / 16 v_wmma / targeted vmcnt / epilogue) AND matches hand TFLOPS at the same clock policy, for every
8B role shape (attn_qo/kv 4096/1024, ffn gate_up 12288, ffn_down) -- then wmma.py is deleted. Layers 1-2 (searched
schedule) and 8-9 (assemble/run) already deliver; 3, 5, and 7 are done; 4/6 above are the exhaustive remaining ISA
handtrace-parity set.

## Post-L3/L5 benchmark readout (2026-07-07)
- `prefill_v2_schedule_table_gate --run-amd --pin-clock --compact`: PASS, measured 35.31 TFLOPS for 4096x4096 and
  37.00 TFLOPS for 5120x5120.
- Route-shaped native-ISA `a @ b.transpose()` custom kernel: PASS, 831 final instructions, 16 `global_load_b128`,
  0 `v_pack_b32_f16`, 0 `global_load_u16`, 16 `v_wmma`, `rmse=0.001664`, `nan=0`.
- `prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact`: still BLOCKED. Baseline table-local is
  35.46 TFLOPS; B tile staging is 35.21 TFLOPS; cooperative B executes but the rewrite is skipped because source B has a
  non-lane `GLOBAL` range outside warp+reduce.
- Targeted-wait regression fix: `AMD_ISA_WAITCNT_TARGETED=1` + b128 route passes the 4x4 GPU harness; scalar-pack 4x4
  targeted waits drop from 31 to 10, and the schedule-table AMD gate returns to 35.23/36.59 TFLOPS instead of the prior
  ~15 TFLOPS regression.
- Conclusion: native-ISA L3 is closed and L6 is correctness-valid but not promotable. The remaining terminal blocker is
  L4: codegen must expose a two-phase/pipelined fragment shape before waitcnt can hide load latency. Cooperative-B
  ownership is still useful for the HIP medium-stage route, but it is no longer blocking native ISA b128 parity.

Reference: reverse-engineered blueprint (docs + tmp/reverse_lds2.py); hand trace (prefill_gen_sched_gemm 85% of forward,
ffn gate/up 42%); census `docs/prefill-substrate-layer-census-20260706.md`; `docs/track-b-100pct-scope.md`.
</content>
