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

## LAYER 3 — instruction selection — ~75% (WMMA done; b128 loads missing = the 16x inflation)
COMPLETE = fragment loads emit `global_load_b128` (one 128-bit load per 8-VGPR half-fragment pair), matching the
blueprint's 32 b128 loads -- NOT the current scalarized `global_load_u16` (16 narrow loads per fragment = ~16x the
memory instructions, the single biggest gap to the blueprint).
- REMAINING: add a b128 load path in `isel_load` (`amd.py:~312`) / `lower_inst` GLOBAL_LOAD (`amd.py:~683`): when a
  contiguous 4-VGPR (128-bit) fp16 fragment slice is loaded from a 16-byte-aligned address, emit one `global_load_b128`
  into the 4-VGPR range instead of N scalarized loads. Import `global_load_b128` (autogen has it). Also `ds_*_b128`
  only if an LDS path is ever taken (NOT for 8B register-buffered).
- REUSE: extend the EXISTING `isel_load`/`lower_inst` GLOBAL_LOAD dispatch as a width branch (same entry point as the
  scalarized path + the Phase-1a u16 branch); the 4-VGPR range comes from `_frag_base`. No new load family.
- GATE: DEV=AMD:ISA GEMM bit-exact with b128 loads; DEBUG=2 disasm shows `global_load_b128` (0 scalarized fp16 loads);
  measured TFLOPS jumps (memory-op count drops ~16x on the load path).

## LAYER 6 — waitcnt — ~40% (full-drain works; targeted vmcnt(n) missing)
COMPLETE = emits targeted `vmcnt(n)` (the blueprint's `vmcnt(8)`) so next-tile loads stay in flight during compute --
NOT a full-drain `s_waitcnt(0)` after every load.
- REMAINING (already scoped as B1.L6): `_insert_waitcnt` (`amd.py:~844-976`) tracks pending loads as issue-ordered
  lists (not sets); at a consumer, wait `vmcnt(count-of-loads-issued-after-the-newest-dependency)`; carry pending loads
  across the loop backedge; keep full-drain at barrier/endpgm/store. Behind a flag; default full-drain preserved.
- REUSE: EXTEND `_insert_waitcnt` in place (reuse its pend/hazard tracking); emit through the centralized
  `_waitcnt_simm16`. The span-aware `_inst_regs` (R1 fix, done) already makes fragment-range hazards correct -> this is
  unblocked. No parallel waitcnt pass.
- GATE: bit-exact vs full-drain (DEV=PYTHON), then DEV=AMD TFLOPS lift; disasm shows `vmcnt(n>0)` before WMMAs.

## LAYER 4 — instruction scheduling / DBUF — ~30% (list-sched + span-aware done; no overlap yet)
COMPLETE = the loop overlaps next-tile `global_load_b128` with current-tile `v_wmma` (software-pipelined /
double-buffered), matching the blueprint's load-ahead + `vmcnt(8)` cadence.
- REMAINING (B1.L4): the existing `_schedule` list scheduler already front-loads height-200 loads; the DBUF unroll-by-2
  peel (`postrange.py::_prefill_dbuf_peel`, WMMA-role-guarded) puts two K-copies in one block; combined with Layer-6
  targeted waitcnt the overlap should emerge with NO new modulo pass. Fixes already landed: `_sched_lat` v_wmma=16.
  Prove overlap; escalate to an explicit software-pipeline pass ONLY if measurement falls short (Fable-review first).
- REUSE: existing `_schedule` + existing `_prefill_dbuf_peel` (codegen owns the shape, renderer owns sched+wait). No dup.
- GATE: disasm shows next-iter loads above current WMMAs, each WMMA preceded by targeted `vmcnt`; TFLOPS -> hand class.

---

## Ordered task list (dependency order; each gated bit-exact-first, same-clock TFLOPS)
1. **DONE: L5 epilogue pressure** -> generated 4x4 now runs on GPU; yesterday's NaN roadblock is closed.
2. **L3 b128 loads** -> ~16x fewer load instructions (largest single TFLOPS lever toward the hand trace).
3. **L6 targeted vmcnt** -> remove full-drain serialization.
4. **L4 DBUF overlap** -> next-tile loads hide behind compute -> converge on the 246-instruction blueprint / hand TFLOPS.
5. **Delete** `extra/qk/prefill/wmma.py` + the raw-`Ops.INS` route; confirm `PURE_MACHINE_SEARCH_ONLY=1`.

## Completion definition (the whole substrate)
COMPLETE = a generated (no `Ops.INS`) prefill GEMM whose per-block instruction histogram matches the hand blueprint
(32 b128 loads / 16 v_wmma / targeted vmcnt / epilogue) AND matches hand TFLOPS at the same clock policy, for every
8B role shape (attn_qo/kv 4096/1024, ffn gate_up 12288, ffn_down) -- then wmma.py is deleted. Layers 1-2 (searched
schedule) and 8-9 (assemble/run) already deliver; 5 and 7 are done; 3/4/6 above are the exhaustive remaining ISA
handtrace-parity set.

## Post-L5 benchmark readout (2026-07-07)
- `prefill_v2_schedule_table_gate --run-amd --pin-clock --compact`: PASS, measured 35.31 TFLOPS for 4096x4096 and
  37.00 TFLOPS for 5120x5120.
- `prefill_graph_gemm_medium_stage_gate --run-amd --pin-clock --compact`: still BLOCKED. Baseline table-local is
  35.46 TFLOPS; B tile staging is 35.21 TFLOPS; cooperative B executes but the rewrite is skipped because source B has a
  non-lane `GLOBAL` range outside warp+reduce.
- Conclusion: no new whole-prefill bench is useful yet. The next codegen work is still L3 b128 for ISA handtrace parity
  and/or the route-bound cooperative-B ownership fix for 8B medium staging.

Reference: reverse-engineered blueprint (docs + tmp/reverse_lds2.py); hand trace (prefill_gen_sched_gemm 85% of forward,
ffn gate/up 42%); census `docs/prefill-substrate-layer-census-20260706.md`; `docs/track-b-100pct-scope.md`.
</content>
