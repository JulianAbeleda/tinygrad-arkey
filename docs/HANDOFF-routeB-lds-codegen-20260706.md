# HANDOFF - Route B (fp16 LDS-staging codegen) — reuse map & first steps

Date: 2026-07-06. Branch: `master`. Goal: replace the shipped hand kernel `build_gemm_lds2_q4k`
(extra/qk/prefill/wmma.py, 808 tok/s) with tinygrad CODEGEN so we can delete it (no-hand-kernel mandate).
Route B = replicate the hand kernel's algorithm (Q4_K decode -> fp16 -> LDS-stage the DECODED tile -> WMMA)
in codegen via `bufferize(LOCAL, removable=False)`. int8/MMQ route is DEAD (throughput-neutral on RDNA3).

This doc is the VERIFIED reuse map (Fable-5 audit, 2026-07-06) so the next session does NOT recreate the wheel.

## GO/NO-GO decider (the cheap first experiment)
Does cooperative LDS input-staging lift PLAIN fp16 codegen WMMA at attn_qo (512x5120x5120) from the Track-1
best ~40 -> ~58+ TFLOPS? **GO(>=58)** -> build Route B (add Q4_K decode fusion, match 808, delete hand kernel).
**NO-GO(<=50)** -> keep the hand kernel (it's a legitimately good single-buffer LDS kernel).

## CRITICAL: the design doc (codegen-wmma-lds-staging-design-20260705.md) is STALE
- `Ops.BUFFERIZE` -> now `Ops.STAGE`. `.bufferize(...)` builds a STAGE (ops.py:597).
- `removable=` is NOT a `.bufferize()` kwarg -> it's a field of `BufferizeOpts` (indexing.py:38).
- All doc rangeify line numbers drifted (bufferize_to_store 394->397; remove_bufferize 239->242).
- `gpudims.py` moved: now `tinygrad/codegen/gpudims.py` (not `late/`).
- The doc's `bufferize(*warp, LOCAL, removable=False)` spelling WILL NOT RUN. Use the recipe below.
- **"milestone-1 SOLVED + VERIFIED 2e-4 on AMD" is UN-REPRODUCIBLE**: the repro (scratchpad/localinput_test.py)
  is gone, and there is NO committed test/gate/artifact. Treat as a design claim; re-derive + re-verify.

## Reuse map (EXISTS+USABLE / EXISTS+STALE / MUST-RECONSTRUCT), file:line
EXISTS+USABLE (import/reuse directly):
- `.bufferize(*rngs, arg=BufferizeOpts(...))` -> Ops.STAGE — ops.py:597
- `BufferizeOpts(device, addrspace, removable)` — indexing.py:38
- `bufferize_to_store(ctx, x, idx, allow_locals=True)`; LOCAL branch auto-adds the barrier
  (`buf.after(do_store.barrier())`) — rangeify.py:397, LOCAL branch 428-432
- **LOCAL entry point that actually fires:** `pm_add_buffers_local` (allow_locals=True) — rangeify.py:463-464;
  scheduled at codegen/__init__.py:132 (AFTER expander@129, AFTER postrange/apply_opts@108).
  TRAP: `pm_add_buffers` (rangeify.py:447) uses allow_locals=False and will NOT lower a LOCAL stage — wrong path.
- **Live template to MIRROR:** `fix_group_for_reduce` LOCAL-bufferize — expander.py:132-159, emit line 142.
- `UOp.contract(...)` primitive (for the explicit CONTRACT fold) — ops.py:536-538; already used by
  `lower_shaped_wmma` (rangeify.py:32 `s[u].contract(u)`).
- Track-1 schedule search + table (THE ~40 baseline AND the clean-measurement harness) —
  extra/qk/prefill_v2_schedule_search.py (`_worker`, search.py:55-84); prefill_v2_schedule_table.json
  (5120x5120=40.66, 1024x5120=36.71, 17408x5120~48).
- Track-1 warmstart apply mechanism: `postrange._WARMSTART_OPTS` dict + `_warmstart_stats["apply"]` assert.
- Renderer WMMA-helper dedup (commit 84efd5172) — helps Route B (multiple WMMA per kernel).
- `lower_shaped_wmma` alt WMMA surface — rangeify.py:25-37 (fallback if TC-heuristic staging fights you).
- `q4k_wmma_tiled_no_hand_kernel_gate.py` — reuse as the DELETION GUARD (forbidden-token scan).

MUST-RECONSTRUCT (doc-only, primitives exist):
- WARP address-key (stage keyed on the WARP parallel special, not a loop range — avoids the END-on-sequential
  CFG cycle, linearizer assert). WARP lowering in tinygrad/codegen/gpudims.py.
- Explicit CONTRACT fold of within-lane vector axes (frag + non-frag UPCAST/UNROLL) into the staged element
  (implicit vectorization mislays the fragment when N&K both large -> rel_rmse ~0.14).
- Pass `removable=False` (else remove_bufferize @248 folds the stage away -> dead decl + bare barrier).

EXISTS but IGNORE for Route B:
- `cooperative_stage_lanemap.py` — decode coalesced-load context, "no codegen wiring." Reuse the IDEA
  (thread-owns-contiguous-chunk) for the cooperative partition, NOT the class.
- int8 MMQ atom + q4k_wmma_tiled/scheduler gates — the int8 route is dead; keep the atom as a correctness asset.
- No `LOCALINPUT` flag / apply_opts input-staging hook exists — it was monkeypatch-only and is GONE.

## Ordered first steps (next session)
1. Reproduce the ~40 baseline via `prefill_v2_schedule_search._worker` at 5120x5120 (sanity + confirms the
   clean-measurement harness: applies table schedule via _WARMSTART_OPTS, asserts apply!=0, TinyJit warm min-of-K,
   no copy pollution). NOTE: a standalone `a@b.T` gets the DEFAULT schedule (~3 TFLOPS) + copy pollution — do NOT
   measure that way.
2. Reconstruct the fp16 B-operand LOCAL stage, mirroring expander.py:142:
   `staged = b_operand.contract(<within-lane frag+extra axes>).bufferize(*warp_ranges,
   arg=BufferizeOpts(addrspace=AddrSpace.LOCAL, removable=False)).index(*warp_ranges)`, rebuild the per-lane
   vec, feed the WMMA src. Ensure it lands in the pm_add_buffers_local stage (barrier comes free).
3. Correctness: DEV=PYTHON is a FALSE-POSITIVE oracle -> verify on DEV=AMD tiny (<=256) THEN large (large N&K
   exposes the implicit-vec bug). Milestone-1a = redundant full-tile store (expect flat perf, confirms wiring).
4. Milestone-1b = cooperative partition (store_keys != read_keys) for the traffic cut — THIS is what must move
   40->58+. Time at 5120x5120 -> GO/NO-GO.
5. Only on GO: SECOND gate (untested) — does `bufferize` place the Q4_K DECODE subgraph at the store site
   (decode-once-per-tile, not per-read)? Then wire into Q4_K, match 808, delete build_gemm_lds2_q4k
   (guard with q4k_wmma_tiled_no_hand_kernel_gate.py). Never kill a live DEV=AMD run (MES-ring wedge).
