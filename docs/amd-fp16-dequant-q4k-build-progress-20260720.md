# BUILD PROGRESS / CONTINUATION: generated fp16-dequant Q4_K primitive (AMD)

Living doc — resume the implementation from here if context is lost. Plan: [`amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md`](amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md). Decision/context: handoff §1.16 (AMD primitive = fp16-dequant-in-register), §1.15 (occupancy routing). Last updated 2026-07-20.

## One-line state
Converting the MMQ generator int8→fp16-dequant IN PLACE (per §1.16; int8 source preserved in git history for later NVIDIA recovery = task #15). Kernel now RENDERS correct-structure fp16 (16 KB LDS, f16 WMMA, 3-buffer ABI); the register-pressure spill is SOLVED; currently closing the last compile blockers. **Numerical correctness NOT yet measured** (that's phase 3/4).

## Commits on master (in order)
- `4eef43945` [wip] convert generator int8→fp16-dequant (renders, not spill-free) — phases 1-2.
- `f6c5c4f6a` [wip] chain accumulator across K32 groups (fixed 512→64 VGPR accumulator blowup; 64 chain heads → 8, matching hand kernel).
- `27e1f0f25` [amd] generalize `_frag_b128_loads` lane stride by dtype itemsize (keeper; general bug — was hardcoded 1-byte, only correct for int8).

## UNCOMMITTED in working tree (from the running spill-fix agent) — DO NOT LOSE
- `extra/qk/mmq_llama_oracle_recurrence.py` `_fragment_at`: load raw bytes with correctly-typed ptr (`uint8.vec(esz)`) then `.bitcast(half)` the VALUE (was `UOp.index(dtype=half)` which mislabeled a uint8 ptr → legalizer shattered into 16 uchar scalar loads).
- `tinygrad/renderer/isa/amd.py` `_wmma_half_addr`: unwrap the per-element `Ops.BITCAST` so `_frag_b128_loads` recognizes the vectorized load.
- **Effect (verified on cheap 2-group probe): scalar DS_LOAD 1950→0 (now 28 vectorized DS_LOAD_B128), peak vregs 491→63, full compile 6.5min→75s.** THE REGISTER-PRESSURE WALL IS DOWN.
- **BUT: regresses 8 tests in `test/unit/test_amd_isa_wmma.py`** (int8 fragment path) — must be fixed before commit. Agent is on it.

## CURRENT BLOCKER (after DS_LOAD fix)
New, smaller: SGPR allocation fails on `v0 = Ops.PARAM(slot=0)` (the `output` kernarg pointer) at uop=1, **zero contention** (peak_live_virtual=1). Not shape-dependent, fires before any group/decode/WMMA. Hypothesis (agent): the 17-slot SGPR-pair candidate set overlaps a `FixedRegisterUse`-reserved ABI slot (workgroup_coords/loop_counter) invisible to the virtual allocator — look in the kernarg/ABI candidate construction in isel + `regalloc.py` fixed-ABI reservations. Was previously masked by the huge VGPR DS_LOAD spill.

## Verification commands
- Full compile (~75s now): `build_llama_five_buffer_full_kernel(128,128,256)` then `compile_llama_five_buffer_full_kernel(k)`; check `.emitted` (True=spill-free), `.blocker`, `.program.arg` (VGPR/LDS). Confirm ISA has `v_wmma_f32_16x16x16_f16`, LDS group_segment 16384.
- Cheap ~65s probe: a 2-K32-group synthetic mirroring `_full_grid_sink` (isel/regalloc only, no GPU) — reproduces the same pressure/blocker at 1/8 cost.
- Pressure introspection: `REGALLOC_DEBUG` env prints peak live vregs + PEAK_CONTRIBUTORS + spill point.

## What's left (ordered)
1. **Finish phase 2b (spill-free):** fix the 8 `test_amd_isa_wmma.py` regressions (fragment-load fix must cover int8 AND fp16; the BITCAST unwrap must be a strict no-op when no bitcast); fix the SGPR PARAM(0) blocker → `emitted=True`, 0 spills. Then commit.
2. **CORRECTNESS FAIL-FAST (pulled forward):** the moment it emits, before building the family, do a small-shape numeric parity of the dequant vs the authority. The `.bitcast(half)` reinterprets raw bytes as half — if byte order/layout is off it compiles but outputs garbage. MUST verify before trusting.
3. **Phase 3 — correctness authority + CPU parity:** author `ffn_gate_up_fp16_dequant_reference` on the GGML `d*sc*code−dmin*mn` math (`tinygrad/llm/gguf.py:76-84` / `extra/qk/layout.py:157` `q4_k_reference`; existing analogue `mmq_ffn_gate_up_guarded_correctness.py:357-375` `ffn_gate_up_direct_dense_reference`). Feed into the same `_validate_numeric_comparison`/`_validate_full_comparison` (`:223-299`). NOT the int8 authority (`mmq_q4k_q8_reference.py`) — different rounding path (§2.5, needs new authority + C0A sign-off). Accept: `rtol=atol=3e-3`, zero mismatch, finite.
4. **Phase 4 — new frozen family + GPU:** new 2-3-buffer ABI family (checklist = plan PART III; canonical ABI constants `extra/qk/prefill/frozen_exact_role_runtime.py:37-39`); C4 no-target canary; then guarded reduced-grid ladder `(1,1,1)…(8,4,1)` zero-mismatch, then the FULL 544-wg dispatch that MUST now pass (16 KB LDS) where int8 wedged at 64.
5. **C6-C8** (full correctness / memory / timing → CERTIFIED_WIN or FALLBACK).

## Follow-ons (separate tasks)
- #10: occupancy-based route admission axis (§1.15) — routes int8→NVIDIA, lean→AMD from facts.
- #15: recover int8 generator as renamed NVIDIA-only, NOT-selectable modules (source is at pre-conversion git history / was HEAD `51cce914c`).

## Key files
- `extra/qk/mmq_llama_candidate_plan.py` (`_geometry` two fp16 16KB regions, `_rdna3_f16_tc`).
- `extra/qk/mmq_llama_oracle_recurrence.py` (`_fragment_at` bytes+bitcast; fp32-accumulate recurrence).
- `extra/qk/mmq_llama_group_chain.py` (chained accumulator across 8 K32 groups; seed src[2] via O(1) DAG re-point).
- `extra/qk/mmq_llama_record_producers.py` (`q4_k_fp16_decode_group_callback` — the decode).
- `extra/qk/mmq_llama_five_buffer_graph.py` / `_full_kernel.py` (3-buffer ABI, per-K32 epoch loop 8× per K256).
- `tinygrad/renderer/isa/amd.py` (`_frag_b128_loads` stride, `_wmma_half_addr` bitcast unwrap).

## Non-negotiables (don't regress)
No dense fp16 weight materialization (§2.4 — decode stays per-tile in-register). Preserve llama Q4_K rounding: f32 intermediate, SINGLE final f16 cast (`d*sc*code−dmin*mn`). New correctness authority signed off before trusting numbers (§2.5). Route stays research/not-promoted, strict fallback to direct-packed, until C8.
