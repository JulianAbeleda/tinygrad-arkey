# BUILD PROGRESS / CONTINUATION: generated fp16-dequant Q4_K primitive (AMD)

Living doc — resume the implementation from here if context is lost. Plan: [`amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md`](amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md). Decision/context: handoff §1.16 (AMD primitive = fp16-dequant-in-register), §1.15 (occupancy routing). Last updated 2026-07-20.

## One-line state
Converting the MMQ generator int8→fp16-dequant IN PLACE (per §1.16; int8 source preserved in git history for later NVIDIA recovery = task #15). Kernel now RENDERS correct-structure fp16 (16 KB LDS, f16 WMMA, 3-buffer ABI); the register-pressure spill is SOLVED; currently closing the last compile blockers. **Numerical correctness NOT yet measured** (that's phase 3/4).

## Commits on master (in order)
- `4eef43945` [wip] convert generator int8→fp16-dequant (renders, not spill-free) — phases 1-2.
- `f6c5c4f6a` [wip] chain accumulator across K32 groups (fixed 512→64 VGPR accumulator blowup; 64 chain heads → 8, matching hand kernel).
- `27e1f0f25` [amd] generalize `_frag_b128_loads` lane stride by dtype itemsize (keeper; superseded by the ratio fix in 65c16271b).
- `65c16271b` [wip] **fix fp16 fragment DS_LOAD vectorization — REGISTER WALL DOWN.** `_fragment_at` load raw bytes (`uint8.vec(esz)`) then `.bitcast(half)` the VALUE (was `UOp.index(dtype=half)` mislabeling a uint8 ptr → legalizer shattered into 16 uchar scalar loads); `_wmma_half_addr` unwraps the BITCAST; `_frag_b128_loads` stride = loaded/pointer itemsize RATIO. Result: scalar DS_LOAD 1950→0 (28 vectorized DS_LOAD_B128), peak vregs 491→63, compile 6.5min→75s. `test_amd_isa_wmma.py` 4 failures are PRE-EXISTING (verified on baseline 51cce914c), not regressions.

## CURRENT BLOCKER (after DS_LOAD fix) — being fixed via real barriers
Full kernel still `emitted=False`. Real target (the "SGPR/PARAM(0)" label was cosmetic — `Register.index` is always 0): an **intra-chain `DS_LOAD_B128` fixed-lease (`v200`) conflict** — 7 reloads within one subtile element's 7 non-head groups are hoisted together by the greedy `pressure_schedule`/`_pressure_schedule_block` (`tinygrad/codegen/late/regalloc.py`) because there is **no real `Ops.BARRIER` between K32 groups** to split the block (only `.after()` pseudo-ops).
- **FIX IN PROGRESS (graph-side, preferred over core-scheduler change): emit real `Ops.BARRIER` between the 8 K32 groups** (hand-kernel cadence, wmma.py:600-631). This splits the block for the scheduler AND — **likely also a real correctness bug** — provides the cross-wave workgroup barrier the single-buffered LDS (DBUF=0, 8 waves sharing LDS) requires; `.after()` only orders within one wave, so without it waves race on shared LDS → wrong runtime results. Agent verifying whether any real barrier exists today.
- Fallback if barriers insufficient: minimal `_pressure_schedule_block` change (split at `.after()`-chained fixed-lease boundaries) — core-scheduler, blast radius, needs review.

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
