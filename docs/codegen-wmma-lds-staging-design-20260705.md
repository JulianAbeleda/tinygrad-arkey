# Design: retire build_gemm_lds2 via codegen (LDS input-staging + schedule search)

Aligned architecture (Fable-5 design-reviewed, source-grounded) for closing the codegen-vs-hand-kernel
WMMA GEMM gap and deleting `extra/qk/prefill/wmma.py::build_gemm_lds2`. Mandate: ZERO hand kernels;
delete only after codegen matches. Status: DESIGN, not yet implemented.

Measured gap (M=512, fp16, synced): hand 52-82 TFLOPS; codegen default 14-34; codegen-best-schedule ~43.
Split: (a) up-to-3x scheduler-recoverable (default omits LOCAL); (b) ~1.5-1.9x fundamental = LDS
input-staging + double-buffer. Peak fp16 ~122 TFLOPS.

## Two corrections that shrink the build
1. **Barriers are AUTOMATIC.** `schedule/rangeify.py:394-429` `bufferize_to_store` emits `buf.after(
   do_store.barrier())` for EVERY `AddrSpace.LOCAL` stage. `gpudims.py` inserts NO barriers (only IF-gating).
   => emitting a LOCAL bufferize gives the cooperative store + workgroup barrier for free. No barrier pass.
2. **Input-staging is ONE postrange change, not opt+late-pass.** `_apply_tc_opt` (`postrange.py:224-317`)
   already has the operand srcs (`srcs[0]/srcs[1]` @289) and axis roles (`in0_ranges/in1_ranges` @239,
   `tc_reduce_axes`). Emit the LOCAL bufferize THERE, right after srcs is built; it rides the existing
   `expander -> pm_add_buffers_local` path. DELETE the separate `stage_input_to_local` late pass from the
   original design.

## TRACK 1 — schedule search (ship first; pure scheduler, no codegen risk)
Grid: UPCAST0 x UPCAST1 in {2,4}, LOCAL in {0,2,4}, UNROLL in {8,16} (=24). `Opt(LOCAL,0,loc)` after TC is
legal (`postrange.py:164-166` needs GLOBAL/LOOP; TC-first satisfied since LOCAL!=index0).
Corrections vs pseudo-code:
- **LDS prune = static, keep it**: `lds_bytes = BK*(local_m+local_n)*tile*itemsize <= ren.shared_max`
  (same shape as `postrange.py:149-151`). Reject before compile.
- **VGPR/spill = NOT static** (tinygrad has no reg model). Compile-then-read `max_vgpr` off the ELF
  (`renderer/amd/elf.py:18-34`); count spill ops (`renderer/isa/__init__.py:40`). Reject high-VGPR/spilling
  AFTER compile.
- **Wedge-safety**: an offline JSON table keyed by CONCRETE shape, populated by a DELIBERATE `PREFILL_V2_SEARCH=1`
  run (NOT implicit at model load) — a pathological schedule that hangs = MES wedge = reboot, can't pkill.
  Steady state reads the table only. Prune static LDS + `DEV=PYTHON` numerics dry-run BEFORE timing on AMD.
- **PADTO/divisibility trap**: `_apply_tc_opt` auto-PADTOs non-TC-divisible dims (`postrange.py:260-263`);
  a tuple tuned on a padded shape can MASK on another. Key cache on concrete shape signature (as
  `_warmstart_match` @344-351); never reuse a tuple across differing divisibility. Reuse the existing
  `try/except KernelOptError -> hand_coded_optimizations` fallback (`postrange.py:369-378`).
Gate: table beats static `_prefill_v2_opts` on every tuned shape; numerics bit-exact vs current codegen.
This also sets the honest baseline (~43) to attribute Track 2 against — NOT the 14-34 default.

## TRACK 2A — LDS input-tile staging (one postrange emission)
Explicit `Opt` (not always-on in `_apply_tc_opt`) so Track-1 can A/B it and because it only pays with
LOCAL>=2 (multiple warps share the tile). Detection is BY CONSTRUCTION, not graph analysis: the operand to
stage is the one invariant across the OTHER operand's LOCAL axis (A invariant across N-LOCAL, B across
M-LOCAL); take srcs + ranges from `_apply_tc_opt` (the `in0_ranges = [u for u in in0.ranges if u not in
in1.ranges]` idiom @239 IS the helper).
Emit: `staged = src0.bufferize(*local_m_axes, *k_tile_axes, arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(
*coop_read_idx)`, feed `staged` into the CONTRACT building WMMA src0 (replace `srcs[0]` @306).
**THE correctness gate = fragment layout.** The LDS re-read index must reproduce WMMA's exact per-lane
fragment element assignment (`permutes_for_shape_str`, @293-306). Derive `coop_read_idx` from the EXISTING
src so the mapping is preserved by construction — do NOT re-derive. Wrong index = silently wrong numerics,
not a compile error.
**Watch**: `pm_remove_bufferize` (`rangeify.py:239,331`) removes bufferizes it deems unprofitable — a
hand-inserted staging bufferize can be silently eliminated. FIRST debug step if LDS "doesn't appear":
confirm the bufferize survives to `bufferize_to_store`.

## TRACK 2B — double-buffer / software-pipeline (last; only if 2A residual justifies)
NOT a PatternMatcher rewrite, NOT imperative peeling (tinygrad has no imperative loop body). STRUCTURAL:
split the K RANGE into an NK-tile loop; add a size-2 buffer axis (index `k&1`) to the bufferize shape
(`flatten_bufferize` @rangeify.py:432 linearizes it); phase-shift (store indexed `(k+1)&1` at tile k+1, WMMA
read `k&1` at tile k). Loop-carried state = the functional index `k&1`; no mutable carry.
**THE trap = WAR hazard.** `bufferize_to_store` gives ONE barrier (RAW, after store). Double-buffer also
needs the WAR edge: prefetch-store into slot s at k+1 must not clobber before consumers of tile k-1 (reading
slot s) finish. Nothing currently expresses this (group-reduce never reuses a slot). MUST add an explicit
`AFTER` edge from next-iter store to prior-iter consume. Missing it = silent data race = FLAKY output.

## Build order + gates
1. Track 1 (days). Gate: beats static, bit-exact.
2. Track 2A single-buffer, ONE operand, ONE shape (LOCAL>=2). Gate: (i) numerics bit-exact (verify
   DEV=PYTHON first — the fragment-layout gate); (ii) kernel shows ds_store/ds_load/s_barrier (bufferize
   survived remove_bufferize). "Flat-or-slightly-worse" is the CORRECT expected result (confirms wiring).
   Then second operand.
3. Track 2B ONLY if 2A-both-operands is still >~1.2x off hand. If 2A already within ~1.1-1.2x -> skip B,
   delete build_gemm_lds2. Gate for B: bit-exact under memory-race stress (flaky = missing WAR edge).
Long pole = B (needs the dependency-edge addition + uncertain payoff; RDNA3 occupancy already hides some
latency). 2A likely captures the larger share (traffic reduction).

## IMPLEMENTATION STATUS (2026-07-05)
- **Track 1 DONE + validated + shipped.** `extra/qk/prefill_v2_schedule_search.py` + frozen
  `prefill_v2_schedule_table.json` (real 14B+8B shapes) + wired into `model.py::_build_prefill_v2_warmstart`.
  Every shape 1.04-2.20x over the default schedule (avg 1.28x); LOCAL (default omitted it) in every winner.
  14B ffn_gate_up 17408x5120: 48.2 vs 26.2 (1.84x). Lifts the codegen BASELINE to ~44 TFLOPS (the honest
  number to measure Track 2 against); does NOT beat the hand kernel alone.
- **Track 2A milestone-1 attempted (scratchpad/localinput_test.py).** WMMA AST confirmed:
  `WMMA(CONTRACT(INDEX_a), CONTRACT(INDEX_b), CONST)`; the SHARED operand = the CONTRACT src whose INDEX
  lacks a LOCAL range (in the probe, src1). Result of naive staging (`idx.bufferize(*shared, LOCAL).index(
  *shared)` on the shared operand): **LDS ops (ds/shared/barrier) DO emit — wiring works, no barrier pass
  needed — BUT rel_rmse=1.37 = WRONG.** This EMPIRICALLY confirms the fragment-layout gate is THE crux: the
  LDS re-read index must reproduce the WMMA per-lane fragment element mapping (`permutes_for_shape_str`);
  `.index(*shared)` drops the fragment-defining axes (GLOBAL/UPCAST/WARP/REDUCE/UNROLL). CORRECT staging
  needs the read index constructed to match the WMMA fragment (cooperative-load + fragment-read, what
  build_gemm_lds2 does by hand) -- a shared INPUT has no natural per-thread partition (unlike the reduce
  partial fix_group_for_reduce stages), so the partition must be IMPOSED. This is the deep remaining work;
  it is multi-session correctness engineering, not a quick rewrite. Both reviews under-specified this
  (assumed bufferize would "just work" like the reduce case).

Key files: `postrange.py` (_apply_tc_opt 224-317, warmstart 340-385), `schedule/rangeify.py`
(bufferize_to_store 394-429 auto-barrier, remove_bufferize 239 silent remover), `codegen/late/expander.py`
(fix_group_for_reduce 132-145 staging template), `renderer/amd/elf.py:18-34` (post-compile max_vgpr),
`codegen/__init__.py:100-108` (pipeline order).

## Second independent design review — deltas (both Fable-5 reviews CONVERGED; these refine)
- **CORRECTION — emit staging in the EXPANDER/add-local phase, NOT in `_apply_tc_opt`.** `_apply_tc_opt`
  (postrange) runs BEFORE the expander (`__init__.py:103` vs `:100-106`); the WMMA + LOCAL axes to bufferize
  don't exist yet there. Emit the STAGE as a sibling to `fix_group_for_reduce` inside `pm_group_for_reduce`
  (`__init__.py:100`) so it flows through `pm_add_buffers_local` (`:106`). (Resolve empirically; the
  ordering argument is strong.)
- **Detection = reuse the exact `gpudims.py:93` predicate**: `missing_locals = [all_ranges[rng] for rng in
  local_dims if all_ranges[rng] not in idx.ranges]`. Operand invariant along LOCAL(0) = N-shared (B) tile;
  along LOCAL(1) = M-shared (A) tile. Deterministic, no heuristic, no postrange tag needed.
- **Three NEW Track-2A' traps:**
  1. **Fragment vec width** — the WMMA input is a vectorized LOAD the expander CONTRACTs; an
     `INDEX`-on-`DEFINE_LOCAL` must PRESERVE that vec width or the devectorizer (`late/devectorizer.py:272-276`
     `no_vectorized_buf`) scalarizes it into per-element `ds_load`s and the win is lost.
  2. **Bank conflicts** — tinygrad has NO LDS-pad knob; allocate the stage buffer at `BN+pad` via a padded
     bufferize shape (nothing computes the pad for you; the hand kernel used PAD=16).
  3. **Single-buffer read→overwrite barrier** — `do_store.barrier()` gives store->read only; single-buffered
     you ALSO need a read->overwrite barrier before the next K-iter clobbers the tile, which current
     machinery does NOT insert. (Double-buffering removes this need — a reason to not linger on single-buffer.)
- **Track 1 refinements**: widen `loc` grid to {0,2,4,8,16} (timid at {0,2,4} for M=512/16-wide tile); drop
  the VGPR prune entirely (not static); spill reject = `prog.scratch_bytes > 0` post-compile. Correctness
  gate to enter the table = bit-identical vs `DEV=PYTHON` + `type_verify`/spec clean, run OFFLINE.
- **Track 2B**: schedule the imperative pipeline pass AFTER `pm_add_buffers_local` (`:106`) but BEFORE
  `pm_reduce` removes the reduce (`:110`) — need the loop still in REDUCE/RANGE form. Encode the
  cross-iteration WAR/RAW explicitly via `AFTER`/`END` edges; the scheduler won't infer it from emit order.
- **Process (memory-critical)**: run the Track-1 sweep + EVERY Track-2 correctness check on `DEV=PYTHON`
  FIRST (proves codegen, no wedge risk); promote to `DEV=AMD` only for timing once output is bit-identical.
  Never let a search iteration timeout/pkill a live AMD run ([[killing-tinygrad-amd-wedges-mes-ring]]).
