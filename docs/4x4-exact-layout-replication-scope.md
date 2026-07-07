# Scope A: reproduce the 4x4 fault in a controlled hand kernel by marching from PASS → our-gen

## Goal
Every INDIVIDUAL feature of our generation's failing 4x4 has been exonerated on GPU (see below). So the trigger is an
EMERGENT property of our generator's exact stream, not any single feature. This scope reproduces the fault in a fully
hand-controlled kernel by starting from the closest PASSING hand analog and adding our-gen's remaining differences ONE at
a time. The first perturbation that flips PASS→FAIL is the trigger (isolated in a clean, fixed-register environment). If
even a byte-faithful replica passes, the trigger is purely the allocator/scheduler DYNAMIC output, and we diff the two
actual streams. This is the terminal isolation step before the fix.

## Established (do not re-test)
- FAIL: our generation's rolled-K 4x4 (DEV=AMD:ISA, 64x64x64). remu bit-exact => logically correct => the fault is
  hardware-timing/datapath, invisible to a functional emulator. Still fails with scheduler OFF and conservative waitcnt.
- PASS (all GPU-confirmed, via DEV=AMD:ISA, same _schedule/_insert_waitcnt/_resolve_labels pipeline as our gen):
  - hand 4x4 unrolled-K (b128 OR v_pack A/B) — /tmp/.../probe4x4.py
  - hand 4x4 rolled-K (b128 A/B) — /tmp/.../rolled_hand_4x4.py
  - hand 4x4 rolled-K + u16-scalar loads + packs (OUR gen's exact load path) — /tmp/.../rolled_hand_u16_4x4.py  <-- CLOSEST PASSING BASELINE
- EXONERATED features (each reproduced in hand ASM and PASSES): WMMA hardware; VALU vs VMEM provenance; high vs low VGPR
  operand placement; rolled-K + loop backedge; 128-accumulator liveness across the backedge; u16-scalar load path.
- The documented gfx11 WMMA hazards do NOT fire in our gen (D in v8-135, A/B in v136-199 disjoint; no D/A/B overlap across
  the backedge; load->pack waitcnt clean; VALU(pack)->WMMA min distance 10 instr). Confirmed by the diff agent.

## The EXACT remaining differences (our-gen FAIL vs rolled_hand_u16 PASS)
Both are 4x4, rolled-K, 128 acc, v_pack A/B, u16 loads. They differ ONLY in:
| # | dimension | our-gen (FAIL) | rolled_hand_u16 (PASS) |
|---|---|---|---|
| D1 | acc / frag placement | acc LOW v8-135, A/B frags HIGH v136-199 | acc HIGH v110-237, A/B frags LOW v10-73 |
| D2 | u16 load-dest / addr-temp placement | HIGH (vpool v200-235) | LOW (v74-89) |
| D3 | A/B address computation | 128 PER-ELEMENT addresses (explicit v_add/v_lshl/v_mul chains, offset=0) | per-FRAGMENT address (8 regs) + per-element load OFFSET immediates (0,2,..,30) |
| D4 | register assignment + instr order | dynamic (linear-scan allocator + list scheduler) | fixed bases, fixed order (then same _schedule/_insert_waitcnt) |
NOTE: D1 layout is likely NOT the trigger — our gen FAILS in BOTH acc-low/frag-high (original) AND acc-high/frag-low
(the reverted option-B), while rolled_hand_u16 PASSES with acc-high/frag-low. So the fault is our-gen-vs-hand, not layout.
=> Prime suspects are D2 (HIGH load-dest/addr-temps) and D3 (per-element address computation) and D4 (dynamic alloc/sched).

## Experiment march (from the PASSING rolled_hand_u16 baseline toward our-gen; first FAIL = trigger)
Each variant = the passing rolled_hand_u16 + exactly ONE added our-gen difference. remu-validate BIT-EXACT per variant
(math-neutral) then GPU-gate (DEV=AMD:ISA). Build in the hand rig (extend build_variant/_build_single_valu_frag).
| V | added difference | tests | register engineering needed |
|---|---|---|---|
| A0 | none (baseline) | control (must PASS) | — |
| A1 | D2: move u16 load-dest + addr-temp scratch HIGH (v200-235) | high load-scratch | acc must vacate the high window -> acc LOW v8-135 (=> also brings D1); frags stay low or move -- keep VA clear of acc |
| A2 | D3: per-element A/B addressing (128 explicit addresses via v_add/v_lshl, offset=0), scratch LOW | per-element address compute | need ~128 addr temps; reuse a small addr-scratch pool (mirror the allocator's cycling) |
| A3 | D2+D3 together (high scratch AND per-element addressing) | the conjunction | acc LOW; addr+load-dest HIGH v200-235 -- the maximal our-gen replica short of the allocator |
| A4 | D1 alone: flip to acc LOW v8-135 / frags HIGH v136-199 (our-gen layout), rest = passing baseline | layout only (expected PASS, confirms D1 innocent) | VA must move out of [8,136): put VA high or in a gap |
FIRST FAIL among A1-A3 isolates D2, D3, or their conjunction. If ALL of A1-A4 PASS => the trigger is D4 (dynamic
alloc/sched) -> go to the D4 fallback.

## Register-engineering constraints (every variant)
- 4x4 needs 128 acc (16*8) + 64 A/B frag (8*8) = 192 pinned + address/scratch + VA + epilogue scratch (v4-9) + PSCR.
- The v>=238 raw-INS garbage trap: keep every used reg < 238 (build_variant asserts this).
- No overlap between acc / frags / VA / u16-scratch / addr-temps / epilogue scratch (build_variant already asserts).
- VA (per-fragment base addresses, TM+TN=8 regs) must not collide with acc when acc is LOW [8,136): park VA in a gap
  ([74,100)) or high (but < 238, and not colliding load-dests). This is the main engineering friction for A1/A3.

## D4 fallback (if A1-A4 all PASS -- trigger is dynamic alloc/sched output)
Capture our-gen's FINAL stream (gen4x4_final.txt, 1165 insts, already captured) and the passing replica's final stream.
Diff at the INSTRUCTION level (not feature level): (a) per-instruction operand REGISTERS around the loop -- find any reg
the allocator reuses across a dependency that the fixed-register hand kernel never does; (b) the exact SCHEDULED order --
any instruction pair our-gen places adjacent that the hand kernel separates (a datapath adjacency hazard). Since remu is
bit-exact on our-gen, it is NOT a logic/aliasing error -> look specifically for a HARDWARE adjacency/reuse pattern remu
cannot model. Candidate: an address/VALU instruction our-gen schedules immediately before a global_load whose result the
WMMA path consumes, at a spacing the hand kernel never produces.

## Validation (per variant, mandatory)
1. remu BIT-EXACT (final_bytes -> run_asm vs numpy A@Bt.T, 0 NaN, rmse<5e-2). NOT ready for GPU otherwise.
2. GPU gate: DEV=AMD:ISA <variant> --gpu. PASS = 0 NaN, rmse<5e-2. Run SERIALLY (single GPU, MES-wedge: never pkill).
3. A variant only isolates a variable if it is byte-identical to the baseline EXCEPT the one added difference (control
   discipline; state explicitly what changed).

## Infrastructure
- Extend build_variant / the E5 per-fragment helper in extra/qk/prefill/wmma_faultprobe.py, or add builders to the
  /tmp probe files. Reuse: rolled_hand_u16_4x4.py (u16 load path), rolled_hand_4x4.py (rolled loop + label markers),
  final_bytes()/remu_validate() (remu), gpu_run/custom_kernel (DEV=AMD:ISA GPU launch, grid (1,1,1), 32 lanes).
- Keep the canonical extra/qk/prefill/wmma.py UNTOUCHED; all variants in /tmp or the faultprobe module.

## Decision tree / success criteria
- A1 or A2 or A3 FAILS -> that difference (D2 / D3 / conjunction) is the trigger. Then: the FIX makes our generation avoid
  it (e.g. if D3 per-element addressing: emit per-fragment b128-style addressing; if D2 high scratch: pin load-dests low).
- All A1-A4 PASS -> D4: the trigger is the allocator/scheduler's dynamic output; the fix is a targeted alloc/sched
  constraint found by the instruction-level diff.
- COMPLETE when: one perturbation flips PASS->FAIL (or the D4 diff names the specific reg-reuse/adjacency), remu stays
  bit-exact throughout, and the identified trigger is confirmed by a matching fix that makes our-gen 4x4 PASS on GPU.

## Risks
- Register-engineering friction (A1/A3 need acc-low + high-scratch + VA placement without overlap; tight but feasible < 238).
- The trigger may be D4 (dynamic-only) -> not reproducible by ANY fixed-register hand variant; then the instruction diff is
  the only route (already have gen4x4_final.txt). Budget for this.
- Single GPU serial; each variant is one supervised DEV=AMD:ISA run.
- remu cannot confirm the fix (functional) -> only the GPU gate proves it; keep remu as the math-neutrality guard.
</content>
