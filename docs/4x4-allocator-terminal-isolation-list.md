# Terminal isolation list: the 4x4 register-allocator dynamic-output fault

## What the constraints leave (read this first)
The trigger must simultaneously satisfy ALL of:
- **remu bit-exact** on our-gen's exact stream => it does NOT change any logical value. Rules out every value-corrupting
  register aliasing/overlap (if a virtual overwrote a live WMMA operand or fragment, remu would be wrong -- it isn't).
- **`AMD_ISA_SCHED=0` still fails** => NOT the list scheduler; the fault is in the LINEARIZED (pre-schedule) allocated stream.
- **`AMD_ISA_WAITCNT_CONSERVATIVE=1` still fails** => NOT a missing/mis-tracked waitcnt (a full drain after every memory op
  does not fix it).
- **A0-A5 (fixed-register hand replicas of every feature + register range) all PASS** => NOT any static feature, layout, or
  VGPR range; it is the DYNAMIC assignment/structure our generator produces that no fixed kernel reproduces.
So the trigger is: a register-assignment / linear-order property of our-gen's exact 4x4 stream that is VALUE-NEUTRAL
(remu-correct) yet HARDWARE-faulting, and absent from any hand kernel. Very narrow.

## Enabling infrastructure (do first -- required for the mutation-based items)
- **I0. Faithful harness reproduction.** Capture our-gen's exact post-regalloc INS stream and run it via custom_kernel so
  it reproduces the SAME failure as the normal DEV=AMD:ISA path (NaN, not a spurious rmse from wrong kernarg order/layout).
  The prior attempt gave rmse=7.77/0-NaN -- almost certainly wrong kernarg order (our-gen kernarg = [out, A, B]; the probe
  harness uses a transposed-B convention). NAIL our-gen's exact arg order + buffer layout so gen-exact reproduces NaN. This
  is the mutation substrate for I-items below; without it, mutation results are meaningless.

## Candidate mechanisms (each: hypothesis -> test -> what confirms)
Prioritized by fit to the constraints (remu-correct + not-sched + not-waitcnt).

- **L1. WMMA operand register touched by a recent non-WMMA write in the LINEAR order (value-neutral).** The allocator/isel
  emits a mov/load/pack whose DEST equals (or 8-reg-overlaps) a WMMA A/B/C operand, where the written value happens to
  equal what's already there (value-neutral, so remu-correct) but the hardware operand-collector still mis-gathers.
  A2/A3 tested VALU->WMMA-source but with a CLEAN separation; the allocator may create a TIGHTER or DIFFERENT-shaped one.
  Test (offline, our-gen stream): for every v_wmma, scan the preceding N instructions for any write whose dest reg is in
  the wmma's src0/src1/src2 8-reg spans; compare the pattern to A5 (which has none). Confirm: a write-into-operand pattern
  present in our-gen, absent in A5.

- **L2. Cross-backedge register reuse (value-neutral).** A virtual live across the loop backedge shares a physical with a
  loop-body virtual; correct in values (the shared value is re-established each iter) but the hardware carries stale
  micro-arch state across the taken backedge for that reg. Test (offline): liveness across the backedge in our-gen vs A5 --
  any physical reg that is BOTH written in the body AND read after the backedge as a different virtual. Confirm: such a reg
  in our-gen, none in A5. (Mutation: pin that virtual to a private reg -> does gen pass.)

- **L3. Same-register read-as-address and write-as-load-dest in a tight window.** The allocator reuses a load-DEST reg as
  an ADDRESS reg for a nearby load (value-correct if ordered), a documented-ish RMW-ish datapath pattern. Test (offline):
  find any reg used as `global_load_u16.addr` and as another `global_load_u16.vdst` within a small window. Confirm:
  present in our-gen, absent in A5.

- **L4. Instruction-adjacency unique to our-gen (linear order).** Since sched-off fails, the LINEARIZER order matters. A
  specific instruction PAIR our-gen places adjacent (that A5 separates) is a hardware hazard. Test (offline): tabulate
  adjacent-instruction-type pairs around the WMMAs/loads/packs in our-gen vs A5; find a pair present in gen, absent in A5.
  Confirm on GPU by inserting a v_nop/independent op to break that adjacency in gen (mutation via I0).

- **L5. isel STRUCTURE difference, not the allocator.** our-gen = 1155 insts, A5 = 1254 (our-gen is MORE compact). The
  ~100-inst delta could carry the trigger (a fused/compact isel pattern A5's hand structure lacks). Test: run our-gen's
  isel output but with FIXED (hand) register assignment (bypass/override the allocator). If it PASSES -> the ALLOCATOR is
  the cause (L1-L3); if it FAILS -> the ISEL structure is (bisect the delta). This cleanly splits allocator vs isel.

- **L6. A specific physical register the allocator uses in a WMMA-adjacent context.** e.g. a virtual placed at a reg that
  the hardware reserves/mishandles when a WMMA is in flight (not a range -- a specific reg + context). Test (offline):
  the full per-instruction reg map of our-gen; any reg used both as a WMMA-window neighbor and a live virtual. Lower prior
  (ranges exonerated) but cheap to scan.

- **L7. Number of DISTINCT physical regs live at the WMMA cluster (occupancy of the operand collector), value-neutral.**
  our-gen keeps ~34 load-dest regs live where A5 reuses ~16. If the WMMA operand collector has a live-register limit that
  corrupts (not stalls) when exceeded. Test: reduce our-gen's live-virtual count (force tighter vpool reuse) -> pass?
  Weak (G3 said collector oversubscription stalls, not corrupts) but cheap.

## Method
1. I0 faithful harness (NaN-reproducing gen-exact) -- gate on it reproducing the fault before any mutation.
2. L5 FIRST (allocator vs isel split) -- one experiment that halves the space.
3. Then the offline scans L1/L2/L3/L4 on our-gen's captured stream (no GPU) to find a pattern present in gen + absent in
   A5; each candidate confirmed by ONE mutation GPU run (break the pattern -> pass).
4. L6/L7 only if L1-L5 are clean.
COMPLETE when: a single pattern is present-in-gen / absent-in-A5, and breaking it (mutation) makes gen PASS on GPU, and
re-introducing it makes A5 FAIL. That two-way test names the exact trigger -> then the fix constrains the allocator/isel
to never emit it.

## Notes / risks
- remu can only guard math-neutrality of mutations; only the GPU proves a mutation fixed/broke the fault.
- All offline scans read Inst FIELDS/raw bytes, never disasm text (disasm has the VOP1-f16-cvt vdst>=128 bug).
- Single GPU serial; every mutation is one supervised DEV=AMD:ISA run; never pkill a live run.
</content>
