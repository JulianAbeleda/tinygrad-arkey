# 4x4 WMMA NaN — investigation findings & current state (authoritative)

This supersedes `wmma-highvgpr-valu-source-fix-scope.md` (its "high-VGPR VALU-produced source" root cause was DISPROVEN
on hardware -- see below). This is the definitive record of what the 4x4 fault is and is NOT, as of this investigation.

## Symptom
On DEV=AMD:ISA (the from-scratch AMDISARenderer), a rolled-K 64x64x64 fp16 WMMA GEMM produces NaN ONLY at the 4x4
(WM=WN=4, 16-subtile) tile. Smaller tiles (2x4, 4x2 = 8 subtiles) are bit-exact. The default DEV=AMD (HIP/LLVM) path
computes the same 4x4 correctly.

## CURRENT ROOT CAUSE (by exhaustive elimination, all HW-confirmed)
The fault is the **register allocator's DYNAMIC register-reuse output for the 16-subtile case** -- a value-neutral
(remu-bit-exact) but hardware-faulting register-assignment / linear-order property of our generator's exact ~1155-inst
stream that NO fixed-register hand kernel reproduces. It is NOT a hardware WMMA quirk, NOT any static feature, NOT the
scheduler, NOT waitcnt. The exact reuse/line is not yet isolated -- see `4x4-allocator-terminal-isolation-list.md`.

## What it is NOT (each reproduced in a controlled hand kernel and PASSES on GPU; do NOT re-chase these)
| Hypothesis | Verdict | Evidence |
|---|---|---|
| Disassembler "v74.h" store-cvt-high-half bug | FALSE (disasm bug) | raw bytes decode correct; test/amd/disasm.py mis-renders VOP1 f16 cvt vdst>=128 |
| Timing / s_delay_alu VALU->WMMA hazard | FALSE | s_delay_alu VALU_DEP_1, full-serialize, s_waitcnt_depctr, v_nop, sched-off -- all still NaN; docs: WMMA not in s_delay_alu scopes |
| Logic / codegen bug | FALSE | remu (RDNA3 functional emulator) BIT-EXACT on our exact stream |
| Encoder mangling regs >= 128 | FALSE | raw instruction bytes decode to correct regs |
| Descriptor / VGPR-count / occupancy | FALSE | 240 declared, granule validated; 240-VGPR bloat control of a passing kernel still passes |
| VGPR bank boundary at ~v120/v128 | FALSE | bank = reg mod 4 (v0/v120/v128 all bank 0); bank conflicts STALL, never corrupt (G3 grounding) |
| Operand PROVENANCE (VALU vs VMEM WMMA source) | FALSE | hand 4x4 passes with A/B via v_pack AND via b128; docs: no provenance rule (LLVM VRegSrc_256) |
| Operand HEIGHT (high vs low VGPR) | FALSE | hand 4x4 passes with A/B/C high; LLVM reads A/B v129-197 and passes |
| rolled-K loop + backedge | FALSE | rolled hand 4x4 passes (rolled_hand_4x4.py) |
| 128-accumulator liveness across the backedge | FALSE | rolled hand 4x4 with 128 acc passes |
| u16 scalar-load path | FALSE | rolled hand 4x4 with 128 u16 loads + packs passes (rolled_hand_u16_4x4.py) |
| our-gen's exact layout (acc-low/frags-high) + high load-scratch | FALSE | A1 passes |
| per-element A/B addressing | FALSE | A2 passes |
| high load-dests v220-235 (our-gen's allocator max) | FALSE | A5 passes |
| the list scheduler | FALSE | AMD_ISA_SCHED=0 still NaN |
| missing/mis-tracked waitcnt | FALSE | AMD_ISA_WAITCNT_CONSERVATIVE=1 still NaN |

## Grounding (AMD/LLVM docs)
No documented rule explains the fault (RDNA3 ISA + LLVM are provenance/height/bank-agnostic for WMMA sources). The ONLY
documented WMMA placement rule is overlap-based ("A/B must not overlap D; back-to-back dependent WMMA where D overlaps
next A/B needs a v_nop") -- provably ABSENT in our stream (D in v8-135, A/B in v136-199, disjoint). So the effect is
undocumented/erratum-adjacent BUT confined to our-gen's dynamic stream (a clean hand kernel with all features passes).

## Reusable tools built (all in-repo, GPU-free where noted)
- `extra/qk/prefill/remu_run.py` -- runs any AMDISARenderer kernel through the remu RDNA3 functional emulator vs numpy; a
  GPU-free LOGIC correctness oracle (proved our stream is logically correct; the fault is datapath, invisible to it).
- `extra/qk/prefill/isa_sim.py` -- field-based address/coverage/routing checker for the emitted INS stream.
- `extra/qk/prefill/wmma_faultprobe.py` -- the controlled hand-kernel experiment rig (build_variant: single-buffer 4x4,
  parameterized register layout / provenance / addressing) used for A0-A5. The `/tmp` probe_*.py, rolled_hand_*.py drive it.

## Method that cracked it (reusable pattern)
1. remu (functional emulator) as a LOGIC oracle -> isolates logic from hardware (remu-correct + GPU-wrong => hardware/datapath).
2. The HAND kernel as a CONTROLLED experiment rig -> perturb the PASSING kernel ONE variable at a time; a fixed-register
   replica that reproduces a feature and still PASSES exonerates that feature. Marching from PASS toward our-gen cornered
   the trigger to the one thing a hand kernel can't reproduce: the allocator's dynamic output.
3. Ground every hardware hypothesis vs AMD ISA / LLVM source, never model introspection.
4. Read Inst FIELDS / raw bytes, NEVER disassembler text (disasm has a VOP1-f16-cvt vdst>=128 rendering bug that caused a
   false root cause early on).

## Next (terminal isolation) -- see `4x4-allocator-terminal-isolation-list.md`
I0 faithful NaN-reproducing harness -> L5 (allocator vs isel split) -> offline scans L1-L4 on the captured stream ->
each confirmed by a two-way mutation GPU test (break-it-fixes-gen / add-it-breaks-A5). Then the fix constrains the
allocator/isel to never emit the named pattern in the multi-tile path.
</content>
