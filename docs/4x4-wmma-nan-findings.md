# 4x4 WMMA NaN — investigation findings & current state (authoritative)

This supersedes `wmma-highvgpr-valu-source-fix-scope.md` (its "high-VGPR VALU-produced source" root cause was DISPROVEN
on hardware -- see below). This is the definitive record of what the 4x4 fault is and is NOT, as of this investigation.

## Symptom
On DEV=AMD:ISA (the from-scratch AMDISARenderer), a rolled-K 64x64x64 fp16 WMMA GEMM produces NaN ONLY at the 4x4
(WM=WN=4, 16-subtile) tile. Smaller tiles (2x4, 4x2 = 8 subtiles) are bit-exact. The default DEV=AMD (HIP/LLVM) path
computes the same 4x4 correctly.

## CURRENT ROOT CAUSE (terminally isolated, HW-confirmed)
The fault was the **post-loop store epilogue reusing high WMMA-loop scratch registers `v201` and `v202`** in the
16-subtile generated stream. Those registers are logically dead, so remu is bit-correct, but the real GPU faults on the
dynamic physical-register role transition from WMMA-loop address/load/pack scratch to epilogue address/data temporaries.

The generated WMMA loop itself is correct:
- replacing only the generated epilogue makes the original generated WMMA body pass on GPU;
- keeping the generated epilogue but remapping `v201` and `v202` to low scratch makes GPU pass;
- freshly reloading/repacking all WMMA fragments before every WMMA still fails until the epilogue temps are moved.

Fix: multi-output WMMA now reclaims the unused low `v1..v7` alignment pad as scalar scratch in `_vpool`, while still
excluding the low accumulator and resident A/B fragment windows. The unmutated generated 4x4 repro now passes on GPU.

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
| WMMA fragment producer chain | FALSE | clean reload/repack before every WMMA still GPU-fails until epilogue temps move |
| generated epilogue logic | FALSE | generated epilogue is remu-correct; moving only `v201/v202` physical temps makes GPU pass |

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
5. Replace one generated slice at a time. The terminal slice was the epilogue: clean epilogue PASS, then generated
   epilogue with only `v201/v202` remapped PASS.

## Status
Resolved in the generated path. The I0 faithful harness passes remu and GPU with no env flags, and
`test/unit/test_amd_isa_wmma.py` passes.
</content>
