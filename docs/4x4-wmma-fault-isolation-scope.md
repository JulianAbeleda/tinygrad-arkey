# Scope: isolate the 4x4 WMMA fault — online grounding + controlled hand-ASM experiments

## Why this, why now
We have flipped the 4x4 root cause TWICE on hardware (disasm-cvt hypothesis, then s_delay_alu timing), and option B (swap
A/B low) also failed. We do NOT have the full picture. Before building the real fix (option A: b128 direct-load A/B), we
must pin the EXACT invariant, or we risk building the wrong rule again.

Method (two independent lines; a claim only stands if BOTH agree):
1. **Online grounding (PRIMARY / trusted):** AMD RDNA3 ISA, LLVM AMDGPU backend, GPUOpen, errata. We trust documented
   hardware contracts over our own HW observations, which have proven partial/misleading.
2. **Controlled hand-ASM experiments (CONFIRMATION):** perturb the KNOWN-GOOD `extra/qk/prefill/wmma.py` `build_gemm_pipe`
   (a passing 4x4) ONE variable at a time. Because the hand kernel is fully hand-controlled (every register + instruction),
   each experiment isolates a single factor — unlike our generation experiments, which moved many things at once.

The hand kernel is the ideal rig: it PASSES on the GPU, so any single perturbation that breaks it isolates the trigger.

## All theories on the table (status as of now)
| # | Theory | Claim | Status | Evidence |
|---|---|---|---|---|
| T1 | Timing/scoreboard hazard | VALU->WMMA latency needs s_delay_alu | REFUTED (HW) | 5 mitigations (s_delay_alu VALU_DEP_1, full-serialize, depctr, nop, waitcnt, sched-off) all failed. RE-GROUND vs docs. |
| T2 | Logic/codegen bug | emitted stream is logically wrong | REFUTED | remu functional emu bit-exact across shapes |
| T3 | Encoding bug | regs>=128 mis-encoded | REFUTED | raw instruction bytes decode to correct regs |
| T4 | Descriptor/VGPR-count | under-declared VGPRs -> aliasing | REFUTED | 240 declared, granule formula validated by 2x4 |
| T5 | Occupancy/footprint | too many VGPRs -> corruption | REFUTED (HW) | bloat control: 2x4 inflated to 240 VGPRs, operands low -> still PASS |
| T6 | Operand PROVENANCE | a VALU-produced (v_pack) WMMA source faults | LEADING | LLVM/hand load A/B via b128 (VMEM), never v_pack; ours v_pack -> fail. Needs isolation from height. |
| T7 | HIGH-VGPR alone | any WMMA source >= boundary faults, regardless of provenance | LIKELY REFUTED | LLVM reads A/B from v129-197 (high) via VMEM and PASSES. Needs clean test. |
| T8 | CONJUNCTION (A/B) | A/B source HIGH **and** VALU-produced faults | PARTIAL | agent's claim; but option B (A/B low VALU, C high) still failed -> not A/B-specific |
| T9 | ANY source (A/B/C) high+VALU | C (accumulator) counts too, not just A/B | UNTESTED cleanly | option B put C high+VALU-init -> failed; but C move was coupled with other changes |
| T10 | Total VALU-source footprint | when VALU-produced WMMA sources don't fit low, some spill high -> fault | FITS ALL DATA | 2x4=112 fits/pass, 3x3=120/4x4=192 don't/fail; untested in isolation |
| T11 | Bank boundary | the ~v120/v128 line is a VGPR-bank/operand-collector boundary | UNTESTED | ground vs docs; boundary sweep on HW |
| T12 | Documented WMMA operand constraint / erratum | RDNA3 has a published rule on WMMA operand delivery/registers | NOT SEARCHED | primary grounding target |
| T13 | fp16/opsel interaction | v_pack fp16 + WMMA opsel at high regs specifically | SPECULATIVE | low prior; the disasm bit-7 was decoder-only |

## Part 1 — Online grounding tasks (do first)
- **G1** RDNA3 ISA reference: the `v_wmma_f32_16x16x16_f16` section — any stated constraint on source-operand registers,
  banks, or how operands must be delivered (T11, T12).
- **G2** LLVM AMDGPU: WHY does it deliver A/B by load, not VALU? Search `SIRegisterInfo`/`SIISelLowering`/WMMA lowering +
  `GCNHazardRecognizer` for comments about WMMA operand registers / VALU-into-WMMA-source (T6, T12). A code COMMENT is the
  strongest grounding.
- **G3** RDNA3 VGPR bank layout / operand-collector: is there a documented bank boundary near v120/v128 that a 24-VGPR
  WMMA operand read (8A+8B+8C) would straddle (T11)?
- **G4** Known errata / forum reports: "gfx1100 v_wmma wrong result VALU source", RGP/rocprof notes (T6, T12).
- **G5** Re-ground T1 (timing) vs docs: does anything say WMMA source latency is software-managed on gfx11 (to reconcile
  with the HW result that s_delay_alu didn't help)?
Each G-task must state: which theory it confirms/refutes and the exact quoted source.

## Part 2 — Controlled hand-ASM experiments (from the PASSING build_gemm_pipe baseline)
Each experiment = ONE perturbation of the working 4x4 hand kernel; run on GPU (bit-exact vs numpy) AND through remu
(remu should ALWAYS stay bit-exact -> confirms the perturbation didn't change the math, only the delivery/placement).
Factors: operand ROLE {A=src0, B=src1, C=src2}, register HEIGHT {low<120, high>=128}, PROVENANCE {VMEM=b128, VALU=v_pack/v_mov}.

| E | Perturbation of build_gemm_pipe | Isolates | Predict (if T6/T8/T9 true) |
|---|---|---|---|
| E0 | none (baseline) | — | PASS (known) |
| E1 | move A/B b128 loads to HIGH dest regs (>=128), keep b128 | HIGH + VMEM | PASS (LLVM-consistent) -> refutes T7 |
| E2 | replace A/B b128 with scalar-load + v_pack into the SAME LOW regs | VALU + LOW | PASS -> provenance alone (low) is safe |
| E3 | replace A/B b128 with v_pack into HIGH regs (>=128) | VALU + HIGH | FAIL -> confirms T6/T8 (VALU+high faults) |
| E4 | move C accumulator regs HIGH (v_mov-init stays) | C, VALU-init + HIGH | FAIL => T9 (C counts); PASS => A/B-specific |
| E5 | boundary sweep: v_pack one operand at v112/v116/v120/v124/v128 | exact boundary | first FAIL locates the line (T11) |
| E6 | footprint control: dead high v_mov writes, all operands low+b128 | footprint | PASS -> re-confirms T5 refuted, in the hand rig |
| E7 | (if E3 FAIL) same v_pack-high but add s_delay_alu/depctr guards | timing vs datapath | still FAIL -> re-confirms T1 refuted (datapath, not timing) |

## Decision matrix (which pass/fail pattern => which invariant)
- E1 PASS + E3 FAIL + E2 PASS  => invariant = "a VALU-produced WMMA source in a HIGH reg faults" (T6∧T8). Fix = keep VALU-
  produced sources low OR make them VMEM (option A for A/B; C already low).
- E4 FAIL as well               => C counts too (T9): ALL VALU-produced sources (incl. C) must be low-or-VMEM.
- E1 FAIL                        => height alone matters even for VMEM (T7) — option A insufficient; rethink.
- E5                            => the exact boundary -> the AB_SRC_MAX / accumulator-placement constant.
- E6 PASS                       => footprint confirmed irrelevant (control).

## Infrastructure
- Runner: import `build_gemm_pipe` (+ perturbed variants) from `extra/qk/prefill/wmma.py`; run on DEV=AMD (it self-
  assembles + runs via run_linear) and capture rmse vs numpy. Add a `WMMA_EXPERIMENT` env or a small variant harness
  `extra/qk/prefill/wmma_faultprobe.py` that builds each E-variant (do NOT edit the canonical kernel in place).
- remu cross-check: feed the same assembled bytes to `extra/qk/prefill/remu_run.py`'s run_asm path (remu must stay
  bit-exact for every E -> proves the perturbation is math-neutral).
- GPU safety: hand-kernel variants are near-correct (won't hang); still, one run at a time, no pkill on a live DEV=AMD.

## Success criteria
The G-tasks + E-experiments AGREE on a single invariant (documented rule confirmed by a controlled experiment). That
invariant becomes the exact spec option A must satisfy. Only then implement option A (b128 A/B), validated remu-offline +
one GPU gate. If docs and experiments DISAGREE, that conflict is the finding — investigate before building anything.

## Explicitly OUT of scope here
Building option A. This scope only IDENTIFIES the invariant. Implementation is the follow-on (docs/wmma-highvgpr-valu-
source-fix-scope.md), rewritten once the invariant is pinned.
</content>
