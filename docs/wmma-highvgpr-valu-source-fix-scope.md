# Scope: fix the gfx11 high-VGPR VALU-produced WMMA-source fault (b128 direct-load path)

## Root cause (HARDWARE-VALIDATED — supersedes the earlier `s_delay_alu` timing hypothesis)
The 4x4 (and 3x3, 3x4, 4x4) WMMA GEMM NaN on DEV=AMD:ISA is a **deterministic RDNA3 operand-collector fault**, NOT a
timing/scoreboard hazard: a `v_wmma_f32_16x16x16_f16` that reads a **Matrix A/B *source* fragment** which is BOTH
(a) in a **high VGPR (>= ~v120)** AND (b) **produced by a VALU op (`v_pack_b32_f16`)** obtains deterministically wrong
data on gfx1100. See memory [[prefill-4x4-nan-is-hardware-hazard]] (updated).

### Evidence (all on real hardware unless noted)
- **Not timing:** `s_delay_alu instid0(VALU_DEP_1)` before every WMMA (byte-identical to LLVM's guard), `s_delay_alu`
  before EVERY instruction, `s_waitcnt_depctr va_vdst(0)`, v_nop before/after, conservative waitcnt, scheduler-off —
  ALL still all-garbage.
- **Deterministic:** bit-identical finite values + identical NaN mask across runs -> not uninitialized/occupancy/aliasing.
- **Not footprint/occupancy (decisive control):** inflated the PASSING 2x4 to 240 declared VGPRs with dead high-reg
  writes while keeping its WMMA operands <=v119 -> still bit-exact PASS. A 240-VGPR kernel is fine; only WHERE the WMMA
  source operands live matters.
- **Not descriptor / VGPR-count / encoding:** LLVM's working 4x4 declares 200 VGPRs; our granule formula matches; the
  failing WMMA bytes decode correctly; remu executing those exact bytes reads the right regs and passes.
- **Pass/fail tracks the WMMA A/B source register:** <=v119 PASS (<=8 subtiles: 2x4/4x2); >=v120 FAIL (>=9-12 subtiles:
  3x3/3x4/4x4). At >=12 subtiles the accumulators (v8 upward) push a shared A/B fragment tuple to v120+.

### Why LLVM/HIP is immune (HIP-vs-ISA diff)
LLVM reads A/B sources from v129-v197 (ALSO high) yet is correct — because it delivers them with `global_load_b128`
STRAIGHT INTO the WMMA source VGPRs (VMEM-resident, `vmcnt`-gated) and **NEVER runs a VALU (`v_pack`) into a WMMA source
register.** The trigger is the CONJUNCTION high-source AND VALU-produced; LLVM breaks it structurally. The hand kernel
`extra/qk/prefill/wmma.py` (`build_gemm_pipe`) does the same: `global_load_b128` directly into the A/B fragment regs, no
`v_pack`. So this fix = teach the substrate what BOTH working references already do.

### remu note
remu (functional) passes because it models architectural semantics, not the regfile/matrix-operand-collector datapath.
So remu stays a correctness oracle for LOGIC (guards against breaking working cases) but CANNOT see this datapath fault —
only the real GPU gate can. (This is why the earlier remu-correct result was necessary but not sufficient.)

## The fix (two options; A recommended, mirrors BOTH working references)
- **A (recommended): b128 direct-load A/B fragments into the WMMA source VGPRs; eliminate `v_pack` from the WMMA operand
  path.** Load each 8-VGPR A/B fragment via `global_load_b128` (2x b128 = 8 fp16-packed VGPRs) addressed per-lane so the
  fragment gather happens in the LOAD ADDRESS (as `build_gemm_pipe` and LLVM both do), `vmcnt`-gated before the WMMA. No
  VALU ever writes a WMMA source -> breaks the fault conjunction structurally. This is the L3 "b128 direct load" work,
  now proven CORRECTNESS-critical (not just the ~16x perf lever). Removes the current scalar-u16-load + `v_pack` +
  residency staging from the WMMA operand path.
- **B (smaller, fallback): pin all `v_wmma` A/B SOURCE tuples below v120/v128** (only 16 VGPRs needed: 8 A + 8 B) in the
  ISA renderer's fragment allocator; push accumulators/staging HIGH (proven safe — the bloat control showed high writes
  are fine for non-WMMA-source use). Keeps `v_pack` but out of the high-VGPR danger zone. Does NOT get the b128 perf win
  and leaves the underlying VALU->WMMA-source pattern (may resurface if regalloc drifts).

## Definition of COMPLETE
1. General (all tile sizes / 3x3 / 4x4 / real prefill shapes), model-agnostic within gfx11.
2. **HW gate is the ONLY correctness proof for this fault:** DEV=AMD:ISA 64x64x64 + 512x4096x4096 bit-exact + TFLOPS.
   remu bit-exact across shapes is a REGRESSION guard (didn't break logic) but does NOT prove the fault fixed.
3. No duplication: reuse the existing address-gen (`isel_index`/`isel_load`), the b128 load Inst, and the WMMA emit;
   for A, retire `_pack_frag`/`_ab_base` residency on the WMMA-source path (keep only if B is chosen).
4. Existing test/unit/test_amd_isa_*.py stay green.

## Components (option A)
- L1 A/B fragment address model: per-lane b128 base addresses so a `global_load_b128` gathers the fragment (reuse the
  existing per-lane address arithmetic already emitted for the scalar loads — same addresses, wider load).
- L2 emit `global_load_b128` into the 8-VGPR WMMA source tuples (2 loads/fragment); drop the `v_pack` chain.
- L3 `vmcnt` gating: the existing `_insert_waitcnt` already handles vmcnt for global loads -> a WMMA consuming the loaded
  fragment gets its vmcnt drain automatically. Verify the span-aware tracking covers the b128 8-VGPR dest.
- L4 fragment residency: A/B are still reused across a row/col; with b128 loads, residency = load once per row/col and
  keep VMEM-resident (like build_gemm_pipe's F0/F1). Reuse the residency KEYING (id(src0)/id(src1)) but back it with
  loads not packs.

## Risks
- The A/B in-register fp16 layout the WMMA expects must match what `global_load_b128` delivers from the fragment
  addresses. `build_gemm_pipe` + LLVM prove a b128 layout works; replicate their addressing exactly (verify vs remu:
  bit-exact functional result AFTER the change = layout still correct).
- Removing `v_pack` interacts with the multi-tile residency register plan (accumulators can now go high freely since only
  WMMA SOURCES must stay VMEM-loaded, not low — but confirm on HW).
- B's exact boundary (v119/v120) is empirical, not a documented bank line — worth grounding, but A sidesteps it entirely.
</content>
