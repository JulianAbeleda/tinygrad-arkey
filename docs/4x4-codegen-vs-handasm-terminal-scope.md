# 4x4 WMMA codegen-vs-handasm terminal scope

## Goal
Make generated AMD:ISA 4x4 WMMA behave like the known-good handwritten/fixed-register ASM path, without adding a new
permanent handwritten kernel. Hand ASM is only the control/comparison corpus. The fix must land in codegen/isel/regalloc.

## Current hard facts
- Generated 64x64x64 WM=WN=4 stream is captured by `extra/qk/prefill/gen4x4_i0_harness.py`.
- I0 generated stream is exactly the expected shape: 1155 `Ops.INS`, 16 `v_wmma`, 64 `v_pack`, 128 `global_load_u16`,
  128 `global_store_b16`.
- I0 remu passes: no NaNs, rmse about 0.00156.
- I0 GPU now passes after the allocator-pool fix: no NaNs, rmse about 0.00156.
- `AMD_ISA_L5_FIXED_RA=1` changes the GPU failure from NaNs to wrong finite values while remu still passes.
- Hand/fixed-register 4x4 variants pass. Therefore this is a codegen-vs-handasm correctness bug, not a general 4x4 WMMA
  impossibility and not a permanent hand-kernel gap.

## Final root cause and fix
The generated WMMA loop itself was correct. The hardware failure came from the generated post-loop store epilogue reusing
high VGPRs `v201` and `v202` as address/data temporaries immediately after those registers had been used in the WMMA
loop's high address/load/pack scratch region. Remu is value-correct because the logical values are valid; the real GPU
faults on this dynamic physical-register role transition.

Minimal proof:
- Replacing only the generated epilogue with a clean epilogue keeps remu passing and makes GPU pass.
- Keeping the generated epilogue but remapping only `v201` and `v202` to low scratch keeps remu passing and makes GPU pass.
- Remapping only `v202` removes NaNs but leaves large finite error; remapping `v200`/`v201` without `v202` does not fix it.
- Freshly reloading/repacking all WMMA fragments before every WMMA still GPU-fails until the epilogue temps are moved.

The production fix is in `_vpool`: multi-output WMMA still reserves the low accumulator and resident A/B fragment windows,
but now reclaims the unused `v1..v7` alignment pad as scalar scratch. This lets short epilogue temps allocate below the
WMMA fragment windows instead of reusing high `v200+` scratch.

## Results from the parallel pass
- **S1 scanner/comparator is in place** at `extra/qk/prefill/gen4x4_stream_scan.py`. It compares generated I0 against the
  remu-valid hand/faultprobe b128 stream using instruction fields/raw bytes, not disassembly parsing.
- **Generated final stream:** 1163 final scheduled instructions, 16 WMMA, 64 pack, 128 scalar `global_load_u16`.
- **Hand/faultprobe b128 stream:** 808 final scheduled instructions, 64 WMMA, 0 pack, 64 `global_load_b128`.
- **L3 direct load address/dest overlap:** generated has 80 direct overlaps and 100 nearby reuses within 8 instructions;
  hand b128 has 0 direct and 0 nearby.
- **S2 mutation result:** `AMD_ISA_NO_LOAD_ADDR_DST_REUSE=1` removes direct addr==dest scalar-load reuse but GPU failure is
  unchanged (`nan=0.0342`, `rmse=10520.69336`). Direct address/dest reuse is therefore exonerated as the primary trigger.
- **S3 mutation result:** `AMD_ISA_WMMA_PACK_GAP=1/4/16/64` keeps remu passing but GPU failure is unchanged. Simple
  pack-to-WMMA spacing is therefore exonerated as the primary trigger.
- **S4 feasibility result:** generated A fragments are b128-contiguous in the UOp graph, but generated B fragments are
  strided under the normal `B[K,N]` matmul layout. A-only b128 lowering is not a complete generated equivalent to the
  hand b128 control, and a quick env-gated prototype hit the backend no-spill boundary before producing a runnable GPU
  diagnostic. Treat b128 conversion as a design track, not a terminal isolation result yet.
- **L2 backedge result:** the generated stream does have a codegen-only role transition where tail scalar-load destination
  registers `v202..v217` are reused after the taken backedge as loop-top address temporaries. The default-off scrub
  `AMD_ISA_L2_BACKEDGE_SCRUB_LOAD_SCRATCH=1` inserts dead writes before the backedge, remu-passes, and leaves GPU failure
  unchanged. This cross-backedge role transition is therefore exonerated as the primary trigger.
- **L4 stale-pack result:** generated streams have stale packed operands, e.g. `v152:159` built by eight `v_pack_b32_f16`
  producers and consumed 250+ instructions later. The scanner exposes this with `--wmma-producer-spans`. The default-off
  refresh `AMD_ISA_WMMA_REFRESH_STALE_PACK=64`, and the stricter `AMD_ISA_WMMA_REFRESH_STALE_PACK=1
  AMD_ISA_WMMA_REFRESH_EVERY=1`, both remu-pass and leave GPU failure unchanged. Stale packed-fragment age is therefore
  exonerated as the primary trigger.
- **Pack-source provenance result:** `AMD_ISA_PACK_SOURCE_SELF_MOVES=1` inserts value-neutral VALU self-moves on both
  `v_pack_b32_f16` sources before every pack. This remu-passes and leaves GPU failure unchanged, so immediate VMEM-load
  provenance into `v_pack` is not the primary trigger.
- **Normal-B hand controls:** a hand rolled-u16 4x4 variant reading normal row-major `B[K,N]` with strided scalar offsets
  remu-passes and GPU-passes. A stricter hand variant that materializes 16 per-element B address VGPRs and overwrites
  them with `global_load_u16(addr==vdst)` also remu-passes and GPU-passes. Therefore normal-B layout, strided scalar B
  delivery, and per-element address-overwrite delivery are not sufficient to reproduce the generated failure.
- **Accumulator placement partial:** a valid hand-u16 variant with low-ish accumulators (`v16..v143`) and high fragments
  remu-passes and GPU-passes. An attempted exact `v8..v135` hand test was invalid because the quick mutation corrupted
  hand epilogue scratch; exact `v8` remains the only unclean placement check.
- **Exact static layout result:** `extra/qk/prefill/genlayout_fixed_u16_4x4.py` implements a clean fixed-register hand
  control with generated-like static layout and WMMA order: accumulators `v8..v135`, fragments `v136..v199`, VA
  `v200..v207`, scratch `v220..v235`, epilogue temps away from `v8`, and normal row-major `B[K,N]`. It remu-passes and
  GPU-passes (`nan=0.0000`, `rmse=0.00156`). Therefore exact static accumulator/fragment placement and generated WMMA
  issue order are exonerated as sufficient causes.
- **Generated-stream role-remap result:** `extra/qk/prefill/gen4x4_stream_role_remap.py` mutates the captured generated
  stream while preserving generated order/math. Splitting scalar load destinations for the `s[8:9]` and `s[10:11]`
  pointer classes into a separate data bank both remu-pass and GPU-fail identically. A pre-WMMA scrub of scratch/address
  regs `v203..v218` and `v220..v235` also remu-passes and GPU-fails identically. These weaken the simple dynamic
  scratch/data sharing theory.
- **Terminal epilogue result:** replacing only the post-loop generated store epilogue passes on GPU. Keeping the generated
  epilogue but remapping `v201` and `v202` to low scratch also passes. This identifies the primary unsafe dynamic pattern:
  post-loop epilogue address/data temporaries reusing the high WMMA-loop scratch band.
- **Allocator fix result:** reclaiming `v1..v7` as scalar scratch for multi-output WMMA makes the unmutated generated I0
  pass on GPU with no env flags. `test/unit/test_amd_isa_wmma.py` passes with the updated pool invariant.

## Key hand-vs-codegen delta
The checked-in canonical hand `build_gemm_pipe(64,64,64,4,4)` cannot build 4x4 because double buffering needs 274 VGPRs.
The useful hand controls are the single-buffer/faultprobe lineage and the original hand builder patterns.

Hand-style passing pattern:
- direct `global_load_b128` into A/B fragment registers
- stable address registers separate from fragment data registers
- `s_waitcnt`, then `v_wmma`
- no scalar u16 temp-load + pack chain in the b128 variant

Generated failing pattern:
- per-element `global_load_u16`
- allocator often assigns the same VGPR as load address and load destination, e.g. `global_load_u16(v203, v203, ...)`
- `v_pack_b32_f16` builds A/B resident fragments from those temp load regs
- `v_wmma` consumes the packed fragments

## Primary theories
Ordered by current evidence:
1. **Resolved: post-loop epilogue temp reuse of high WMMA scratch.** The exact required move is `v201` and `v202` out of
   the high scratch band; the allocator-pool fix gives the epilogue low scalar scratch naturally.
2. **No remaining primary theory for the 4x4 NaN.** The original repro passes on hardware.

Exonerated as primary triggers:
- direct scalar `global_load_u16` address==destination reuse
- simple pack-to-WMMA instruction spacing/latency
- cross-backedge tail-load-dest to loop-top-address role transition
- stale packed-fragment age
- immediate VMEM-load provenance into `v_pack`
- normal row-major B layout with strided scalar loads
- per-element B address-register overwrite with `global_load_u16(addr==vdst)`
- exact static generated accumulator/fragment placement (`ACC v8..v135`, fragments `v136..v199`)
- generated WMMA issue order, when reproduced in a clean fixed-register hand stream
- per-pointer-class scalar load data remapping into a separate bank
- pre-WMMA scratch/address register scrubbing
- scheduler-off alone
- conservative waitcnt alone
- high VGPR/VALU provenance alone

## Required proof bar
A candidate is not complete until it passes all of:
1. Present in generated failing stream.
2. Absent, or materially different, in a passing hand/fixed-register stream.
3. Breaking/removing it in generated code keeps remu passing and makes GPU pass or materially moves toward pass.
4. Reintroducing the pattern into a passing hand/fixed-register stream makes GPU fail, if the mutation is feasible without
   changing math.

## Workstreams

### S1. Stream scanner and comparator
Build a reusable field-based scanner. It must read `Inst` fields/raw bytes, not disassembly text.

Inputs:
- generated final/pre-final stream from I0
- hand/faultprobe stream
- optional regalloc metadata for L2/L7

Scans:
- L3: `global_load*` where destination span overlaps address span, plus nearby load addr/dest reuse.
- L1: recent non-WMMA writes whose dest overlaps a WMMA A/B/C operand span.
- L4: adjacent instruction-type/register-role pairs unique to generated stream near load/pack/WMMA.
- L6/L7: physical register context and live pressure near WMMA.

Output:
- compact ranked findings with instruction indices, raw instruction strings for orientation, spans, and distances.

### S2. L3 mutation: no load address/dest reuse
Add an env-gated codegen/regalloc rule that prevents a `GLOBAL_LOAD` destination from being allocated to the same physical
VGPR as its address source. This must be general enough to test generated code and default-off.

Gate:
- default path unchanged
- `AMD_ISA_NO_LOAD_ADDR_DST_REUSE=1` still remu-passes I0
- GPU result recorded for I0

### S3. L1/L4 mutation: pack-to-WMMA spacing
Add an env-gated renderer mutation that inserts neutral separation after A/B pack clusters or before WMMA, without changing
logical values. This tests whether codegen’s compact pack-to-WMMA adjacency is unsafe.

Gate:
- remu passes I0
- GPU result recorded

### S4. B128 generated-fragment feasibility
Do not add a hand kernel. Analyze whether the generated WMMA A/B fragment loads can be lowered as b128-style contiguous
fragment loads for this pattern, matching the hand safe path. This may become the long-term performance/correctness fix,
but it is not the first mutation unless S2/S3 fail or strongly point there.

Gate:
- identify exact codegen/isel layer where scalar u16 loads are introduced
- propose minimal generated-lowering change or explain why current UOp shape cannot express it yet

## Execution order
1. S1 and S2 run in parallel.
2. Run I0 GPU gate for S2 as soon as it lowers and remu passes.
3. If S2 does not pass, run S1 findings to choose S3 or L2/L4 next.
4. Keep S4 as design track unless scanners/mutations show b128 is the cleanest fix.

## Non-goals
- No permanent handwritten 4x4 kernel.
- No broad rewrite of the AMD ISA backend.
- No reopening disproven theories: high VGPR alone, VALU provenance alone, scheduler alone, waitcnt alone, rolled-K alone,
  or 128 accumulators alone.
