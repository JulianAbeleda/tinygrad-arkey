# LEVER A GROUNDED (refines CG-W) — addressing overhead = offsets overflow the 13-bit immediate field

Rendered the actual prefill WMMA matmul kernel (12288x4096x512 fp16, production TC opts) via to_program + DEBUG=4
(`r_192_2_32_4_2_2_2_4_4_256_2`). Inspected the K-loop body on CURRENT code.

## Finding (refines/corrects CG-W)
tinygrad's HIP C renderer ALREADY emits the good pattern: ONE loop-variant base per iteration
`alu130 = alu1 + (Ridx0<<13)` (1 shift + 1 add), then all ~128 operand loads as `*(data2 + (alu130 + CONST))` =
base + constant offset. CG-W's "no base+immediate-offset" was imprecise.

The ACTUAL problem: the CONST offsets are LARGE -- up to ~5632 elements = ~11264 BYTES (the WMMA tile spans a big
address range). **RDNA3 `global_load` has a 13-bit SIGNED byte immediate offset field (~+-4096 bytes).** Offsets
beyond that (most of the tile's loads) CANNOT use immediate addressing -> clang materializes a per-load 64-bit
register address (v_add_co + v_addc, ~2 ops/load) -> ~128 address ops/iter. THIS is CG-W's ~160 ALU/iter vs 16 WMMA.

## So the fix (Lever A, refined)
NOT "emit base+offset" (already done). The fix is **offset-range-aware base grouping**: when a tile's loads share
a base but their constant offsets exceed the +-4KB immediate field, emit a few intermediate base variables
(base_g = alu130 + group_offset, one per ~4KB span / per tile-row) so each load is `*(base_g + small_offset)` with
small_offset within the immediate field -> clang uses immediate addressing -> ~8 base-adds/iter instead of ~128.
This is exactly Tensile's multi-base-pointer + pointer-increment pattern. Lives in the renderer (cstyle.py) or a
codegen pass that factors large-stride load-index terms into hoisted base vars.

## Open question for implementation
The default path is HIP C -> clang. tinygrad emits `data2 + (alu130 + CONST)`; clang chooses the addressing mode.
Need to confirm: does emitting grouped base VARIABLES in the HIP C (so clang sees `base_g + small`) actually make
clang use `offset:` immediate + cut the adds? (Likely yes -- clang folds small constant offsets into the load.)
P0 must verify this empirically (emit a grouped-base variant, compile, count ISA adds + measure TFLOPS).

## Status of "both levers"
- Lever A (this): mechanism grounded + refined. Fix = offset-range-aware base grouping. Renderer/codegen change,
  bounded but real (broad test surface).
- Lever B (pipelining): unchanged from scope -- new prefetch/async UOp + double-buffer + modulo-schedule pass.
  Bigger (new IR primitive + pass). Project-level.
Both are substantial tinygrad-CORE work; this turn grounded A and refined the exact mechanism (offset overflow),
which is the prerequisite for a correct implementation.

## Files
renderer/cstyle.py (HIP C load/index rendering), the index UOp lowering. Kernel dumped:
r_192_2_32_4_2_2_2_4_4_256_2. Prior: wmma-make-expressible-scope-20260619.md, CG-W.

## ⚠ LEVER A REFUTED (empirical, decisive) — clang already does the base grouping
Worktree agent rendered the exact kernel, built a grouped-base variant of the identical source, compiled+
disassembled+timed both (interleaved best-of-50, same process):
- ORIGINAL 41.7 TFLOPS (34.1% peak) vs GROUPED 41.1 (33.7%) = **0.988x** (gate was >=1.2x) -> FAIL.
- Outputs **bitwise identical** (relerr 0).
- **Disassembly identical except register renumbering.** clang ALREADY materializes several base-address VGPR
  pairs and emits every load with the `offset:` immediate field (incl. negative `offset:-4096` to cover two 4KB
  windows per base) -> hoisting bases in C source changes nothing; instruction selection already does it.
- ISA loop: `v_add_co_u32`=**61** (NOT ~128/160), global_load=544, v_wmma=64. **CG-W's "~160 address-ALU/iter,
  no base+offset" was a MISREAD of the ISA** -- the addressing is already optimal.

CONCLUSION: Lever A (renderer addressing/base-grouping) is REFUTED. The ~34% peak is bound by load throughput /
WMMA scheduling / occupancy / single-wave latency-hiding (the POWN "software-pipelined K-loop" wall), NOT address
ALU. This PROMOTES Lever B (latency hiding / pipelining / occupancy) from secondary to the PRIMARY remaining lever
-- the bottleneck is exactly what B targets (memory-latency/scheduling), not what A targeted (ALU). No renderer
change made.
