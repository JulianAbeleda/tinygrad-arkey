# Instruction-count measurement: is there headroom BELOW fp's dequant? (2026-06-15)

Method: disassemble the actually-emitted kernels (`to_program` -> `Ops.BINARY` lib ->
`/opt/rocm/llvm/bin/llvm-objdump -d`), count VALU instructions in the hot reduce-loop body,
normalize per weight (weights/body = #`v_cvt_f32_ubyte0` for fp; 256/block for int-dot).
This answers the only open question from the consolidation: does fp sit near the Q4_K instruction
floor, or is there real instruction headroom below it?

## Measured: VALU per weight in the hot body
| kernel | hot-body VALU | weights/body | **VALU/weight** | dot instruction |
|---|---|---|---|---|
| **fp** (dequant + fp dot) | 130 | 32 | **4.06** | `v_fma_f32` (scalar, 1 MAC) |
| **int-dot** (tinygrad q8_1) | 860 | 256 | **3.36** | `v_mad_i32_i24` (scalar, 1 MAC) |
| `v_dot4` present in EITHER? | — | — | — | **NO** (`grep v_dot` = 0 in both) |

### fp per-weight breakdown (4.06/weight)
| op | /weight | role | reducible by DP4A int-dot? |
|---|---|---|---|
| `v_bfe_u32`+`v_and`+`v_lshrrev` | ~1.0 | 4-bit nibble + 6-bit scale unpack | **NO -- essential floor** |
| `v_cvt_f32_ubyte0` | 1.0 | int weight -> fp32 convert | **YES -- gone (weights stay int)** |
| `v_fma_f32` | 1.0 | the dot MAC (scalar) | **YES -> `v_dot4` = 0.25 (4 MACs/instr)** |
| `v_fma_mix_f32` | 1.0 | per-weight affine `d*sc*q - dmin*mn` | **YES -- folds to per-group, ~0.03** |

## The answer: YES, ~3x instruction headroom exists -- and it is ENTIRELY in the dot.
The DP4A int-dot floor for Q4_K:
- nibble unpack ~1.0/weight (irreducible -- you must spread 4-bit nibbles to int8 lanes either way),
- `v_dot4_i32_i8` dot = **0.25/weight** (one instruction = 4 int8 MACs),
- per-group affine + qsum amortized ~0.1/weight,
- **NO fp convert, NO per-weight fp fma.**
=> DP4A floor ~= **1.35 VALU/weight** vs fp's **4.06** -> **~3x fewer instructions/weight.**

This is consistent with llama.cpp running 1.8x faster (58 -> 104 tok/s): it captures most of the
instruction headroom (the residual <3x is because nibble-unpack and HBM still cost). **fp is NOT near
the Q4_K instruction floor** -- it wastes ~1.0/weight on the int->fp convert and ~0.75/weight on scalar
(vs packed) MACs. The READRAW experiment said the dequant ALU halves bandwidth; this says ~3/4 of that
ALU is reducible in principle.

## BUT the headroom is locked behind a codegen capability tinygrad lacks.
The win lives ENTIRELY in `v_dot4`, and **tinygrad emits zero `v_dot4` in either kernel** (measured).
Its only int-dot lowering produces `v_mad_i32_i24` -- a *scalar* int MAC (0.97/weight, same count as
fp's fma, no packing) -- plus qsum overhead (`v_add_nc_u32` 0.375/wt) and cross-lane `v_readfirstlane`
(0.156/wt) and higher register pressure (VGPR 68 vs 47). That is why the tinygrad int-dot is SLOWER e2e
(28 vs 58) despite a marginally lower raw VALU/weight: its instructions are worse (cross-lane, int24,
more registers -> worse occupancy/pipelining), and crucially it never reaches the 0.25/weight packed
dot. Without DP4A lowering, the int path is strictly worse than fp; WITH it, the floor is ~3x below fp.

## Conclusion (grounded, no build)
1. There IS large instruction headroom below fp (~3x VALU/weight), so fp at 58 tok/s is NOT a Q4_K
   instruction-count floor -- it is a tinygrad-codegen floor.
2. The entire headroom is the packed dot (`v_dot4`), which tinygrad's codegen does not emit (only the
   `Ops.CUSTOMI` inline-asm escape hatch reaches it, and that blocks the optimizer -- W2/Q0a wall).
3. This QUANTIFIES the "hand-asm / Writer" boundary precisely: the decode gap to llama.cpp is a ~3x
   instruction-count gap in the dot, realizable ONLY by a DP4A codegen lowering (or hand asm). It is a
   real, physics-justified lever -- but it requires adding `v_dot4` lowering to tinygrad's renderer,
   not a schedule/search trick. Single-layer search over today's primitives cannot reach it. This is
   consistent with every prior decode result and with the Mirage probe's hand-asm conclusion -- now
   measured in instructions/weight rather than inferred.
