# AMD Q4_K Decode -- Consolidated First-Principles Diagnosis (2026-06-15)

Purpose: separate MEASURED facts from THEORIES, and ground the decode bottleneck in a controlled
experiment instead of inference. (We guessed wrong once -- occupancy -- so every claim here is tagged
MEASURED / REFUTED / CONFIRMED with how it was settled.)

## Hardware peaks (MEASURED)
- HBM bandwidth peak: **859 GB/s** (warm streaming copy, 89% of 960 datasheet).
- fp16 compute peak: 83.64 TFLOPS.
- llama.cpp 8B Q4_K single-stream decode: **103.84 tok/s** (~500 GB/s, ~58% of HBM peak).

## Decode kernels (MEASURED)
| kernel | standalone GB/s (ffn_gate) | e2e tok/s | e2e GB/s | VGPR | LDS |
|---|---|---|---|---|---|
| fp partial (dequant+fp dot) | 173-365* | **58** | 278 | 47 | 0 |
| int-dot (int dot + qsum) | 242 | 28 | 136 | 68 | 0 |
| coop fused (LDS+barrier) | 409 | 24 | 117 | 93 | 4608 |
*fp standalone varies by harness (173 in the q8_1 bench, 365 in the read-raw bench, same kernel).

## THE DECISIVE EXPERIMENT (the one that settles ALU-bound vs memory-bound)
Same weight bytes, same per-row read pattern, the ONLY variable changed is the dequant ALU:
- **READRAW** (read the 36 words/block, sum them, NO dequant): **730 GB/s = 85% of HBM peak**.
- **FP GEMV** (read + dequant + dot): **365 GB/s = 42% of peak**.
The memory system delivers 730 GB/s for this exact pattern; the dequant ALU HALVES it to 365. This is
a controlled experiment, not an inference: **the decode GEMV is ALU/dequant-instruction-bound.**

## Theory ledger -- what is settled, and how
| theory | verdict | settled by |
|---|---|---|
| memory-bound (at the bandwidth ceiling) | **REFUTED** | read-raw = 730 GB/s; GEMV = 365 -> memory can do 2x more |
| occupancy / register pressure | **REFUTED** | measured VGPR 47/68/93, all <=96 -> all full 16-wave occupancy (identical) |
| **ALU / dequant instruction count** | **CONFIRMED** | read-raw 730 vs GEMV 365 + M0's ~3862 ALU ops/kernel (~55 ops/global load) |
| "latency-bound" (M0's wording) | **imprecise** | if latency-bound, read-raw (same pattern) would also be ~365; it's 730 -> it's instruction *throughput*, not stall latency |

## First-principles diagnosis (now grounded, not guessed)
The decode GEMV is bound by **dequant instruction throughput**. The dequant (~3862 ALU ops/kernel: 4-bit
nibble unpack via shift/mask/cndmask/alignbit + ubyte->fp32 convert + the d*sc*q - dmin*mn affine, per
weight) consumes the cycles the GPU could spend issuing loads. At full occupancy the bottleneck is how
many instructions sit between memory transactions (consistent with the autotuning literature: "the
kernel is so memory-starved that the bottleneck is instructions between memory transactions").

The single lever the physics allows: **fewer dequant instructions per weight.** Everything we measured
is consistent with this and nothing else:
- int-dot lost (28 e2e) because it ADDED instructions (the qsum reduction + int->fp affine).
- hoist_scale_min regressed -80% because it BLOATED ALU (5150 vs 3862 ops).
- packed_load +6% (wider loads) helped marginally -- loads weren't the bottleneck (already b128).
- fp wins by being the SIMPLEST dequant tinygrad emits.
- llama.cpp wins (58% vs our 32% of peak) because its DP4A dot is FEWER instructions/weight (one
  v_dot4 = 4 MACs; packing folds the qsum into the dot).

## The open question -- now MEASURED (see q0a/INSTRUCTION_COUNT_RESULT.md)
Q: is there instruction headroom below fp's dequant, or does fp sit near the Q4_K instruction floor?

A (measured from disassembly, VALU/weight in the hot body):
- **fp = 4.06 VALU/weight** (1.0 nibble-unpack + 1.0 int->fp convert + 1.0 scalar dot-fma + 1.0 affine).
- **DP4A int-dot floor ~= 1.35 VALU/weight** (1.0 nibble-unpack + 0.25 `v_dot4` + ~0.1 amortized affine).
- => **~3x instruction headroom below fp, ENTIRELY in the dot.** fp is NOT a Q4_K instruction floor; it
  is a tinygrad-codegen floor (it wastes the convert + scalar-vs-packed MAC).
- BUT: **tinygrad emits zero `v_dot4`** (measured in both fp and its int-dot kernel). Its only int path
  emits scalar `v_mad_i32_i24` (no packing) + qsum + cross-lane `v_readfirstlane` + more registers ->
  strictly worse e2e (28 vs 58). The headroom is locked behind a DP4A codegen lowering tinygrad lacks.

Implication: the decode gap to llama.cpp is a ~3x instructions/weight gap in the dot, realizable ONLY
by adding `v_dot4` lowering to the renderer (or hand asm) -- NOT a schedule/search trick over today's
primitives. This is the "Writer / hand-asm" boundary, now quantified in instructions/weight rather than
inferred. It decides the decode question: no single-layer-search decode work is worth doing; the only
physics-justified lever is a DP4A codegen capability, which is a renderer feature, not a search result.

## UPDATE (2026-06-15) -- the v_dot4 lever was BUILT, and the e2e verdict is NULL (D0/D1)
We then realized the lever via the SCHEDULABLE builtin `__builtin_amdgcn_udot4` (not the asm-volatile
barrier Phase D mistakenly used). `dp4a-d0/{BUILTIN_VS_ASM_RESULT,D1_E2E_RESULT}.md`:
- KERNEL level: builtin udot4 GEMV = **302 Q4-GB/s, 1.77x faster than fp** (171), correct, ~1.58
  VALU/weight (the predicted floor). The instruction-count headroom IS real and reachable. So the
  "VALU/weight is the lever" diagnosis above is confirmed AT THE KERNEL LEVEL.
- E2E level: **decode tok/s UNCHANGED (30.2 vs fp 30.3)** despite the 1.77x kernel + half the bytes/token.
  The GEMV kernel throughput was never the e2e bottleneck -- decode is latency/launch-bound at the TOKEN
  level. So the per-kernel instruction-count win does NOT cash out to tok/s.
Corrected implication: the consolidated diagnosis (GEMV kernel is ALU/instruction-bound) is RIGHT, but it
is the wrong LEVEL for the e2e gap. fewer dequant instructions makes the kernel faster and does nothing
for decode tok/s, because the token is latency-bound across ~252 launches, not GEMV-throughput-bound.
The decode lever hunt is closed: the last kernel-level lever is real-but-null e2e; the residual gap is
structural (cross-kernel latency), the cross-layer frontier, not a single-kernel codegen feature.
