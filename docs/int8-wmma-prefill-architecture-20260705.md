# int8 WMMA prefill for Q4_K/Q6_K — architecture + implementation (2026-07-05)

Status: CODE WRITTEN, NOT YET RUN. This document is the exhaustive design; the code changes it
describes are committed to the working tree behind flags (all additive, default OFF, so the current
365-tok/s path is untouched). Run + validate per §7 tomorrow.

Companion measured context: `docs/prefill-packed-generated-tile-scope-20260704.md`
(2026-07-05 sections: the VALU ceiling, the fp16 memory wall, the llama-int8-MMQ trace, the scoreboard).

--------------------------------------------------------------------------------
## PROGRESS (2026-07-05, branch int8-wmma-vocab; GPU-free pieces validated)
- **int8 MMQ math VALIDATED (numpy, CPU):** rel_rmse 4.8e-3 vs fp32 ref (int8-activation-quant error only,
  int32 range safe). llama's Q4_K decomposition confirmed: `W=d*sc*q4-dmin*mn`; quantize activations per-32-group
  to int8 (xq,xsc,xsum); `DOT=sum_{k in g} q4*xq` (int32); `out=sum_g(d*sc*xsc*DOT - dmin*mn*xsc*xsum)`. The
  primitive is now a TRANSCRIPTION, not a research question. Repro: scratchpad/q4k_int8_mmq_math.py.
- **iu8 tensor core ADDED to vocab (tc.py + cstyle.py), CPU-verified:** first int8 TC in tinygrad;
  `(dtypes.char, dtypes.int)` in amd_rdna3 (same 16x16x16/32-thread/swizzle as fp16; lane_map.validate() passes).
  Renderer wrapper around `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (char16->int4 pack, signed*signed --
  reuse the value-validated `_sdot4` signedness). Build imports, descriptor validates in-list. Committed WIP.
- **LDS input-staging (Track 2A) is REUSABLE for iu8** -- it stages any WMMA operand (reads fragment axes
  dynamically); the int8 MMQ tile staging (llama's shared-mem tiles) inherits it once milestone-2 (partition) lands.
- REMAINING (GPU-gated): (a) confirm an int8 matmul tensorizes to iu8 WMMA + correct numerics on AMD
  (scratchpad/iu8_wmma_test.py -- verify the intrinsic signedness bools + char16->int4 pack; NEVER trust
  DEV=PYTHON::gfx1100, it false-positives); (b) transcribe the int8 Q4_K primitive (unpack nibbles->int8,
  quantize activations->Q8_1 int8, reduce int32, per-group scale/min correction) so its int8 dot tensorizes;
  (c) apply the LDS-staging; (d) wire the route + measure pp512 vs 365 baseline and vs llama 1849.

## 0. One-paragraph summary

llama.cpp does 14B Q4_K prefill at 1849 tok/s (vs our 365) with **MMQ**: everything stays quantized —
weights Q4_K, activations quantized to int8 (Q8_1) — and the matmul runs in **int8** on the RDNA3 int8
WMMA (`v_wmma_i32_16x16x16_iu8`), with the Q4_K per-group scales/mins applied AFTER the int32
accumulation. tinygrad can already emit fp16 WMMA but the fp16 path OOMs on 14B (fp16 weights don't fit)
and tinygrad has **no int8 tensor core in its vocabulary**. This change (a) adds the iu8 int8 WMMA to
tinygrad's RDNA3 codegen vocab, and (b) adds a Q4_K/Q6_K prefill primitive expressed as an int8 matmul so
tinygrad's tensor-core matcher tensorizes it to iu8 WMMA. No hand-written kernel: the primitive is a
tinygrad tensor/UOp expression; the WMMA comes out of the codegen.

--------------------------------------------------------------------------------
## 1. Why int8 (recap of the measured dead-ends)

- VALU direct-out (current default, generated UOp): **365 tok/s** — the VALU ceiling; grouping regresses.
- fp16 WMMA, weights materialized: 21 TFLOPS/GEMM but **OOMs whole-model at 23.8 GB** (fp16 weights don't
  fit; this is exactly why the router gates 14B to VALU — `_pf16_w` is None).
- fp16 WMMA, dequant fused-lazy: fits memory but **3 TFLOPS** (Q4_K dequant runs redundantly per
  M-fragment).
- fork dp4a (`PREFILL_Q4K_Q8=sdot4`): **17 tok/s** (no LDS tiling); `mmq`: OOM.
- llama int8 MMQ: **1849 tok/s** — the target.

int8 wins on BOTH axes vs fp16: memory (int8/int4 weights never expand to fp16) and compute (iu8 WMMA is
~2x the fp16 WMMA rate on RDNA3). It is the only measured-viable structural path.

--------------------------------------------------------------------------------
## 2. The math (what the int8 matmul computes)

Q4_K weight for output row n, input col k, group g = k//32 (32 cols/group), super-block fp16 d/dmin,
per-group 6-bit s6/m6:  W[n,k] = d[n]*s6[n,g]*q4[n,k] - dmin[n]*m6[n,g].
(q4 in 0..15.)

The GEMM out[m,n] = sum_k W[n,k]*x[m,k]. Quantize activations per group: x[m,k] ~= xsc[m,g]*xq[m,k]
(xq int8, xsc fp16 per-group scale). Then, grouping the sum by g:

  out[m,n] = sum_g { d[n]*s6[n,g]*xsc[m,g] * DOT[n,m,g]  -  dmin[n]*m6[n,g]*xsc[m,g] * XSUM[m,g] }

  where  DOT[n,m,g]  = sum_{k in g} q4[n,k] * xq[m,k]      (int8 x int8 -> int32; THE WMMA)
         XSUM[m,g]   = sum_{k in g} xq[m,k]                (int; for the min correction)

This is llama's MMQ decomposition. Q6_K is the same minus the min term (no m6/dmin, q6 signed):
  out[m,n] = sum_g d[n]*s8[n,g]*xsc[m,g] * DOT[n,m,g].

Key structural point: the int dot DOT is a clean `reduce_k MUL(int8, int8) -> int32`, which is exactly
what tinygrad's `_apply_tc_opt` tensorizes — once an iu8 tensor core exists. The per-group scale
corrections are ordinary fp32 elementwise applied to the int32 group partials, then reduced over g.

--------------------------------------------------------------------------------
## 3. Change 1 — VOCAB: add iu8 int8 WMMA to tinygrad's RDNA3 tensor cores

Files: `tinygrad/codegen/opt/tc.py`, `tinygrad/renderer/cstyle.py`. Additive; default fp16 path untouched.

### 3a. tc.py — the descriptor
The RDNA3 iu8 WMMA is structurally identical to the fp16 one: dims (16,16,16), 32 threads,
elements_per_thread (16,16,8), SAME opts + swizzle. Only the dtypes change: dtype_in = int8, dtype_out =
int32. We add it as a second entry in `amd_rdna3` so `_apply_tc_opt` (which iterates tensor_cores and
matches on dtype) uses it for int8 matmuls and the fp16 one for half matmuls:

```python
amd_rdna3 = [TensorCore(dims=(16,16,16), threads=32, elements_per_thread=(16,16,8), dtype_in=di, dtype_out=do,
    opts=("l0","l0","l0","l0","l1","u1","u1","u1"),
    swizzle=(...same as today...))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),
                (dtypes.char,dtypes.int)]]   # <-- ADDED: int8 in, int32 out (iu8 WMMA)
```
VALIDATE: elements_per_thread for iu8 A/B is 16 int8/lane (4x int32 packed) — same count as fp16's 16
half/lane, so (16,16,8) holds. The swizzle/lane_map must lower the same way for int8; `__post_init__` +
`lane_map.validate()` will assert if not. If validate() fails for int8, the swizzle needs an iu8-specific
LaneMap (unlikely — dims/threads/epc identical to fp16).

### 3b. cstyle.py — the render (the one genuinely HW-specific bit)
Current RDNA3 render (line ~443) only handles fp16/bf16 (`dtype_out == dtypes.float`). Add an int32 case.
The iu8 intrinsic signature differs from fp16 (it has signedness + clamp bool args), so it needs a
wrapper (like the existing half8 wrapper at line 445), plus the A/B operands packed int8x16 -> int32x4:

```python
# in HIPRenderer.render_kernel, the `for name,(N,M,K),dtype_in,dtype_out,... in wmma_args(uops)` loop,
# RDNA3 branch, ADD before the final `else`:
elif dtype_out == dtypes.int:   # iu8 int8 WMMA
    prefix.append(f"static inline __attribute__((device)) int8 __{name}(int4 a, int4 b, int8 c) {{\n"
                  f"  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);\n}}")
    # true,true = signed A,B (both operands are signed int8); false = no saturating clamp.
```
And the operand bitcast: tinygrad hands the WMMA int8x16 A/B vectors; the intrinsic wants v4i32. Mirror
the existing fp16-half8 bitcast rewrite (cstyle.py ~line 375) with an int8-case:
```python
(UPat(Ops.WMMA, name="x", dtype=dtypes.int.vec(8)),
 lambda x: UOp(Ops.WMMA, x.dtype, (x.src[0].bitcast(dtypes.int.vec(4)), x.src[1].bitcast(dtypes.int.vec(4)),
              x.src[2]), x.arg) if x.src[0].dtype != dtypes.int.vec(4) else None),
```
VALIDATE (highest risk, check tomorrow with DEBUG=4 on a tiny int8 matmul):
  1. intrinsic name `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` compiles under this HIP/clang.
  2. the two sign bools: signed*signed. If Q4 codes are treated unsigned (0..15) use `false` for A.
  3. A/B packing order (int8x16 -> int4) matches the swizzle's expected fragment layout — a wrong pack
     gives wrong numerics, not a compile error. Cross-check vs a reference int8 matmul (§7 T1).
  4. `int4`/`int8` vector typedefs exist in the emitted prefix (render_vector_prefix) for int.

--------------------------------------------------------------------------------
## 4. Change 2 — VOCAB/PRIMITIVE: int8 Q4_K prefill kernel

File: `extra/qk/quant/q4_k_gemv_primitive.py` (new fn `q4k_gemm_int8_tc_kernel`), plus an activation
quantizer. Expressed as tinygrad UOps/Tensors so the int dot tensorizes to iu8 WMMA (Change 1).

Shape [m=512 tokens, n=out_f, k=in_f]. Steps (see §2 math):
1. Activation quant (separate small kernel / tensor expr): xq[m,k] int8, xsc[m,g] fp16, XSUM[m,g] int32,
   per group g of 32. (llama's block_q8_1 layout: int8 + scale + sum.)
2. Weight codes: unpack Q4_K nibbles -> q4[n,k] as int8 (0..15, or shift to signed -8..7; MUST match the
   sign bool in §3b).
3. Inner int dot (tensorizes): for group tile, `DOT[n,m,g] = reduce_{kk} MUL(q4_i8[n,g,kk], xq_i8[m,g,kk])`
   -> int32. Express the k-reduce so K aligns to the WMMA (K=16; a group=32=2 WMMA steps unrolled).
4. Corrections (fp32, elementwise + reduce over g): 
   `out[m,n] = reduce_g( d[n]*s6[n,g]*xsc[m,g]*DOT - dmin[n]*m6[n,g]*xsc[m,g]*XSUM )`.
Output fp32 [m,n] (then cast as the route expects).

The primitive carries NO opts_to_apply -> the heuristic path runs -> `_apply_tc_opt` (with TC=1, and for
CAST'd operands TC_OPT>=1) tensorizes step 3's int reduce to iu8 WMMA. Q6_K variant: drop the min term.

--------------------------------------------------------------------------------
## 5. Change 3 — ROUTE wiring

File: `tinygrad/llm/prefill_routes.py`. New flag `PREFILL_Q4K_INT8_TC` (default 0). When set, the
direct-packed branch (where Q4_K prefill currently binds for 14B) routes to the int8-TC primitive instead
of the VALU direct-out kernel. Default OFF keeps the 365 path. Env to also set at run: `TC=1 TC_OPT=1`
(TC_OPT>=1 lets the matcher accept the CAST'd/quantized operands).

Selection precedence inside `route_direct_packed_prefill`:
  PREFILL_Q4K_INT8_TC=1 and quant in {q4k,q6k} and gfx1100 -> int8-TC primitive; else existing behavior.

--------------------------------------------------------------------------------
## 6. Why this fits memory (the whole point)

Weights stay Q4_K in HBM (~9 GB); only int8 activation tiles + int32 accumulators touch LDS/registers.
No fp16 weight buffer is ever materialized -> no 31 GB, no OOM. This is the same memory profile as llama.

--------------------------------------------------------------------------------
## 7. Validation plan (RUN TOMORROW, in order; never kill a live DEV=AMD run)

Env for all: `DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 DEV=AMD TC=1 TC_OPT=1 PYTHONPATH=.`

T1 (codegen unit, no model): tiny int8 matmul `Tensor.empty(64,64,int8) @ ...` -> DEBUG=4, confirm
   `__builtin_amdgcn_wmma_i32_16x16x16_iu8` appears AND result matches a numpy int32 reference.
   -> validates Change 1 (the risky part) in isolation. Script: scratchpad `int8_tc_probe.py` (written).
T2 (primitive numerics, small shape): int8 Q4_K primitive vs the existing fp32 dequant reference on a
   [128,256,256]-ish Q4_K block -> rel_rmse < ~1e-2 (int8 activation quant adds error; llama-level ok).
   Script: scratchpad `q4k_int8_correct.py` (written).
T3 (per-GEMM perf): hot shape [512,17408,5120], int8-TC vs VALU -> TFLOPS + confirm iu8 WMMA emitted +
   fits memory (no OOM). Script: scratchpad `q4k_int8_realshape.py` (written).
T4 (whole-model): `PREFILL_Q4K_INT8_TC=1 ... prefill_whole_synced.py --model .../Qwen3-14B-Q4_K_M.gguf
   -K 5` -> pp512 vs 365 baseline and vs llama 1849. Correctness via output parity vs VALU.

If T1 fails -> the iu8 render/descriptor is wrong (see §3b VALIDATE); fix there before anything else.
If T1 passes but T3 is slow -> scheduling (LDS staging of the int8 tile); tune LOCAL/UPCAST on the primitive.

--------------------------------------------------------------------------------
## 8. Risks / open questions (ranked)

1. iu8 intrinsic signature + operand pack (§3b) — the one真正 HW-specific unknown; T1 isolates it.
2. Does `_apply_tc_opt` accept int8 CAST'd operands + the group-structured reduce? (fp16 dequant did, §
   scope-doc; int8 is analogous but unverified.) T2/T3 tell us.
3. Activation-quant error acceptable? llama ships it; parity check in T4.
4. Scheduling: does tinygrad stage the int8 weight tile in LDS once (not re-unpack per M)? If not, tune
   opts. This is the same LDS-staging question as the fp16 path, but int8 tiles are half the size.
