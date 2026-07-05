# Route B: fused Q4_K int8 MMQ that tensorizes to iu8 WMMA (design spec)

Status: SPEC. Prereq: the GPU-free MMQ parity gate (`extra/qk/prefill_mmq_parity_gate.py`) passing,
so the WMMA kernel can be validated against the same dequant reference. Branch `int8-wmma-vocab`.

## Why this exists (and its contingency)

Fable-agent finding (see memory `int8-wmma-no-rdna3-throughput-win`): iu8 WMMA on gfx1100 runs at the
**same rate as fp16** (identical descriptor: dims=(16,16,16), epc=(16,16,8)). So Route B does NOT win by a
faster tensor core. Its ONLY lever over the already-wired scalar `_sdot4` MMQ is that **WMMA executes on
separate silicon from VALU**, so the int dot can overlap the per-group scale-correction VALU work. Whether
that nets a win is **contingent on Route A**: if scalar `_sdot4` MMQ is measured VALU-bound at pp512, Route B
is justified; if `_sdot4` already beats 365 tok/s, Route B is optional polish. Build it, but gate the wiring
decision on the Route A number.

## The MMQ algebra (what must tensorize)

Q4_K weight, super-block = 256 elems = 8 groups of 32. For output row n, column k with blk=k//256, g=(k%256)//32:

    W[n,k] = D[n,blk] * SC[n,blk,g] * q4[n,k]  -  DMIN[n,blk] * MN[n,blk,g]

    (D,DMIN are the fp16 super-scale/min; SC,MN are the 6-bit per-group scale/min; q4 in 0..15 UNSIGNED)

Activation x[m,k], q8_1-quantized: xq[m,k] int8, d8[m,j] fp32 per-32-block scale (j=k//32), x ≈ xq*d8.

Output, grouping the k-sum by 32-blocks (j = k//32, which selects blk and g):

    out[m,n] = Σ_j d8[m,j] * ( D[n]·SC[n,j]·RAW[m,n,j]  -  DMIN[n]·MN[n,j]·QSUM[m,j] )

where the two REDUCTIONS over the 32 elems of group j are:

    RAW[m,n,j]  = Σ_{k∈j} xq[m,k] · q4[n,k]     ← int8(signed) × int8(0..15) → int32   *** the WMMA ***
    QSUM[m,j]   = Σ_{k∈j} xq[m,k]               ← n-independent; = RAW with q4≡1  (or a cheap add-reduce)

Everything outside RAW/QSUM is per-group fp scalar work (VALU) that overlaps the WMMA.

## WMMA tiling

iu8 WMMA is 16×16×16 (M_tokens × N_cols × K). Q4_K group = 32 k → **2 WMMAs per group**, both int32-accumulated
into the SAME RAW[m,n,j] tile before the group scale is applied. Tile assignment:
- A fragment = xq activation tile: 16 tokens × 16 k  (int8; already the natural q8_1 layout)
- B fragment = q4 weight tile: 16 cols × 16 k  (q4 nibble unpacked to int8 0..15 — shift+mask, NO fp math)
- C/D = int32 [16 tokens × 16 cols] RAW accumulator
- Reduce loop: over j (groups) and the 2 K-halves within a group → accumulate RAW int32, then at group
  boundary fold `d8·D·SC·RAW − d8·DMIN·MN·QSUM` into the fp32 output accumulator.

## Codegen route (NO hand kernel)

Do NOT hand-emit WMMA UOps. Two candidate constructions, in order of preference:

1. **Tensor-expression + TC opt.** Express RAW as `xq_tile.matmul(q4_tile.transpose(), dtype=dtypes.int)` per
   group inside a generated-UOp kernel mirroring `q4k_q8_1_gemm_kernel` (q4_k_gemv_primitive.py:670), then apply
   the fp scale-fold in tensor land. The Fable result proved a plain `matmul(...,dtype=dtypes.int)` tensorizes to
   iu8 on the DEFAULT path — so the reduce must be shaped as a clean M×N×K int matmul for `_apply_tc_opt`
   (postrange.py) to match. Thread `OptOps.TC` through the kernel's `opts`/`_kernel_info(name, schedule, opts)`.
2. If the grouped-scale reduce won't present a clean matmul to the TC matcher (per-group scale breaks the K
   reduce into 32-chunks), split: emit RAW as a pure int8 matmul kernel (tensorizes cleanly), materialize
   RAW[m,n,j] int32, then a cheap second kernel does the fp scale-fold + QSUM correction. Costs an int32 RAW
   round-trip to VRAM (m·n·(k/32) int32) — acceptable only if fused (1) fails; measure both.

## Validation & wiring

- Reuse the Route A gate harness (dequant reference, rel_rmse < 6e-3) — add a WMMA-kernel case to
  `prefill_mmq_parity_gate.py`. First bit-exact/rel_rmse on DEV=PYTHON (math), THEN DEV=AMD (real WMMA emit —
  grep the kernel for `wmma_i32_16x16x16_iu8`; DEV=PYTHON is a false-positive oracle for WMMA presence).
- New route branch in `route_direct_packed_prefill` gated by e.g. `PREFILL_Q4K_Q8=wmma`, parallel to the
  existing `mmq`/`sdot4` branches — do NOT change the default until the pp512 number beats scalar `_sdot4`.
- pp512 authority: `extra/qk/prefill_whole_synced.py` on the 14B shapes vs 365 (fp16-dequant) and vs the Route A
  `_sdot4` number.

## Open risks
- The TC matcher may refuse the grouped reduce (construction 1) → fall back to split (2).
- iu8 B-fragment wants signed int8; q4 nibbles are 0..15 (fits signed int8 fine, no sign issue). xq is signed.
  The iu8 builtin sign flags are BOTH true (signed×signed) — q4 as small positive signed is correct.
- QSUM via `matmul(ones)` would waste a WMMA; prefer a plain add-reduce of xq per group (cheap, VALU, overlaps).
