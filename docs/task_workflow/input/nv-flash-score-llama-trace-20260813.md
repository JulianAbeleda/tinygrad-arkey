# NV flash score kernel trace - llama `flash_attn_ext_vec` vs tinygrad live-split tile (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `fb8116254`)
Status: **trace record (read-only).** Source-level trace of the decode flash score
kernel on both sides, to answer "what is llama doing that we are not" before any
implementation. No emitter, route, or policy change follows from this record.

## 0. Scope and the one sentence answer

The flash score gap is structural, not a per-token arithmetic defect. llama's
`flash_attn_ext_vec` computes the whole decode attention in **one pass**: Q is
loaded into registers once, K/V are streamed straight from global/L2 (no shared
memory staging), each 128-thread block scores 128 KV columns in parallel via
8-lane dots, and score + online-softmax + PV accumulation all happen in
registers in the same loop. The only separate kernel is the 4-way KV-split
combine.

tinygrad's `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` does the
same math but in a **tiled two-kernel decomposition**: it stages K and V into
shared memory per 16-token tile (two barriers per tile), re-reads Q from global
inside every token dot, does a full 32-lane shuffle reduce per token (16 tokens
serialized per warp), and always writes partial PV to global so a second
`flash_fused_gmax_combine` kernel can merge 48 splits.

The arithmetic is identical. The substrate is not.

## 1. Measurement provenance

Two independent measurements agree the gap is in score, not combine:

| source | llama score | tinygrad score | llama combine | tinygrad combine |
| --- | ---: | ---: | ---: | ---: |
| CUPTI node ledger (same measurement) | 113.9 us / 36 | 182.0 us / 36 | 120.5 us / 36 | 89.0 us / 36 |
| per-node | 3.16 us | 5.06 us | 3.35 us | 2.47 us |
| DEBUG=2 prime-token in-loop (NV) | 3.17 us | 7.52 us | 3.6 us | 3.39 us |

- CUPTI: `nv-tinygrad-node-ledger-gap-record-20260813.md` (tinygrad `DEV=CUDA`,
  same CUPTI method as the pinned llama ledger). flash score delta +68.1 us,
  combine delta -31.5 us.
- DEBUG=2 in-loop: `flash-score-tile-structure-measurement-record-20260803.md`
  section 6 (tinygrad `DEV=NV`). score 7.52 vs llama 3.17 us.

The two numbers for tinygrad differ because they measure different routes and
different things (CUPTI profiled node-sum on the CUDA route vs unprofiled
per-launch wall on the NV route); they agree on the direction and the
one-kernel focus. The combine is at or below llama parity on both routes, so it
is **not** recoverable mass.

## 2. llama `flash_attn_ext_vec` (decode path), traced

Source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`, HEAD
`ac4cddeb0` (the pinned audit commit). Decode is `Q->ne[1] == 1`, so the
launcher (`:549-563`) picks `cols_per_block = 1` and
`launch_fattn<D, 1, 1>(..., false)` (`:540`), i.e. `stream_k=false`.

### 2.1 Launch shape

`fattn-common.cuh:1083-1177` with `stream_k=false`:

- `ntiles_x = 1`, `ntiles_z_gqa = gqa_ratio = 4`, `ntiles_dst = 32`.
- `blocks_num = (1, parallel_blocks, 32)`; grid.z = 32 heads.
- `ntiles_KV = ceil(K->ne[1] / nbatch_fa)`, `nbatch_fa = D = 128`, so at
  context 512 the KV is split into `parallel_blocks = 4`.
- Block is `(warp_size=32, nwarps=4, 1)` = 128 threads, `__launch_bounds__(128, 1)`
  (`fattn-vec.cuh:9-11,20`).

Each of the 4 blocks per head owns one 128-column KV slice
(`k_VKQ_0 = blockIdx.y*128; k_VKQ_0 += gridDim.y*128`, `:250-256`), and the
`flash_attn_combine_results` kernel (`fattn-common.cuh:913-967`) merges the 4
partials afterward. That is the only place a second kernel exists, and only
because of the 4-way KV split, not because score and PV are separated.

### 2.2 The inner loop is single-pass

Q residency (`:204-250`, f16 branch): Q (128 f16) is loaded **once** into
`Q_reg[ncols][(D/2)/nthreads_KQ]` registers and scaled once by
`scale_h2 = make_half2(scale, scale)`. `nthreads_KQ = 128 / cpy_nb = 8`
(`cpy_nb = 16` on Volta+, `common.cuh:374-385`), so each of 8 lanes holds
`D/2/8 = 8` half2 = 16 halves = 128/8 dims of Q.

The KV loop (`:250-379`) then, for each 128-column chunk, does in registers:

1. **Score all 128 columns in parallel.** `i_KQ = threadIdx.y*32 +
   (threadIdx.x & ~7) + i_KQ_0` maps 4 warps x 4 groups x 8 lanes to 128
   columns (`:268-269`). Each 8-lane group computes one Q.K dot with
   `vec_dot_KQ` (half2 FMA, no tensor cores) then `warp_reduce_sum<8>` (3
   shuffle stages, `:273-274`).
2. **Online softmax in registers.** `KQ_max_new[j]` via
   `__shfl_xor_sync` from offset 8 (only 2 more stages, `:294-297`);
   `KQ_reg = exp(KQ_reg - KQ_max)`, `KQ_sum = KQ_sum*scale + KQ_reg`
   (`:298-303`).
3. **PV accumulation in registers.** `VKQ[j][...]` is rescaled by the running
   max (`:305-317`) and accumulated via `VKQ += tmp * KQ_k` (`:324-378`).

K and V are read straight from global/L2 (`K + i_KQ*nb11`, `V + k*nb21`); the
only shared write in the hot loop is `KQ[j*nthreads + tid] = KQ_reg[j]` at
`:303`, which broadcasts the exp'd score across warps for the PV phase. There
is **no per-tile K/V staging and no barrier inside the stream**.

### 2.3 Final cross-warp reduction and write

`:414-433` gathers per-warp `KQ_max`/`KQ_sum` into shared, re-reduces, then
`:434-500` reuses the `KQ` buffer to gather per-warp `VKQ` and writes `dst`
directly (each thread writes `D`-strided elements). One kernel, one pass, final
output; the 4-way split is merged by the separate 4-element combine kernel.

## 3. tinygrad live-split tile, traced

Source: `tinygrad/llm/flash_decode_attention.py:92-203` (score),
`:206-249` (combine). Production d512 is G4: Hq=32, Hkv=8, G=4, split S=48,
`query_group_size=None -> QG=G=4`, `stage_width=1`
(`decode_routes.py:607-608`, `flash_decode_attention.py:584-610`). So
`LANES=32, WARPS=4, THREADS=128, TK=16, R=4, RP=2` (`:99-104`), grid
`kvh(8) x split(48) x query_group(1)` = 384 blocks.

### 3.1 Per-tile LDS staging with two barriers

For each `TK=16` token tile (`block` reduce, `:135`), all 128 threads
cooperatively stage K **and** V from global into `ksh`/`vsh` shared buffers
(`:125-126, 153-156`), then `UOp.barrier` (`:156`). A second barrier closes the
tile after the PV merge (`:183`). So every 16-token tile pays a global->LDS
round-trip for K and V plus two barriers, on top of the score/PV work.

### 3.2 Q re-read per token, full 32-lane reduce, serialized tokens

`dot_reduce(token_in_tile)` (`:160-173`) loads Q from global
(`q[head*Hd + elem]`, `:166`) **inside** the per-token loop, so Q is re-read for
every token in every tile rather than held in registers. The dot is
`RP=2` `fdot2` per lane (`:164-169`) followed by `_warp_reduce_sum_staged`
(`:171-172`), a full 32-lane XOR-shuffle ladder = 5 stages
(`codegen/late/warp_reduce.py:79-84`). The `token_in_tile` reduce (`:185`) runs
the 16 tokens serially: 16 tokens x 5 shuffle stages = 80 serialized stages per
warp per tile, versus llama's 128 columns scored in parallel with a 3-stage
8-lane reduce.

Online softmax and sharded PV (`:175-183`) match llama's arithmetic, but PV is
sharded as `R=4` dims per lane into `acc`, then written to **global** `pout`
with per-split `den`/`mx` (`:194-199`).

### 3.3 Always a separate combine kernel over 48 splits

`flash_fused_gmax_combine_kernel` (`:206-249`) reads the 48 split partials back
from global, recomputes the global max over all splits, and re-normalizes PV
(`:215-247`). This runs for every token and is the price of the 48-way split +
global-partial ABI. llama's equivalent merges only 4 splits.

## 4. Structural delta

| axis | llama `flash_attn_ext_vec` | tinygrad live-split tile |
| --- | --- | --- |
| passes | 1 (score+softmax+PV in one kernel) | 2 (score/PV kernel + combine kernel) |
| Q residency | registers, loaded once, scaled once | global, re-read per token per tile |
| K/V residency | streamed straight from global/L2 | staged to LDS per 16-token tile, 2 barriers/tile |
| dot width | 8-lane group, 3 shfl stages, 128 cols parallel | 32-lane full reduce, 5 shfl stages, 16 tokens serial |
| KV split | 4 (combine merges 4 partials) | 48 (combine merges 48 partials) |
| output | written directly in the score kernel | partial PV + den + max to global, combine re-reads |

This is the same conclusion the 08-03 structural sweep reached empirically: the
tile structure's own zero-load ceiling is 5.31 us, still ~1.7x llama's 3.16 us
floor, so no tile geometry or staging width closes the gap. The recoverable
substrate is register-resident Q + no-LDS K/V staging + a parallel per-column
dot (8-lane groups) + single-pass score/softmax/PV, i.e. a structural emitter
change, not a values or geometry change.

## 5. Tok/s arithmetic and the overlap caveat

Closing the CUPTI flash-score delta (+68.1 us at 36 nodes) is ~+3 tok/s at the
1:1 removal mapping (~25 us/token ~ +1 tok/s in the 190-205 band). The DEBUG=2
in-loop delta (7.52 vs 3.17 us, +4.35 us x 36 = ~157 us) bounds a larger ~+6
tok/s if the rewrite also removes the staging/launch overhead.

Important: the 08-05 causal record concluded llama's flash runs **overlapped
with MMQ** (only ~57.6 us exposed) while native serializes. A flash rewrite only
earns wall credit through a token-identical real-token A/B that also lets it
overlap the GEMV chain; the kernel-level number is necessary but not sufficient.

## 6. What a substrate fix would have to do (for the next scope, not this one)

1. Hold Q in registers per (head, warp) for the whole kernel; scale once.
2. Stream K/V from global/L2 instead of staging to LDS per tile; drop the two
   per-tile barriers.
3. Replace the 32-lane full reduce with 8-lane groups so multiple KV columns
   score in parallel per warp (3 shuffle stages, not 5, and not serialized over
   a 16-token tile).
4. Fuse score + online-softmax + PV into one kernel and write output directly;
   keep the combine only for the KV split, and make that split coarse (llama
   uses 4, we use 48).

This must render identically across the AMD/Metal arms and re-gate on the
fixed-depth wall + token sha before promotion, per the standing process.

## Evidence

- llama source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`
  (`flash_attn_ext_vec`), `fattn-common.cuh:913-967` (`flash_attn_combine_results`),
  `fattn.cu:250` (vec dispatch), HEAD `ac4cddeb0`.
- tinygrad source: `tinygrad/llm/flash_decode_attention.py:92-249`,
  `tinygrad/codegen/late/warp_reduce.py:79-84`, `tinygrad/llm/decode_routes.py:607-608`.
- Prior records: `nv-tinygrad-node-ledger-gap-record-20260813.md` (CUPTI numbers),
  `flash-score-tile-structure-measurement-record-20260803.md` (DEBUG=2 numbers and
  the zero-load structural ceiling), `nv-decode-native-flash-causal-record-20260805.md`
  (overlap caveat).
