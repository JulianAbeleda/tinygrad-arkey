# NV flash-decode score codegen structural diff: tinygrad generated tile vs llama handwritten vec kernel

Date: 2026-08-19
Branch: `nvidia-bringup-20260731`
Status: **read-only source analysis.** No code was modified, no GPU commands were run,
and no network was used. The deliverable is this file.

Subjects:

- tinygrad emitter:
  [flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:142)
  (`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`) and its reduce helpers in
  [warp_reduce.py](/home/ubuntu/tinygrad-arkey/tinygrad/codegen/late/warp_reduce.py:105).
- llama.cpp handwritten kernel:
  [fattn-vec.cuh](/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh:21)
  (`flash_attn_ext_vec`) and its launch/config helper in `fattn-common.cuh`.
- isolation microbench:
  [llama_fattn_vec_iso.cu](/home/ubuntu/tinygrad-arkey/extra/llm_research/microbench/llama_fattn_vec_iso.cu:1).

Shape under analysis: Qwen3-8B decode, `Hq=32`, `Hkv=8`, `Hd=128`, context 512.
tinygrad production is G4: `S=48`, `lane_width=32`, `warps=4`, `token_block=16`,
`stage_width=1`, `reduce_structure=staged`, `dot_pair_width=2`; the control kernel name is
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`.

## 1. Structural parameter table

| parameter | tinygrad generated tile (`..._32_128`, S=48) | llama `flash_attn_ext_vec<128,1,F16,F16,false>` |
| --- | --- | --- |
| total threads per threadgroup | 128 (`THREADS = LANES*WARPS = 32*4`) | 128 (`ggml_cuda_fattn_vec_get_nthreads_host/device` returns 128; 4 warps x 32 lanes), `__launch_bounds__(128, 1)` |
| lanes per warp | 32 (`LANES = lane_width`) | 32 (`WARP_SIZE`) |
| warps | 4 (`WARPS = QG = G = Hq/Hkv = 4`) | 4 (`nwarps = 128/32`) |
| KV split count | 48 (`S`); one score launch has grid `Hkv x S x NG = 8 x 48 x 1 = 384` blocks | source-derived d512 install is 4 (`parallel_blocks`, grid `(1,4,32)` = 128 blocks); the isolation microbench pins `grid.y=2` = 64 blocks |
| serial token loop / tokens per thread | `token_in_tile` REDUCE over `TK=16`; each warp scores 16 tokens serially, one full 32-lane reduce per token; at d512 there is exactly `NB=1` tile per split | `i_KQ_0` loop over `nthreads_KQ=8`; each step has 16 columns in flight (4 warps x 4 groups), 8 steps cover 128 columns per chunk |
| dot-pair width / per-lane Q slice | `dot_pair_width=2` -> `RP=2` fdot2 per lane per token; `R=Hd/LANES=4` dims/lane | `Q_reg[ncols][8]` half2 = 16 dims/lane; 8 half2 FMAs per column dot (`vec_dot_fattn_vec_KQ_f16<128,8>`) |
| score reduce width / stages | full 32-lane reduce, 5 shuffle stages (offsets 16,8,4,2,1) | 8-lane group reduce, 3 shuffle stages (offsets 4,2,1) |
| cross-group softmax max | none inside score: the 32-lane sum is warp-uniform, so `new_max = old_max.maximum(score)` needs no extra ladder | 2 more shuffle stages (offsets 8,16) unify the running max over the 4 groups of a warp |
| online-softmax staging | register recurrence `acc[R=4]` + `den[1]` + `mx[1]`; warp-uniform max; one barrier closes each 16-token tile | register `KQ_max/KQ_sum/KQ_reg` plus register PV `VKQ`; shared `KQ_max_shared`/`KQ_sum_shared` plus `__syncthreads` re-reduce at chunk end |
| Q heads -> warps | warp owns one GQA head: `grouped_head = query_group*QG + warp`, `head = kvh*G + grouped_head`; 4 heads per block | block owns one Q head (`blockIdx.z`); all 4 warps work the same head and split 128 KV columns (`threadIdx.y*32`) |
| Q residency | global re-read inside every token dot (`q[head*Hd + elem]`) | register `Q_reg`, loaded once, scaled once |
| K/V residency | staged to LDS `ksh`+`vsh` per 16-token tile, with a staging barrier and a tile-end barrier | streamed from global/L2; only the exp'd score `KQ[j*nthreads+tid]` is written to shared |
| LDS / shared memory | `ksh` + `vsh` = 2 x 2048 half = 8192 bytes (8 KiB) | `KQ` = max(128, 2048) half = 4096 bytes + `KQ_max_shared` 128 bytes + `KQ_sum_shared` 128 bytes = 4352 bytes (~4.25 KiB static) |
| output ABI | partial PV/den/max to global `pout` of `Hq x S x (Hd+2)` = 32 x 48 x 130 floats (~780 KiB), then a second combine kernel over 48 splits | direct `dst` when `gridDim.y==1`; otherwise `dst_meta` partials + a 4-way combine (`dst_tmp` ~64 KiB) |
| launches per token | 36 score launches (one per layer) + 36 combine launches | 36 score launches (one per layer, grid.y=4) + 36 combine launches |

Both kernels use 128 threads / 4 warps, so the difference is not block size. It is what
each warp owns: tinygrad gives each warp a whole Q head and serializes 16 tokens under a
32-lane reduce; llama gives each warp a 32-column KV slice, splits that into four 8-lane
groups, and scores 128 columns per chunk.

## 2. The "shape" phrase: what is correct and what is not

The ledger shorthand appears in
[nv-fuse-hide-eliminate-ledger-20260818.md](/home/ubuntu/tinygrad-arkey/docs/task_workflow/input/nv-fuse-hide-eliminate-ledger-20260818.md:28):

> tinygrad 32-lane/5-stage/16-serial/48-split vs llama 8-lane/3-stage/128-parallel/2-split

Verified terms:

- tinygrad `32-lane`: correct. The score dot reduces over the whole 32-lane warp.
- tinygrad `5-stage`: correct. `_warp_reduce_sum_staged` with width 32 runs offsets
  16, 8, 4, 2, 1 = 5 shuffle-and-add stages
  ([warp_reduce.py](/home/ubuntu/tinygrad-arkey/tinygrad/codegen/late/warp_reduce.py:105)).
- tinygrad `16-serial`: correct at d512. `token_in_tile = UOp.range(TK=16, ..., REDUCE)`
  serializes 16 tokens inside each warp, each carrying a full 32-lane reduce
  ([flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:258)).
  There is also a 16-iteration K/V staging loop (`STAGES = TK*Hd/THREADS = 16`) and, at
  d512, exactly one tile per split (`NB=1`), but the ledger's "16" is the token loop.
- tinygrad `48-split`: correct (`S=48`, G4).
- llama `8-lane`: correct. `nthreads_KQ = 128/16 = 8`, and the QK dot is
  `warp_reduce_sum<nthreads_KQ>` over an 8-lane group
  ([fattn-vec.cuh](/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh:87),
  [fattn-vec.cuh](/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh:274)).
- llama `3-stage`: correct. An 8-wide reduce runs offsets 4, 2, 1 = 3 stages.

Two terms need correction:

1. **`2-split` is the microbench pin, not llama's source-derived install.**
   [llama_fattn_vec_iso.cu](/home/ubuntu/tinygrad-arkey/extra/llm_research/microbench/llama_fattn_vec_iso.cu:14)
   hardcodes `grid(1, 2, 32)` with a `--gridy` sweep defaulting to 2. From the source,
   `launch_fattn` with decode (`Q->ne[1] == 1`, `stream_k=false`) computes
   `nbatch_fa = D = 128`, `ntiles_KV = ceil(512/128) = 4`, and the efficiency loop raises
   `parallel_blocks` to 4, so the real d512 grid is `(1, 4, 32)`. The trace record
   [nv-flash-score-llama-trace-20260813.md](/home/ubuntu/tinygrad-arkey/docs/task_workflow/input/nv-flash-score-llama-trace-20260813.md:86)
   also documents the 4-way split. So "2-split vs 48-split" is comparing a pinned
   microbench config to a production install; llama's installed split count is 4.
2. **`128-parallel` overstates the concurrency.** The block covers 128 KV columns per
   chunk, but only 16 dots are in flight at once (4 warps x 4 groups of 8 lanes); the
   `i_KQ_0` loop serializes the other 7 steps. A precise reading is "128 columns per
   chunk = 16 concurrent x 8 serial", not 128 simultaneous dots.

The durable contrast is therefore:

| axis | tinygrad tile | llama vec |
| --- | --- | --- |
| score reduce width | 32 lanes (whole warp), 5 stages | 8 lanes (group), 3 stages |
| column ownership | 16 tokens serialized per warp, one head per warp | 128 columns per chunk, 16 in flight, one head per block |
| K/V residency | LDS tile staging + 2 barriers/tile | direct global/L2 streaming |
| KV split | 48 global partials + combine | 4 partials + combine (microbench pin 2) |

## 3. Emitter changes required to emit llama's shape

These are changes to `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` and its
`FlashDecodeTileSpec`, classified by what blocks them today.

### 3.1 Blocked by the current legality gates

- **`score_group_width=8` is rejected by the gate.**
  [flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:168)
  raises unless `score_group_width == lane_width` (or `None`). llama needs
  `score_group_width=8` with `lane_width=32`. Removing the gate is not enough: the dot
  ownership is `elem = pair_axis*(LANES*dot_pair_width) + lane*dot_pair_width`
  ([flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:234)),
  which assumes every lane contributes `Hd/LANES` dims and that reducing below `lane_width`
  would sum only a fraction of the dot. Emitting 8-lane groups requires a subgroup
  ownership rewrite (`glane = lane & 7`, `group = lane >> 3`, `elem = (group*8+glane)*...`)
  and a per-group 8-wide reduce. The reduce primitive itself already exists
  (`_warp_reduce_sum_staged(val, lane, 8)`); the missing piece is the dot ownership, not
  a new shuffle op.
- **`warps >= query_group_size` is adjacent, but not the real fix.**
  [flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:163)
  forbids `warps < QG`. llama's mapping is not "fewer warps than heads"; it is "warps are
  orthogonal to heads". The current formula ties the warp index to head ownership
  ([flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:189))
  and cannot express "4 warps, all on one head, splitting columns".

### 3.2 Requiring new codegen / emitter support

- **Warp as a column partition instead of a Q-head owner.** Need a GLOBAL `head` axis per
  block plus a warp-derived column base (`warp*32`), with `kvh = head // G`. This is a new
  warp-role semantic; it cannot be reached by parameter values alone.
- **Register-resident Q.** Hoist the Q load out of `dot_reduce` (currently loaded at
  [flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:235))
  into a REG placeholder (`R=Hd/8=16` scalar halves per lane) and rebuild the half2 pair
  with `Ops.STACK` at the dot.
- **No-LDS K/V streaming.** `staging` only accepts `KV_BOTH`/`K_ONLY`
  ([flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:151)).
  A "none"/stream mode must remove `ksh`/`vsh`, the `CooperativeStageLaneMap` staging loop,
  and both per-tile barriers, and load K/V directly in the dot and PV phases.
- **128-column chunk loop.** Replace the `block` + `token_in_tile` nested REDUCE with a
  chunk loop over `NCHUNK` and an `NKQ=8` column REDUCE per group, with
  `col = split*128 + chunk*512 + warp*32 + group*8 + j`. This is the core structural
  rewrite and depends on 3.1.
- **Cross-group max and cross-warp combine.** After the 8-lane score reduce, unify the
  running max with 2 more stages (offsets 8,16); at chunk end reduce PV/den/max across
  warps through shared memory. `warp_reduce_max_across_groups` and
  `warp_reduce_sum_across_groups` already exist
  ([warp_reduce.py](/home/ubuntu/tinygrad-arkey/tinygrad/codegen/late/warp_reduce.py:80),
  [warp_reduce.py](/home/ubuntu/tinygrad-arkey/tinygrad/codegen/late/warp_reduce.py:94)).
- **Output ABI.** llama writes `dst` directly (or a 4-way `dst_meta` partial), while the
  tile always writes the 48-way global `pout` partial and a separate combine. Fusing the
  final reduction and shrinking the partial ABI is an emitter change beyond the gates.

### 3.3 Already legal, value-only changes

- `split_count=2` or `4` is a free parameter today; `FlashDecodeTileSpec` accepts any
  positive `split_count` and the emitter accepts any `S`. The restriction is route
  binding: `FLASH_DECODE_G4`/`FLASH_DECODE_G5` only admit `(Hq=32,S=48)` and
  `(Hq=40,S=32)`
  ([flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:907)).
  A new `FlashDecodeRouteConfig` (plus the `_s2/_s4` name suffix path) is needed to route
  it, not a new gate.
- `reduce_structure=staged` already matches llama's "materialize the shuffle into a REG
  before consuming it" style (`_staged_shfl`), and the cross-group helpers exist. No new
  shuffle primitive is required.

Note: the file already contains a faithful transcription of llama's shape as the
closed-default, not-routed `flash_vec_llama_score_pv_kernel`
([flash_decode_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/flash_decode_attention.py:445)),
fixed to `NKQ=8, LANES=32, WARPS=4, S=4`, with register Q, streamed K/V, bitwise lane
split, cross-group reduces, and a shared cross-warp combine. That proves the shape is
expressible in current UOps; it is not wired into a route, and the 08-13 audit measured
its transcription slower than both llama and the production tile, so shape legality is
not the only remaining distance.

## 4. Theoretical wall saving, and where the real gap is

The isolated body control is at parity. The matched measurement
[nv-llama-fattn-matched-isolated-record-20260816.md](/home/ubuntu/tinygrad-arkey/docs/task_workflow/input/nv-llama-fattn-matched-isolated-record-20260816.md:37):

- tinygrad production tile, S=48: 4.160-4.192 us isolated (400 replays).
- llama `flash_attn_ext_vec`, pinned grid.y=2: 4.096 us isolated (399 replays).

That is a ~0.09 us/launch difference, or ~3.2 us/token across 36 launches, roughly
0.1-0.15 tok/s at the ~206 tok/s / ~4855 us wall anchor. A shape rewrite that only changes
the score body therefore has essentially zero wall saving: the two bodies are within ~2%
under matched config.

The remaining installed gap is launch/graph/overlap, not score math:

- llama in-situ score is ~3.16 us/node (its graph/PDL install), which is faster than its
  own 4.10 us isolated body, i.e. llama hides flash work.
- tinygrad installed score is ~6.48-6.56 us (its isolated body is 4.19 us), i.e. tinygrad
  adds cold-L2 plus launch overhead on each of the 36 score launches and does not hide it.
- The latest side-by-side census puts the flash_score node_sum delta at +39.4 us/token
  (tinygrad 213.1 vs llama 173.6 us), the L3 ledger row's "39.4 at 1:1" ceiling
  (~+2 tok/s to ~208). The earlier CUPTI audit put the same class at +68.1 us in-situ.

Both estimates are installed-graph mass. Matching llama's 8-lane/3-stage/128-column shape
would only become a wall win if it also changes how the launch is installed and overlapped
(fewer/coarser launches, PDL semantics, co-scheduling with the GEMV chain). That is the
HIDE/launch axis, not the ELIMINATE shape axis, and it is exactly what the matched isolated
record and the 08-13 first-principles record conclude: the kernel body is flat, the win is
overlap.

One honest caveat on the body number: llama's source-derived d512 install is grid.y=4, whose
isolated body is 3.136 us, not the 4.096 us grid.y=2 pin. If the coarse 4-way split alone
transferred, the isolated body delta would be ~1.05 us/launch (~38 us/token). That is still
a body-to-body comparison at different split counts, and it does not change the conclusion
that the installed +39.4 to +68 us is dominated by launch/PDL/overlap behavior rather than
the score dot structure.

## Evidence

- tinygrad emitter: `tinygrad/llm/flash_decode_attention.py:142-280` (score tile),
  `:445-608` (llama-vec transcription), `:633-700` (spec defaults), `:907-910` (G4/G5 routes).
- tinygrad reduce: `tinygrad/codegen/late/warp_reduce.py:52-110`.
- llama kernel: `ggml/src/ggml-cuda/fattn-vec.cuh:21-530`; launch helper
  `fattn-common.cuh:970-1180`; `common.cuh:374-385` (`cpy_nb=16`), `:433-469`
  (`warp_reduce_sum`).
- microbench: `extra/llm_research/microbench/llama_fattn_vec_iso.cu`.
- ledger/records: `nv-fuse-hide-eliminate-ledger-20260818.md:28`,
  `nv-llama-fattn-matched-isolated-record-20260816.md`,
  `nv-flash-score-llama-trace-20260813.md`,
  `nv-flash-score-floor-test-head-20260816.md`.
