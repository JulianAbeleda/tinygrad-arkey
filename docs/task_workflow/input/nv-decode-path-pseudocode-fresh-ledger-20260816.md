# NV decode: our path vs llama's path, pseudo-code + fresh ledger (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731`
Measurement baseline: `86d653651`
Initial document commit: `d44d53378`
Status: **design-reviewed research document. Documentation only; no runtime code
change.**
Pseudo-code is traced from the actual sources on both sides; every number in
the ledger was re-measured at HEAD this session (census) or is quoted from the
pinned llama CUPTI ledger.

Review verdict: the score-kernel mapping is a credible lead, but the evidence
does not support framing this as "single-pass llama versus two-pass tinygrad."
Both implementations produce split partials and launch a combine kernel. The
measured differences are the split count, score-body mapping, and installed
launch behavior. Compiler-level claims about Q loads also remain hypotheses
until checked in generated SASS.

## 0. The question this document answers

What does llama's decode kernel actually do, step by step, and what does ours
do? Where exactly is the flash score slower, and is the "single-pass" structure
itself the win, or is something else? The ledger at the end is the fresh
exhaustive view at HEAD so a reviewer can see every class, its measured cost,
its llama reference, and its open/closed status.

## 1. Shared contract (both sides compute the same math)

Decode attention at context `Tc` for one new token, `Hq=32` query heads,
`Hkv=8` KV heads (GQA 4), `Hd=128`, fp16 K/V. Each head's output is
`softmax(Q K^T / sqrt(Hd)) V`. Both sides keep a running max and denominator
(online softmax) so the KV cache can be streamed in chunks. That arithmetic is
identical; the difference is only how it is mapped to threads, memory, and
kernels.

## 2. llama's path (fused score/PV per split + combine)

Source: `ggml-cuda/fattn-vec.cuh` `flash_attn_ext_vec`, HEAD `ac4cddeb0`.
The pinned CUPTI ledger records grid `(1, 2, 32)` = 64 blocks, block
`(32, 4, 1)` = 128 threads. `gridDim.y = 2` assigns two interleaved KV streams;
each block visits two 128-token chunks at `Tc=512`. `gridDim.z = 32` selects
the query head. This measured launch is authoritative for the comparison;
`grid.y=4` in the initial draft was incorrect.

```c
kernel flash_attn_ext_vec(Q[32][128], K[8][512][128], V[8][512][128], scale):
  head = blockIdx.z
  // ---- prologue: Q into registers, ONCE, scaled once ----
  Q_reg[8] = load_half2(Q[head][lane_dim*8 + ...])     // 8 half2 per lane
  Q_reg *= scale                                       // scaled once
  K += blockIdx.y * 128; V += blockIdx.y * 128         // first chunk for my split

  // ---- the whole attention in one KV loop ----
  for k_VKQ_0 = blockIdx.y*128; k_VKQ_0 < 512; k_VKQ_0 += 2*128:
    KQ_max_new = KQ_max
    // score all 128 columns in parallel: 4 warps x 4 groups x 8 lanes
    for i_KQ in my_8_of_128:
      sum = vec_dot_KQ(K[i_KQ], Q_reg)                 // half2 FMA, no TC
      sum = warp_reduce_sum<8>(sum)                    // 3 shfl stages
      KQ_max_new = max(KQ_max_new, sum)
    // online softmax, registers only
    KQ_max = shfl_xor_reduce(KQ_max_new)               // 2 more stages
    KQ_reg = exp(KQ_reg - KQ_max)
    KQ_sum = KQ_sum * exp(KQ_max_old - KQ_max) + KQ_reg
    KQ[group] = KQ_reg                                 // one shared write
    // PV accumulate, registers
    VKQ *= exp(KQ_max_old - KQ_max)
    VKQ += V[k] * KQ[k]                                // per-lane dims

  // ---- epilogue: cross-warp reduce + write final output ----
  // shared reduce of KQ_max/KQ_sum over 4 warps
  // dst = VKQ / KQ_sum  (directly, per head per slice)
  dst_meta[head][blockIdx.y] = {KQ_max, KQ_sum}        // for 2-way combine

kernel flash_attn_combine_results(VKQ_parts, VKQ_meta, dst):
  // merges gridDim.y=2 partials: exp(meta.x - max) weighted sum / sum
```

Structural facts of llama's path:
- **Two kernel stages.** Score, online softmax, and PV are fused while visiting
  each split's KV once; the second kernel merges two split partials.
- **Q is register-resident.** Loaded once into `Q_reg`, scaled once.
- **K/V are not tile-staged in shared memory.** The kernel still uses shared
  `KQ` scratch (8,448 static bytes in the pinned trace), one `__syncwarp` per
  chunk, and synchronization in its reduction/epilogue path. The narrower
  claim is what distinguishes it from tinygrad's K/V staging.
- **8-lane dots.** Each warp scores 32 columns in parallel via 4 groups x 8
  lanes; the reduce is 3 shuffle stages, and all 128 columns of a chunk are in
  flight at once.
- Measured in-situ: **3.16 us/node** (CUPTI full-graph, 36 nodes = 113.9 us).

## 3. tinygrad's path (tiled score/PV per split + combine)

Source: `tinygrad/llm/flash_decode_attention.py:92-249`. Production d512 is
G4: `S=48` splits, `LANES=32, WARPS=4, THREADS=128, TK=16, R=4, RP=2`,
`stage_width=1`. Grid `kvh(8) x split(48) x query_group(1)` = 384 blocks.

```python
# ---- kernel 1: score + sharded PV ----
kernel flash_block_tiled_xlane_score_pv_tile_whole_cache(pout, q, cache, Tc):
  for kvh, split, query_group:                       # grid 8 x 48 x 1
    acc[R] = 0; den = 0; mx = -inf                   # per-lane registers
    for block_tile in range(ceil(L/16)):             # NB tiles
      # stage K AND V for this 16-token tile into LDS
      ksh, vsh = cooperative_load(cache[kvh, split*L + block*16 : +16])
      barrier()                                       # barrier 1
      for token_in_tile in range(16):                 # SERIALIZED tokens
        qpair = load q[head*128 + elem]               # Q re-read from global
        dot = fdot2(qpair, ksh[token])                # RP=2 half2 per lane
        score = warp_reduce_sum_32lane(dot)           # 5 shfl stages, full 32
        score *= scale
        new_max = max(mx, score)
        correction = exp(old_max - new_max)
        acc = acc * correction + exp(score-new_max) * vsh[token]
        den = den * correction + exp(score-new_max)
        mx = new_max
      barrier()                                       # barrier 2
    # write 48-way partial to global
    pout[head*48 + split] = {acc, den, mx}

# ---- kernel 2: combine the 48 splits ----
kernel flash_fused_gmax_combine_f16(out, pout):
  for head:
    global_max = max over 48 splits of pout[..].mx    # reduce
    weight[s] = exp(pout[s].mx - global_max)
    out = sum_s(weight[s] * pout[s].acc) / sum_s(weight[s] * pout[s].den)
```

Structural facts of tinygrad's path:
- **Two kernel stages.** The score/PV kernel writes 48 partials to global and
  the combine kernel reads and normalizes them. This is the same high-level
  topology as llama, but with 24x more split partials in the measured launch.
- **Q loads are expressed inside the per-token reduction**
  (`q[head*Hd+elem]`). That is a source/UOp fact, not proof of one hardware LDG
  per token: invariant-load hoisting and cache behavior must be checked in
  generated SASS and profiler counters.
- **K/V staged to LDS per 16-token tile** with two barriers per tile.
- **32-lane full reduce** (5 shuffle stages) and the 16 tokens of a tile are
  reduced serially: 16 x 5 = 80 serialized shuffle stages per warp per tile,
  versus llama's 128 columns in parallel with a 3-stage 8-lane reduce.
- Measured isolated (CUPTI, HEAD, this session): **4.19 us median**. The llama
  3.16 us number is in-situ, so subtracting the two does not isolate a body
  delta. Historical tinygrad in-situ (08-13) was 5.06 us; installed NV in-loop
  at the fresh baseline is 6.56 us x 36 = 241.4 us.

## 4. Structural delta (one axis at a time)

| axis | llama | tinygrad | supported inference |
| --- | --- | --- | --- |
| kernel stages | split score/PV + combine | split score/PV + combine | topology is shared; combine rows are at parity |
| Q expression | explicit register preload | load expression inside token loop | possible hoist opportunity; hardware loads unproven |
| K/V staging | no K/V tile staging; shared KQ scratch | LDS-staged K/V, 2 barriers/tile | staging/barrier cost is plausible, not isolated |
| dot width | 8-lane group, 3 shfl, 128 cols parallel | 32-lane, 5 shfl, 16 tokens serial | serialization |
| KV split | 2 measured | 48 | 6x score-block count; combine cost itself is not observed |
| output | split partials to global, then combine | split partials to global, then combine | same ABI shape, different partial count |

The 08-13 trace (`nv-flash-score-llama-trace-20260813.md`) and the 08-03
structural sweep show that tested tile geometries and staging widths did not
close the gap. They do not establish a single cause because split count, Q
placement, staging, and reduction width were not isolated in matched kernels.

## 5. Fresh floor measurements at HEAD (this session)

Re-run at `86d653651` (record: `nv-flash-score-floor-test-head-20260816.md`):

| measure | value |
| --- | ---: |
| NV census score row | 6.56 us x 36 = 241.4 us (unchanged vs prior pin 6.52) |
| NV census combine row | 3.39 us x 36 = 122.8 us (unchanged) |
| isolated tile body (CUPTI) | 4.19 us median (unchanged from 08-13) |
| llama in-situ | 3.16 us (not an isolated-body measurement) |
| warm microbench peak (production shape) | 1.95 us (not reachable in graph) |
| cold single-launch microbench | 7.6 us |

The "~90 us structural floor" that the old ledger attributed to flash is a warm
microbench artifact, not an installed-graph number (cold per-launch is 7.6 us).
The fresh installed score-row gap is `(6.56 - 3.16) * 36 ~= 122 us`, but that
mixes body, launch, and graph-install effects. The isolated tinygrad body is
4.19 us; there is no matched isolated llama body in this evidence set. The old
`(5.06 - 3.16) * 36 ~= 68 us` result is historical, not the fresh HEAD result.

## 6. Fresh ledger crosswalk at the measurement baseline

llama class totals are from the pinned CUPTI ledger
(`nv-llama-d512-node-ledger-20260812.json`); the per-shape llama floors are the
same-session fresh llama numbers from
`nv-decode-gap-attribution-same-session-20260812.md` section 4; tinygrad rows
are the HEAD census (`/tmp/census_q6kv_promoted_head.json`, DEV=NV, d512, prime
token, same harness). Because the per-shape floors and class totals are
independently measured/medianed, this table is a crosswalk, not an additive
decomposition. `*` = class boundary differs between sides (see notes).

| tinygrad row (per-kernel) | count | us | llama shape floor | llama us | delta | status |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| q4k gate/up w1w3fused16 | 36 | 1407.4 | gate/up 37.86 x 36 | 1363.0 | +44 | parity |
| q4k attn-O epi_resadd | 36 | 365.8 | attn-O 11.78 x 36 | 424.1 | -58 | **better** |
| q4k FFN-down fp16 mmvq | 18 | 408.2 | ffn-down Q4 11.78 x 18 | 212.0 | +196 | open* |
| q4k attn-Q 4096 (2 routes) | 36 | 343.0 | attn-Q 9.54 x 36 | 343.4 | -0.4 | parity |
| q4k attn-K/V 1024 (2 routes) | 54 | 235.6 | K/V Q4 3.33 x 54 | 179.8 | +56 | open* |
| q6k FFN-down fp16 mmvq | 18 | 576.3 | ffn-down Q6 28.75 x 18 | 517.5 | +59 | open* |
| q6k attn-V (2 routes) | 18 | 84.8 | attn-V Q6 4.90 x 18 | 88.2 | -3 | parity (landed) |
| vocab 151936 GEMV | 1 | 331.3 | vocab | 303.6 | +28 | parity |
| flash score | 36 | 241.4 | flash_score | 113.9 | +127 | open (body +37) |
| flash combine | 36 | 122.8 | flash_combine | 120.5 | +2 | parity |
| reduce-output rmsnorm (3 shapes) | 91 | 382.9 | absorbed in-kernel | 0 | +383 | open* |
| rmsnorm q8_1 provider | 17 | 45.4 | rms_norm | 307.6 | -262 | **better** (fused) |
| M1 chains + E/r plumbing + scatter | 248 | 538.0 | elementwise/get_rows | 4.8 | +533 | open* |
| rope / kv-store | 0 | 0 | rope + kv_set_rows | 201.0 | -201 | **better** (fused) |
| quantize q8_1 | 0 | 0 | quantize_q8_1 | 482.2 | -482 | absorbed in tinygrad GEMV* |
| llama MMQ crosswalk residual | - | - | class total minus displayed shape floors | 111.0 | - | measurement-basis residual* |
| **independently medianed node sums** | 596 | **5082.7** | 762 llama nodes | **4774.4** | **+308.3** | not the sum of rows |

Notes on class boundaries (`*`):
- tinygrad folds quant into the GEMV (llama runs a separate `quantize_q8_1`,
  482.2 us). The explicit row above prevents that credit from disappearing,
  but it cannot be assigned honestly among individual GEMV shapes.
- The displayed llama shape floors total about 111 us less than the pinned
  CUPTI `mmq.node_sum_us=3542.555`. That residual is shown explicitly rather
  than silently reconciling a shape-floor table to a class-total footer.
- llama's norms/rope/kv are separate kernels that overlap its mmq anchor;
  tinygrad fuses them into GEMV epilogues. On node-sum they cancel
  (-262 -201 vs the fused-in cost), but the overlap story is different: llama
  hides them, tinygrad serializes them into the critical path.
- llama "absorbed in-kernel" rows are inside its mmq node sum already. Per-row
  deltas therefore must not be summed. The source ledger also warns that
  profiled intervals are not an unprofiled token-wall decomposition.

## 7. What is actually open (measured), and what is closed

Closed (no mechanism, per prior records, unchanged at HEAD):
- flash tile geometry sweep: NO-GO (08-03)
- 512-thread single-stage combine: NO-GO (08-05)
- llama-vec single-pass as-is: 10.2 us NCHUNK=2, slower than the 4.19 us tile
  (08-13, flash source unchanged since)
- multi-stream / anchor-shadow overlap: FLAT (08-15); native channel blocked
- Q4 FFN-down DP4A/Q8 load patterns: NO-GO (08-12)

Open (real measured excess, mechanism not yet proven):
- reduce-output epilogue rows (~383 us total): llama absorbs in-kernel, we run
  a separate reduce. Prior body-free fold attempts measured FLAT, so the
  mechanism (absorb into GEMV epilogue without adding body) is unproven.
- M1 norm chains + E/r plumbing (~367 us): llama overlaps; we serialize.
  Body-free fold FLAT, body-adding NO-GO.
- flash score: +122 us in the fresh installed row. Body-only attribution is
  unresolved because 4.19 us isolated tinygrad and 3.16 us in-situ llama are
  not matched measurements. The existing vec candidate is slower as-is.
- q4k/q6k FFN-down and q4k 1024 rows: ~+300 us vs llama before quant credit;
  llama's mmq absorb plus quant folding makes this smaller than it looks.

## 8. The honest bottom line

- The fresh flash score-row excess is ~122 us, but its recoverable body mass is
  unknown until both kernels are measured in the same isolated and installed
  harnesses. The combine row is already at parity (122.8 vs 120.5 us total),
  so removing the combine ABI is not supported as the first target.
- The biggest measured excess is the reduce-output + M1/E-r plumbing family
  (~750 us of rows llama either absorbs or hides). Fusion has measured FLAT on
  body-free folds; overlap has measured FLAT on our DAG.
- The independently medianed node-sum gap is ~+308 us (5082.7 vs 4774.4), but
  class-boundary differences (quant credit, absorbed reduces, hidden norms)
  make per-row attribution non-additive; the wall gap is 4868 vs 4036-4074 us.

## 9. Design review: refactor direction and acceptance gates

### Findings

1. **Do not design around "one pass versus two passes."** Both paths have a
   split score/PV kernel and a combine kernel. The combine rows are at parity,
   so the score kernel is the justified flash target.
2. **Do not treat source placement as hardware traffic.** The tinygrad Q load
   appears inside the token loop, but only SASS and load counters can establish
   whether it is repeatedly issued. An explicit register preload is useful only
   if that gate fails.
3. **Do not transplant all llama choices at once.** The prior vec attempt
   changed several axes and regressed. Split count, Q residency, K/V staging,
   and reduction width need one-axis variants against the same baseline.
4. **Keep backend semantics separate from NV scheduling.** Preserve the current
   semantic kernel/fallback for AMD and Metal. Put any 8-lane grouped-dot or
   register-preload schedule behind an NV capability/policy boundary rather
   than changing cross-backend render equality by accident.
5. **Do not optimize from additive row deltas.** Rebuild the crosswalk from one
   metric basis before using it to rank non-flash work. Use wall/union exposure
   for recoverable latency and node sums only for kernel-work accounting.

### Recommended sequence

1. Add a matched harness that records isolated and installed timings for both
   score kernels at the same `Tc`, split count, warmup, and profiler mode.
2. Capture tinygrad SASS plus global-load transactions for Q. If repeated Q
   loads are present, refactor a pre-loop `AddrSpace.REG` Q fragment while
   leaving staging, split count, and reduction width unchanged.
3. Sweep split count with the existing tiled body unchanged. This isolates
   block-count/installed-launch effects before any vec rewrite.
4. Prototype the 8-lane grouped dot as a target-specific scheduling policy with
   the existing partial-output ABI. Do not couple it to combine removal.
5. Only after those gates, test K-only/no-KV-staging variants. Accept a change
   on installed graph latency and numerical parity, not warm microbench peak.
6. Rebuild the full ledger with explicit rows for llama quantization and any
   crosswalk residual, then rank reduce/M1 work by exposed wall time.

## Evidence

- llama source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`,
  `fattn-common.cuh:913-967`, `fattn.cu:250` (HEAD `ac4cddeb0`)
- tinygrad source: `tinygrad/llm/flash_decode_attention.py:92-249`
- census at HEAD: `/tmp/census_q6kv_promoted_head.json`
- llama ledger: `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
- fresh floor test: `docs/task_workflow/evidence/nv-flash-score-floor-test-head-20260816.json`
