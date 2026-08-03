# Decode GEMV-efficiency forward scope - L2 partial / L4 vocab / flash substrate (ranked)

Date: 2026-08-03

Status: scoped, docs only. No code changes and no GPU use are authorized by this
document; the diagnostic microbenches below are the first action of each scope item,
run separately under their own implementation scope. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `fcf3774f9`.

This scope is the decode GEMV-efficiency work named by the current wall authority.
`nv-campaign-forward-review-amendment-20260803.md` section 4.1 item 5 requires that
this work be prioritized and NOT parked behind the withdrawn L1 forecast:

> Continue the decode GEMV-efficiency work named by
> `nv-decode-parity-final-20260802.md`; do not park it behind the old L1 forecast.
> L2 single-pass partial, L4 vocab substrate, and flash substrate remain separate
> scopes ranked by newly measured wall opportunity, not the superseded gross budget.

It supersedes none of the closed records. M3/M4/M5/Path 3 stay closed; the M2
`decode_epilogue_fusion` promotion record stays OPEN for `NV:sm_120` (Q6K down-coop
in-kernel merge); the L4 `row_tile=2` values row (commit `ab3cb84c1`) stays landed.
Every recovery number here is node-sum and is an upper bound until a same-session wall
measurement replaces it (section 7).

---

## 1. Why this scope exists - the measured wall authority

`nv-decode-parity-final-20260802.md` (M2-open baseline, same session, same model
Qwen3-8B-Q4_K_M, same RTX 5090 / sm_120) is the current wall authority:

| depth | tinygrad tok/s | llama tok/s | ratio | gap |
| --- | ---: | ---: | ---: | ---: |
| d512 | 172.80 | 248.20 +/- 7.37 | 69.6% | 1.44x |
| d2048 | 161.50 | 235.14 +/- 7.36 | 68.7% | 1.46x |
| d4096 | 149.00 | 225.95 +/- 6.71 | 65.9% | 1.52x |

Correctness pins hold at every depth: token sha256
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3, first token
`151936` 3/3. Census: 1021 kernels/token, 6187 / 6576 / 7091 us/token at
d512/d2048/d4096.

The parity record's verdict: the decode gap is inside the bandwidth-bound GEMV regime.
Per-token kernel count is flat with depth (1021) and the depth penalty is context-side
(KV reads), matching llama's own decay. The remaining 1.4-1.5x is the per-kernel GEMV
efficiency gap (llama `mul_mat_vec_q` dp4a vs our q4k/q6k GEMV family). The decode-norm
campaign (M3/M4/M5/Path 3) is closed non-landing, so the next evidenced wall lever is
decode GEMV efficiency. The old `0.9-1.0ms` / `1.07-1.21x` L1 arithmetic is withdrawn
by amendment section 2.4 and is not restated here.

`decode-gap-per-target-lever-scope-20260802.md` supersedes the lever list in
`nv-performance-campaign-scope-20260801.md` section 14.5 and already contains the
verdicts this scope builds on: L3 flash = SUBSTRATE (section 10), L4 values row landed
(section 10), L5 q4k lanemap = SUBSTRATE (section 11), L2 partial = SUBSTRATE
(section 11), M2 partial in-kernel merge non-landing (section 12). This document turns
those verdicts into separate named scope items, ranked by measured wall opportunity.

---

## 2. The ranked scope list

Ranking is by the measured wall opportunity behind each item: the node-sum recovery
upper bound per the current evidence, read as an upper bound only (section 7). Ordering
matches amendment section 4.1 item 5. The like-for-like cap discipline is item D: a
bounding rule for the GEMV-class claims, not a recovery item.

| rank | scope | class | node-sum recovery (upper bound) |
| --- | --- | --- | ---: |
| A | L2 Q6K partial single-pass | SUBSTRATE | ~0.25 ms |
| B | L4 vocab substrate fusion | SUBSTRATE | ~0.14-0.24 ms |
| C | flash score tile structure | SUBSTRATE | ~0.16 ms |
| D | like-for-like cap discipline (0.92 ms quantize-excluded) | discipline, bounds A + L5 | n/a |

Each of A-C is a separate named scope with its own diagnostic microbench, controls,
and go/no-go. They are separate commits, not one stack; no composed forecast is
produced until each item has an isolated same-session d512 wall measurement
(section 7).

---

## 3. Scope A - L2 Q6K partial single-pass (SUBSTRATE)

### Evidence

`q6k_gen_partial_1024_4096_4` (k/v projection on the 18 Q6_K layers, parts=4 packed
storage) measures **17.15 us at 0.20 TB/s** vs llama v kernel **3.3 us at 1.04 TB/s**
on the same 1024x4096 shape (decode-gap scope section 11; per-kernel table in campaign
scope section 14.1). That is 5x off llama on the same shape and ~11% of the bandwidth
ceiling. The generic `partial.sum(axis=1)` merge chain (2.09 us/layer) is part of the
measured r_ class and is structural to the partial route (decode-gap scope section 3).

The only free value knob on the partial spec is `opts`; the machine-search row
`LOCAL:0:32` (17.15 us) was already found optimal in the section 11 sweep. The route is
structurally bound: 4096 threads (rows x parts) each serially reduce 4 blocks x 16 pos;
llama's single-pass shape spreads the reduction over more threads.

### Mechanism class

**SUBSTRATE**, per the decode-gap scope section 11 verdict. The legal values rows are
exhausted; the gap is the thread decomposition of the reduction, which is one shared
emitter (`_emit_q6k_partial`) lowered for every target. The fix is a structural
single-pass variant of the same parts=4 storage: parts merged in-kernel, or the reduce
split across more threads. The landing mechanism is STRUCTURAL-additive (a new route
family, admitted per target, legacy `external_sum` route untouched), but the gap being
fixed is substrate.

### Diagnostic microbench (run first, under this scope's own implementation)

The `wmma_peak`-style method (`extra/llm_research/microbench/wmma_peak.cpp`,
`mma_peak_cuda.cu`): isolate the steady-state reduction loop with multiple independent
accumulators, hoist the operand setup out of the timed loop, keep loads out of the hot
loop where the structural knob allows, sweep the knob, and inspect the rendered source
to verify purity before believing a number.

For the partial single-pass shape the knob is the thread decomposition of the in-kernel
reduce on the fixed parts=4 storage (no load-time repack): threads per row x parts per
thread x blocks per part, covering at minimum the two recorded shapes (4-thread part
blocks; 8-row x 4-part 32-thread blocks) plus 16-row and split-reduce variants. The
go/no-go floor is llama-class: **~3.3 us / ~1.04 TB/s** on 1024x4096. If a legal
decomposition clears the floor, the winning row is per-target data on the new additive
route; if every decomposition stalls below the floor, the fix is deeper substrate
(access pattern / instruction mix) and stays shared-emitter work.

### The 466.6 us anomaly caveat (mandatory control in the microbench)

The M2 partial in-kernel merge attempt (decode-gap scope section 12) is a recorded
non-landing with a reproducible **466.6 us** in the real decode loop for the 8-row x
4-part 32-thread block shape (27x slower than legacy 17.15 + 2.09 us), with no
structural explanation found at source level; a standalone launch-config hypothesis
(2D block 4x8 vs 1D 32) was not the driver in the 4-thread control (25.3 us, a loss).
The 4-thread block shape is the documented nearest-working control. Scope A must
reproduce both recorded shapes in the microbench before trusting any new decomposition,
and must not reopen the `in_kernel` rejection on the existing partial spec
(`Q6KGEMVRouteSpec.validate` rejects it with a pointer to the M2 record); the single-pass
work is a new additive route family, not a change to the rejected shape.

### Per-target values row vs shared emitter

The emitter (`_emit_q6k_partial`) and the packed parts=4 storage are shared by every
target; there is no NV-only values row that fixes a 5x gap (the `opts` sweep already
found the installed row optimal). The single-pass variant is shared code with
per-target admission (route admission in `decode_routes.py`), so AMD's and Metal's
legacy partial routes stay byte-identical until each target admits the new family.

### Recovery and controls

Recovery: **~0.25 ms node-sum** (18 x (17.15 - 3.3) us), an upper bound. Controls:
pg3 decode render-equality hashes byte-identical on all 10 legacy rows (section 8.1);
NV pins (first-token digits, decode sha256 `0721c16f...`, bench census row); fixed-depth
token sha per harness. The like-for-like cap (section 6) bounds this item's combined
claims with L5.

---

## 4. Scope B - L4 vocab substrate fusion (SUBSTRATE)

### Evidence

The vocab head is three pieces today: `q6k_gen_coop_151936_4096` (397.2 us at row_tile=4,
1.29 TB/s = 72% of the 1792 GB/s ceiling), `q6k_vocab_scalar_reduce` (72.5 us), and the
generic scatter chain (~0.07 ms; `r_32_4_1187` 38.5 us plus the E_1187 companions), for
~0.54 ms total vs llama's single mmq vocab kernel at 303.75 us (campaign scope
section 14.1/14.4; decode-gap scope section 4 L4). The row_tile values hypothesis is
already falsified (151936 / 4 = 37,984 exactly), and the occupancy/vector-width branch
is bounded at ~93 us.

### The landed values row must stay

The L4 VALUES-ONLY piece already landed as per-target data at commit `ab3cb84c1`:
`Q6K_COOP_ROW_TILE_BY_TARGET = {("NV", "sm_120"): 2}` in `tinygrad/llm/decode_kernels.py`,
resolved at bind time from the installed primitive's TG3 admission capability. It moved
vocab coop 397.4 -> 330.1 us (1.55 TB/s, 86% of ceiling), down coop 49.7 -> 35.5 us, and
decode 163.5 -> 172.6 tok/s at d512 with every NV pin unchanged and the pg3 HIP hashes
byte-identical. Scope B does not touch that row; it is the baseline this scope builds
on.

### Mechanism class

**SUBSTRATE** for the fusion: the recoverable mass is the 72.5 us scalar reduce plus the
~70 us scatter chain (~0.14 ms), and removing them is a shared emitter change
(capability-gated, additive). The `reduction="in_kernel"` machinery already exists in
the coop emitter for the down path; at NV's row_tile=2, `row_tile * lane_extent = 32`
is legal under the single-warp constraint (AMD's row_tile=4 is not, and stays
`external_sum`). The vocab head currently keeps the scalar-reduce path by design
(M2 record, design Q9 / L4 boundary).

### Diagnostic microbench (run first, under this scope's own implementation)

Same `wmma_peak`-style method. Two stages:

1. Values ceiling check on the landed coop head: sweep the remaining occupancy/vector
   width surface at row_tile=2 with hoisted operand setup and accumulator staging,
   floor = llama mmq 303.75 us. This bounds the values-only residual and should confirm
   it is already saturated (the sweep is the go/no-go evidence, not an assumption).
2. Fused-shape probe: time the coop kernel with `reduction="in_kernel"` on the
   151936-row head (pos-lane ladder, single warp, legal at row_tile=2) plus the scatter
   epilogue absorbed, against the current 330.1 + 72.5 + ~70 us stack. The go/no-go:
   if the fused shape lands near llama-class on the same shape and removes the two
   separate kernels, the fusion is worth landing as a capability-gated SUBSTRATE
   variant.

### Per-target values row vs shared emitter

The value (`row_tile=2`) is per-target data and stays. The fusion is a shared emitter
change: one UOp builder, rendered per target, admitted per target. AMD keeps row_tile=4
`external_sum` and Metal keeps its admitted route; both must render byte-identical on
the legacy rows (pg3).

### Recovery and controls

Recovery: **~0.14-0.24 ms node-sum** (72.5 us scalar reduce + ~70 us scatter chain for
the 0.14 ms substrate floor; 0.24 ms is the decode-gap scope's combined values+substrate
upper bound, of which the values share is already landed). Upper bound only. Controls:
pg3 legacy hashes byte-identical; NV pins; the token stream is the strongest pin here
(the vocab head output selects the token), so first-token digits and decode sha256 are
mandatory.

---

## 5. Scope C - flash score tile structure (SUBSTRATE)

### Evidence

`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` measures 7.6 us x 36 vs
llama `flash_attn_ext_vec` 3.17 us x 36 (campaign scope section 14.1/14.4; decode-gap
scope section 4 L3). The score-kernel delta is 7.6 - 3.17 = 4.43 us x 36 = **~0.16 ms**
node-sum at d512, growing with depth as cache reads scale. Our own combine
(`flash_fused_gmax_combine`, 3.6 us x 36) is at parity with llama's and is not part of
the recoverable mass.

The L3 36-row values sweep (decode-gap scope section 10) found no win: QG in {1,2,4},
stage_width in {1,2,4,8}, split_size in {16,32,48} in-process on the G4 route, best row
7.41 us score (vs 7.59 baseline), tok/s spread 149.1-163.4. No row reaches the ~4 us
go/no-go bar.

### Mechanism class

**SUBSTRATE** (L3 verdict, decode-gap scope section 10): the whole-cache tile structure
itself is the gap, not the route values. It is the same tile AMD admits, so the
structure fix lifts every target that admits the route.

### Diagnostic microbench (run first, under this scope's own implementation)

Same `wmma_peak`-style method, applied to the tile structure instead of route values:
hoist the cache/score operand setup out of the timed loop, use independent accumulators
over the score computation, zero loads in the hot loop where the structural variant
allows, and sweep the tile geometry: LANES, WARPS (QG), TK, and the LDS staging shape of
the whole-cache tile (the `DECODE_STAGE_COALESCE` surface), at the campaign's measured
max_context=4608. Floor: llama-class ~3.2 us per score kernel. The 36-row values result
is the control that the sweep is structural: a geometry that clears the floor is
evidence the shared tile structure is the lever; no geometry reaching the floor closes
the values question permanently and the structural change must still render identically
across pg3 arms.

### Per-target values row vs shared emitter

The G4/G5 route config rows (`flash_decode_attention.py:472`) are per-target values data
and stay as the admission surface; the 36-row sweep already showed they are not the
lever. The whole-cache tile structure is one shared UOp builder
(`flash_decode_attention.py:92`), so the fix is shared code with per-target admission
(capability-from-renderer-facts AND target-promotion).

### Recovery and controls

Recovery: **~0.16 ms node-sum at d512**, growing with depth; upper bound only. Controls:
pg3 legacy hashes byte-identical; NV pins; fixed-depth token sha per harness (the
36-config sweep kept `5662f1cd...` at nmeas=40 and the decode-gap section 11 harness kept
`9d6b3787ce...` at nmeas=20/reps=3; the sha is per-harness).

---

## 6. Scope D - like-for-like cap discipline (bounds A and L5)

The GEMV (non-vocab, incl quantize) class comparison hides an asymmetry: llama's 3.72 ms
includes its `quantize_q8_1` kernels (0.482 ms across 217 nodes), which tinygrad does not
pay. The like-for-like comparison excludes quantize from both sides: llama's bare GEMV
class is 3.543 - 0.304 (vocab) = **3.24 ms** vs ours 4.16 ms, i.e. **0.92 ms of
headroom** (decode-gap scope section 8.1). That is the cap that matters for the combined
L2 + L5 claims; tinygrad consuming packed storage directly is the point of the fused
storage, not an arithmetic violation.

Scope A (partial single-pass, ~0.25 ms) plus any future L5 structural work must stay
under the 0.92 ms quantize-excluded cap when their claims are combined. The settling
check is a measured quantize-excluded comparison: a DEBUG=2 trace of our GEMV class and
a node-filtered llama trace excluding `quantize_q8_1`, run on the 5090. This scope does
not claim L5 mass: the L5 lanemap lanes sweep was a SUBSTRATE verdict (flat at
lanes=32/64/128, decode-gap scope section 11) and there is no live values-only mass in
the q4k class beyond the already-saturated coop-down residual.

---

## 7. Endpoint discipline

- **No composed node-sum stack forecast.** Amendment section 4.1 item 6: no node-sum
  stack or wall forecast is published until components have isolated same-session
  measurements. This document states only per-item node-sum upper bounds and explicitly
  does not add them.
- **Each item gets an isolated same-session d512 wall measurement before any
  composition.** The measurement protocol is the parity record's: fixed-depth decode at
  d512 (then d2048/d4096), same-session llama `tg10 @ d`, 5 reps median, busy baseline
  recorded per item (decode-gap scope section 5.3).
- **Recovery numbers are node-sum upper bounds.** Node-sum over-counts replay wall by
  ~8% and is licensed for relative class attribution, not for sizing wall levers
  (decode-gap scope section 1). Recovery numbers are also not additive across items.
- The old `0.9-1.0ms` / `1.07-1.21x` L1 arithmetic is withdrawn and must not re-enter
  any endpoint claim (amendment section 2.4).

---

## 8. Controls

### 8.1 pg3 decode render-equality (existing pin table, decode-gap scope section 5.1)

Pinned with the house convention (`sha256((src + "\n").encode())`, first 12 hex) on
HIPRenderer gfx1100, render-only. Any VALUES-ONLY lever must leave these byte-identical;
any SUBSTRATE lever must change the AMD and Metal arms in the same way; any
STRUCTURAL-additive lever must keep the AMD- and Metal-admitted routes' hashes
unchanged.

| kernel | sha256 |
| --- | --- |
| q4k_g3_lanemap_gemv_12288_4096 | 312422c73a49 |
| q4k_g3_lanemap_gemv_4096_4096 | 27857cb8ca03 |
| q4k_g3_lanemap_gemv_4096_12288 | 851760e2053c |
| q4k_g3_lanemap_gemv_1024_4096 | 39ddb717ddd4 |
| q6k_gen_coop_4096_12288 | cc38fbb3db92 |
| q6k_gen_coop_151936_4096 | 5795e66a7292 |
| q6k_gen_partial_1024_4096_4 | 344e1c388eeb |
| q6k_vocab_scalar_reduce_151936_4096 | c708302aa2d2 |
| flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128 | 66d4c4da3108 |
| flash_fused_gmax_combine_32_128 | c78e4651ad35 |

The M2 promoted fused row is additionally pinned: `q6k_gen_coop_4096_12288_inkernel` =
`add50a7aa43f` (src_len 9440, ds_bpermute=4); it is the baseline's fused state and must
stay byte-identical.

### 8.2 NV pins (existing, re-run per item)

- First-token digits `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655,
  12080, 25, 576, 35264, 5624]`.
- Decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`.
- Bench census row `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.
- Fixed-depth token sha per harness (per-harness, e.g. `9d6b3787ce...` at nmeas=20
  reps=3, `5662f1cd...` at nmeas=40); every legal sweep row must keep it.

### 8.3 AMD/Metal render-equality requirement for SUBSTRATE fixes

A SUBSTRATE fix is proven generic by the pg3 render arms moving together (AMD HIP arm +
Metal arm, render-only; the Metal arm runs on the macOS box and gets its own pinned
block when it first runs), never by assertion. The CUDA arm additionally renders
compile-only in the M2 unit gate (`test_decode_epilogue_fusion_gate.py` renders the
fused kernel through HIPRenderer and CUDARenderer). A measured AMD runtime number is the
promotion gate later, not the landing gate here (decode-gap scope section 5.3).

---

## 9. Delivery and bans

This document is docs-only. Each scope item (A-C) is implemented and measured separately
under its own implementation scope with the diagnostic microbench first, its own
settling command, legacy hash controls, correctness pins, and fixed-depth wall gate, and
is a separate commit on `nvidia-bringup-20260731` only. No promotion to
`dev`/`exp`/`master`. Never touch the untracked scratchpads
(`extra/llm_research/microbench/dp4a_peak_cuda*`, `scratchpad/t6_metal_admission_probe.py`).
No push; the parent pushes after review.

---

HARD STOP after this section. Nothing beyond this scope without review. The next
implementation requires a separate, variant-specific scope with its settling command,
legacy hash controls, correctness pins, and fixed-depth wall gate.
