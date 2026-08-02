# NV decode gap - per-target lever scope (review draft)

Date: 2026-08-02

Status: scoped, not implemented. Supersedes the lever list in
`nv-performance-campaign-scope-20260801.md` section 14.5, which over-applied the AMD
validation guardrail (see section 2 here for the retraction). Branch boundary:
tinygrad `nvidia-bringup-20260731`. Does not authorize promotion to `dev`/`master`.

Bans for this scope: no `prefill_routes.py`, no dtype cleanup, no commits to
`master`/`dev`/`exp`, and never commit the untracked scratchpads
(`scratchpad/t6_metal_admission_probe.py`, `extra/llm_research/microbench/dp4a_peak_cuda*`).

---

## 1. Why - the measurement that motivates this

Same-session d512 fixed-depth decode, Qwen3-8B-Q4_K_M, RTX 5090, llama `ac4cddeb0`
(campaign doc section 12/14, `/tmp/qwen3-8b-nv-p4-decode.json`,
`/tmp/llama_tg10_node.sqlite`):

| depth | tinygrad tok/s | tinygrad ms/token | llama tok/s | gap |
| --- | ---: | ---: | ---: | ---: |
| d512 | 163.4 | 6.12 | 245.6 | 1.50x |
| d2048 | 153.5 | 6.52 | 234.8 | 1.53x |
| d4096 | 142.7 | 7.01 | 225.1 | 1.58x |

Decode is GPU/kernel-bound, not host-bound: 5.83ms GPU busy of 6.12ms wall (95%), and the
flash-decode rollout is already graph-replayed into 6 batches (`batched 32/64/128/256/512/29`
= 1021 programs/token). Per-token node-sum attribution of the 1.84ms gap (campaign doc 14.3):

| class | tinygrad | llama | delta |
| --- | ---: | ---: | ---: |
| GEMV (non-vocab, incl quantize) | 4.16 ms | 3.72 ms | +0.44 ms |
| flash attention | 0.40 ms | 0.234 ms | +0.17 ms |
| norms / rope / adds / aux | 1.56 ms | 0.51 ms | +1.05 ms |
| vocab head | 0.54 ms | 0.304 ms | +0.24 ms |
| node-sum total | 6.61 ms | 4.77 ms | +1.84 ms |

**Evidence class warning (review, 2026-08-02, corrected 2026-08-02).** Node-sum is not wall
time and must not be substituted for it. For tinygrad, prime node-sum 6.61 ms > replay wall
6.12 ms > replay busy 5.83 ms (prime node-sum over-counts replay wall by ~8%); for llama,
node-sum 4.77 ms vs same-session wall 4.44 ms at 225 tok/s (over-count ~7.5%). The earlier
17% figure for llama mixed sessions: 4.07 ms is P4's wall at 245.6 tok/s, a different run
than the tg-trace session that produced node-sum 4.77. A serialized decode cannot contain
more kernel-time than wall-time, so node-sum over-counts by roughly the same ~8% on both
sides, and the class-level relative attribution is preserved. What is NOT licensed: the
"0.21 ms unattributed" claim (2.05 - 1.84) subtracts a P4 same-session wall gap from a
cross-session node-sum delta; both numbers are real, but they are different instruments and
the difference is not attributable. The P4 paired wall gap (6.12 - 4.07 = 2.05 ms) stands on
its own as the same-session gap.

Per `structure/Development/coding-principles.md` ("Classify Evidence Before Fixing
Mechanisms"), node-sum is licensed to prove *relative attribution between kernel classes*. It
is not licensed to size a lever in wall-time milliseconds. Every recovery number in section 4
is stated in node-sum and must be read that way until re-measured against wall.

llama's tg graph is 762 nodes/token; tinygrad's is 1021. llama fuses the add/silu/norm
plumbing into GEMV epilogues (its graph has no separate add or silu kernels), fuses w1+w3 into
one 12288-row GEMV, and runs Q6_K decode kernels at 1.4 TB/s vs tinygrad's 0.82 TB/s (coop)
and 0.2 TB/s (partial). Full per-kernel tables are in campaign doc sections 14.1-14.4; this
scope does not repeat them.

---

## 2. The design correction - shared emitter, per-target values

Campaign doc 14.5 said every lever is "shared AMD+NV code" and therefore needs a measured AMD
runtime number before landing. That conflated two different things:

1. The **emitter** is shared: `q4k_g3_lanemap_gemv_kernel`,
   `emit_q6k_gemv_kernel`, and `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` are
   each one UOp builder in tinygrad (`decode_kernels.py:110`, `flash_decode_attention.py:92`).
   There is no hand-written CUDA copy and no hand-written HIP copy; each target's renderer
   compiles the same graph.
2. The **values** are already per-target data, not shared. The repo already has the full
   mechanism:
   - `DeviceCapabilities` declared facts (`device_facts.py:44`): wave_size,
     supports_warp_shfl_xor, supports_tensor_cores, supports_fp16, lds_bytes, read from the
     opened renderer, never inferred from a target string.
   - Renderer-declared facts consumed generically (`renderer/__init__.py`:
     `lds_bank_dwords`, `lds_read_before_next_write_ordered`; postrange reads them through the
     renderer, an undeclared target gets the safe default automatically).
   - Per-target route admission: decode candidates take `arch_ok`
     (`decode_routes.py:62,101`), flash admission is capability-from-renderer-facts AND
     target-promotion (`flash_decode_attention.py:417`, `FlashDecodeAdmission`), and
     `FlashDecodeRouteConfig` G4/G5 are per-target config objects
     (`flash_decode_attention.py:472`).
   - `Q6KGEMVRouteSpec` carries `target`, `row_tile`, `parts`, `lane_extent` as data
     (`decode_kernels.py:163`). The `target: str = "amd_gfx1100"` defaults are provenance
     from the AMD machine search, not a hard requirement that every target share AMD's values.

**Retraction.** Section 14.5's blanket "every lever needs AMD runtime measurement" is
withdrawn. AMD needs a runtime measurement only if AMD's admitted values or admitted route
actually change. A lever that tunes NV values through the per-target data mechanism leaves
AMD's generated source byte-identical, which is exactly what the render-equality control
proves.

The principle this scope follows, unchanged from the repo's established pattern: tuning knobs
are per-target **data** (fact rows, route config rows), never per-target **branches** inside
shared emitters; an undeclared target keeps the safe default; a structural change lands as an
additive route variant with per-target admission, leaving the legacy route untouched.

**Third category - substrate.** A gap is SUBSTRATE-shaped when the shared kernel sits far
below a hardware-agnostic ceiling: per-kernel bandwidth, kernel count, occupancy, and access
pattern are substrate, not values. The partial route at 0.2 TB/s (5x below llama on the same
shape) and the 695-kernel plumbing chain (llama has no separate add/silu kernels) are
substrate-shaped on their face; they are not NV tuning problems. Fixing substrate lifts every
target that admits the route, because there is one emitter and one lowering. NV is simply
where we can measure it. The criterion for SUBSTRATE is "if the math allows it, all targets
gain": a substrate lever is proven generic by the AMD **and** Metal render arms moving
together (section 5.1), never by assertion. Values-shaped gaps ("this default row was tuned
on gfx1100") stay fact rows. Some levers are both: the emitter generalization is substrate,
the numeric row for the given target is a value (L2's staging, L5's lanes).

So the per-lever class is decided by measurement, not assumption: each lever opens with a
diagnostic microbench (the `wmma_peak`-style method: multiple accumulators, operands hoisted,
zero loads in the loop, sweep the knob) that names the class, then a go/no-go: if the knob
sweep clears the llama-class floor, the gap is values and the fact row is the answer; if the
sweep stalls well below the floor, the gap is substrate and the shared emitter/lowering is
the fix, for all three targets.

---

## 3. Established state (checked, with refs)

- Q4K decode candidate: `_Q4KDecodeCandidate` (`decode_routes.py:43`), executes
  `q4k_g3_lanemap_gemv_kernel(binding.N, binding.K)` with `lanes: int = WARP` fixed at 32
  (`decode_kernels.py:110`). Kernel name `q4k_g3_lanemap_gemv_<rows>_<k>`, output fp32 `(N,)`.
- Q6K decode candidate: `_Q6KDecodeCandidate` (`decode_routes.py:93`), `row_tile=4`; `parts`
  comes from the loaded weight storage (`linear.parts`), `use_coop = parts == 1 and
  out_features % row_tile == 0`; non-coop path emits `partial (N, parts)` then a **generic**
  `partial.sum(axis=1)` reduce (`decode_routes.py:113-131`). The generic reduce is part of the
  measured r_ class, so part of the "plumbing" cost is structural to the partial route.
- Flash decode: G4 `split_size=48, query_group_size=None, stage_width=1`; G5
  `split_size=32, query_group_size=2, stage_width=4` (`flash_decode_attention.py:472`); the
  block tile lowers LANES=32, WARPS=QG, TK=16, THREADS=32*QG, whole-cache tile with LDS
  staging (`flash_decode_attention.py:97-125`). `DECODE_STAGE_COALESCE` already selects the
  staging width when the spec leaves it None.
- Plumbing kernels (E_ 510 + r_ 185 per token) are ordinary JIT-lowered elementwise/reduce
  programs. The census diff between the flash and SDPA decode graphs shows the same E_/r_ set
  in both (campaign doc 14.4), so they are model plumbing, not flash-specific; the 18-count
  classes (r_8_8_16_2_4, r_32_32_4_2_8, E_8_8_16_2) correlate exactly with the 18 Q6_K
  layers and are the partial-route merge chain.
- Controls today: pg2 (`scratchpad/pg2_amd_all_routes_rendered_source_equality.py`) pins
  AMD rendered-source hashes for the six **prefill** packed-wmma routes only; decode kernels
  have no render-equality control today. NV pins: first-token digits, decode sha256, bench
  census row (campaign doc 13.2).

---

## 4. The levers (exhaustive, ranked by recovery)

Every lever states: evidence, mechanism class (VALUES-ONLY / SUBSTRATE / STRUCTURAL-additive,
decided by the diagnostic step, not assumed), where the per-target value lives, recovery
estimate, controls, and the open question for review. Recovery estimates are node-sum based;
they are upper bounds until measured, and each lever re-measures in-place. Recovery numbers
are not additive across levers (the node-sum total exceeds replay busy exceeds wall); the
stacked estimate is 60-80% of the sum, and the campaign doc's 4.0-4.2ms target is the
re-measured end-state, not the arithmetic sum.

**The 60-80% haircut and the 4.0-4.2ms target are incompatible** (review, 2026-08-02). Sum of
levers 2.05-2.25 ms, x 60-80% = **1.23-1.80 ms** of real recovery; from 6.12 ms that lands at
**4.32-4.89 ms**. The 4.0-4.2ms target needs 1.92-2.12 ms and is unreachable even in the best
case. Keep the haircut - it is the honest number - and restate the target as what the levers
support. See section 8 for the revised end-state after the per-lever corrections below.

### L1 - plumbing fusion (E_/r_), ~0.9-1.0 ms. SUBSTRATE

- Evidence: 695 kernels (510 E_ + 185 r_) at 1.6-3.9us each = 1.56ms vs llama's 327 at
  1.3-3.4us = 0.51ms. llama has no separate add/silu kernels; the epilogue is folded into the
  GEMV kernels.
- Mechanism: substrate. The plumbing is ordinary generic JIT lowering shared by every target
  (the E_/r_ chain is present in both the flash and SDPA decode graphs, and in AMD's decode
  graph too); llama's graph shows the ceiling by folding the epilogue into the GEMV kernels.
  Two candidate shapes for review:
  (a) epilogue absorption: the q4k/q6k custom GEMV kernels and the flash block tile absorb
  their immediate per-layer epilogue (residual add, cast, norm, ffn activation mul). Gated by
  a per-target declared capability (e.g. `decode_epilogue_fusion`) so no target's admitted
  route changes without opting in;
  (b) graph-level fusion pass at JIT lowering, keyed by the same per-target fact, which would
  also cover the generic `partial.sum(axis=1)` merges without touching the custom emitters.
- Diagnostic first: count E_/r_ kernels per target in the AMD and Metal decode graphs; if the
  chain is present in all three, the fix is shared and the only per-target question is the
  capability gate's default.
- Per-target value lives in: new `DeviceCapabilities` field or a decode route fact row
  (gate default only; the fusion itself is one shared change).
- Recovery: node-sum 6.61 -> ~5.6ms; replay busy 5.83 -> ~4.9ms (estimate).
- Controls: NV digits/sha; AMD and Metal render equality on the legacy route (capability
  defaults off for both, so both stay byte-identical until each target opts in).
- Open question: which shape fits the repo's migration pattern (additive, data-driven, no
  per-target branches)? (a) touches two custom emitters; (b) touches generic machinery with a
  wider blast radius but covers the partial merges too.

### L2 - Q6_K decode kernels, ~0.6 ms. VALUES-ONLY or SUBSTRATE, decided by diagnostic

- Evidence: `q6k_gen_coop_4096_12288` (down, 18x) 50.1us at 0.82 TB/s (kernel's own mem
  estimate, 46% of ceiling) vs llama w2 Q6_K same shape 29.3us (18x); `q6k_gen_partial_1024_4096_4`
  (k/v, 18x) 17.3us at 0.2 TB/s vs llama v 3.3us (18x).
- Mechanism: values-first. The emitter knobs (`row_tile`, lane extent, staging, vector
  width) are per-target rows in the route spec table; AMD's row keeps the current values.
  Diagnostic: microbench the coop kernel sweeping those knobs. Go/no-go: if the sweep clears
  ~1.1 TB/s (llama-class), the gap is values and the fact row is the whole answer. If it
  stalls below that, the gap is substrate (access pattern / occupancy / instruction mix) and
  the fix is shared, not a NV row -- which is the likely case for the partial route at 0.2
  TB/s (5x off llama's 3.3us on the same shape).
  The partial route's parts count is fixed by the packed storage (`linear.parts`), so the
  llama-class single-pass shape requires either a new additive route family that consumes the
  same parts=4 storage with an in-kernel reduce (SUBSTRATE variant -- the generic
  `partial.sum(axis=1)` merge chain is shared, so absorbing it lifts AMD's partial layers
  too), or a load-time repack (expensive, not recommended).
- Recovery: coop 18x(50.1-29.3)us + partial 18x(17.3-3.3)us ~= 0.63ms.
- Controls: decode render equality (section 5); NV digits/sha.
- Open question: is the single-pass variant worth building, or is the values-only gain
  (coop at llama-class BW, partial staged wider) sufficient for this campaign?

### L3 - flash score kernel, ~0.15 ms. VALUES-ONLY or SUBSTRATE, decided by diagnostic

- Evidence: `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` 7.6us x 36 vs llama
  `flash_attn_ext_vec` 3.17us x 36 (+ combine 3.35us x 36). G4 runs `query_group_size=None`
  (QG=4 -> WARPS=4, THREADS=128), `stage_width=1`, `split_size=48`.
- Mechanism: values-first. Per-target route config row (G4/G5 already are exactly that): NV
  row sweeps QG/stage_width/split_size; AMD G4 row unchanged. Diagnostic: sweep per the
  protocol below; if no row reaches ~3.2us per kernel, the whole-cache tile structure itself
  is substrate (it is the same tile AMD admits, so the structure fix lifts both).
- Recovery: **~0.16ms at d512** (corrected, review 2026-08-02); grows with depth (cache reads
  scale). Corrected back up from 0.039ms: our 7.6us is score-only, and we pay our own combine
  (`flash_fused_gmax_combine`, 3.6us x 36 = 0.13ms) at parity with llama's (3.35us x 36).
  Apples-to-apples per layer: our flash is 7.6 + 3.6 = 11.2us vs llama 3.17 + 3.35 = 6.52us.
  The score-kernel delta is 7.6 - 3.17 = 4.43us x 36 = **0.16ms**; the combine delta is
  0.25us x 36 = 0.009ms. The flash *class* delta +0.17ms is therefore ~all score kernel, and
  the earlier ~0.15ms estimate was essentially right; the "remaining ~0.13ms in untouched
  kernels" was our own combine cost, already at parity.
- Controls: decode render equality; NV digits/sha.
- Open question: at d512 the kernel is latency-bound on tiny data; is the sweep protocol
  (QG in {1,2,4}, stage_width in {1,2,4,8}, split_size in {16,32,48}) the right frame, or is
  the whole-cache tile structure itself the issue at larger depths?

### L4 - vocab head, ~0.2 ms. VALUES-ONLY or SUBSTRATE, decided by diagnostic

- Evidence: `q6k_gen_coop_151936_4096` 397.2us + `q6k_vocab_scalar_reduce` 72.5us + generic
  scatter chain ~0.07ms = 0.54ms vs llama single mmq 303.75us.
- Mechanism: values-first: tune the coop head kernel (row_tile, staging) toward llama-class
  BW. Diagnostic decides the 397us question: if it is occupancy/vector width, that is
  substrate (shared emitter, AMD's head kernel has the same shape) and the fix is shared; if
  it is the row_tile=4 layout on a 151936-row shape, that is a value. The scalar-reduce
  fusion into the coop kernel is a SUBSTRATE variant if pursued (emitter change,
  capability-gated; the generic scatter chain it replaces is shared).
- Recovery: **~0.093ms VALUES-ONLY; ~0.24ms only with the substrate variant** (corrected,
  review 2026-08-02). The 397.2us coop kernel moves ~510 MB of Q6_K weights (151936 x 4096 x
  210/256 bytes) at **1.29 TB/s = 72% of the 1792 GB/s ceiling**. There is not 0.2ms of
  headroom inside it; there is 93us to llama's 303.75us. The recoverable mass is the 72.5us
  scalar reduce plus the ~70us scatter chain (0.14ms), and removing those *is* the substrate
  fusion variant. So L4 cannot deliver 0.2ms in its VALUES-ONLY form.
- Controls: decode render equality; NV digits/sha (token stream is the strongest pin here).
- Open question (narrowed): **the row_tile hypothesis is already falsified - 151936 / 4 =
  37,984 exactly**, so the shape is divisible and no diagnostic is needed to rule it out. With
  the kernel at 72% of the bandwidth ceiling, the only live values-only branch is
  occupancy/vector width, and it is bounded at 93us. The real L4 question is therefore whether
  the substrate fusion (scalar reduce + scatter into the coop kernel) is worth building.

### L5 - q4k lanemap bandwidth, ~0.2-0.3 ms. SUBSTRATE (emitter) + VALUES (lanes)

- Evidence: gate/up 21.2us (72x) and down q4k 27us (18x) vs llama's fused w1w3 37.9us for the
  pair and w2 q4k 19.2us; q/o at parity (9.5 vs 9.5-10.3us). `lanes` is hard-wired to WARP
  (32) in `q4k_g3_lanemap_gemv_kernel`; llama's mmq blocks are 128 threads.
- Mechanism: split. Widening the lane map past WARP=32 is a shared emitter generalization
  (`LanePartition` is one UOp builder; Metal's simdgroups are also 32-wide today, and AMD's
  wave-64 can use a wider lane map through the same mechanism) -- substrate. The numeric
  `lanes`/vector-width row per target is a value; AMD and Metal keep their admitted rows.
- Recovery: ~0.2-0.3ms if the per-kernel delta is occupancy/lane-map.
- Controls: decode render equality; NV digits/sha.
- Open question: `LanePartition` validates `lane_extent == WARP` today
  (`decode_kernels.py:53`); widening the lane map is a small emitter generalization - is that
  in scope, or does L5 wait for L2's spec-table work to exist first?

---

## 5. Controls and measurement protocol

### 5.1 New control - decode render equality (pg3)

pg2 covers prefill routes only. This scope adds the same technique for the decode emitters:
render `q4k_g3_lanemap_gemv` (4 measured shapes), `q6k_gen_coop` (down + vocab),
`q6k_gen_partial`, `q6k_vocab_scalar_reduce`, `flash_block_tiled_xlane_score_pv`,
`flash_fused_gmax_combine` through `HIPRenderer` **and** `MetalRenderer` (render-only, no
ROCm compiler and no Apple hardware needed, exactly as pg2 does) and pin both SHA-256 hash
sets in the campaign doc. Any VALUES-ONLY lever must leave both hash sets unchanged; any
SUBSTRATE lever must change both in the same way (that is the proof the fix is generic); any
STRUCTURAL-additive lever must keep the AMD- and Metal-admitted routes' hashes unchanged.
The Metal arm runs on the macOS box where pg2 already runs (MetalRenderer imports the macOS
Metal runtime; it cannot instantiate on this Linux NV box). The AMD arm runs on either.

**pg3 HIP baseline (2026-08-02, `scratchpad/pg3_decode_rendered_source_equality.py`, HIPRenderer
gfx1100, render-only).** Re-derive with `PYTHONPATH=. .venv/bin/python
scratchpad/pg3_decode_rendered_source_equality.py`; hashes are the house convention
(`sha256((src + "\n").encode())`, first 12 hex; pg2). The flash tile is pinned at the campaign's
measured max_context=4608 (`/tmp/qwen3-8b-nv-p4-decode.json`), rendered at Tc=start_pos+1. The
Metal arm is macOS-only and gets its own pinned block when it first runs on the macOS box.

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

### 5.2 NV pins (existing, re-run per lever)

- First-token digits `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080,
  25, 576, 35264, 5624]`.
- Decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`.
- Bench census row `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d...`.

### 5.3 Perf protocol

Fixed-depth W decode at d512/d2048/d4096 (same-session llama `tg10 @ d`), 5 reps median,
5.83ms busy baseline recorded per lever. AMD runtime measurement is NOT required for
VALUES-ONLY levers; it is required only if a future scope changes AMD's admitted values or
route. SUBSTRATE levers get the same treatment up front: the AMD and Metal render arms prove
the change is shared, and a measured AMD runtime number is the promotion gate later, not the
landing gate here.

---

## 6. Sequencing and delivery

1. P0: pg3 decode render-equality script (HIP + Metal arms) + baseline re-measure of all
   pins (no code change).
2. P1: L3 (smallest surface) and L4 (head tuning), each opening with its diagnostic sweep,
   then the values row or the shared fix per the go/no-go.
3. P2: L5 lane-map generalization (shared emitter) + per-target lanes rows.
4. P3: L2 diagnostic first (coop knob sweep; partial single-pass substrate variant only if
   the sweep stalls below ~1.1 TB/s).
5. P4: L1 substrate fusion, capability-gated, additive (largest recovery, largest blast
   radius; needs its own design doc before code).

Each piece is a separate commit with one owning prefix (`[nn]`, `[test]`, `[docs]`), pushed
to `nvidia-bringup-20260731` only. No promotion to `dev`/`exp`/`master` is authorized by this
scope.

Endpoint expectation, stated up front: the re-measured stack targets 195-210 tok/s at d512
vs llama's 245.6 (close to parity, not beating it); the last 10-15% sits in per-kernel q4k
bandwidth on the lanemap path and is explicitly not claimed by this scope.

HARD STOP after this section. Nothing beyond this scope without review.

---

## 7. Review request

Please review with the repo's data-driven, additive-migration principles in mind. The seven
open questions to settle:

1. Is the SUBSTRATE / VALUES-ONLY / STRUCTURAL-additive trichotomy the right classification,
   and is the diagnostic-first go/no-go per lever the right way to decide it? Are any levers
   misclassified (in particular L5, where the emitter change is substrate but the lanes value
   is per-target)?
2. L1 shape (epilogue absorption vs graph-level fusion pass), and whether the capability gate
   belongs in `DeviceCapabilities` or in a decode route fact row.
3. L2: is the parts=4 single-pass variant worth building, or is values-only sufficient? Does
   the 0.2 TB/s partial route count as substrate by the stated criterion?
4. L3: is QG/stage_width/split_size the right sweep frame, and is there a d512-vs-d4096
   tradeoff to name?
5. L4: does the vocab coop kernel's 397us point at row_tile on a 151936-row shape, or at
   occupancy/vector width (substrate)?
6. Controls: is pg3's kernel set complete, and is the Metal render arm (macOS box, render
   only) the right way to prove a substrate fix is generic?
7. Is the claim "VALUES-ONLY levers need no AMD runtime measurement" still the right call
   given the substrate trichotomy, or should any lever carry a measured AMD number out of
   caution?

---

## 8. Revised budget after the review corrections (2026-08-02)

### 8.1 L2 + L5 versus the GEMV class - cap is the bare-GEMV delta, not the incl-quantize delta

L2 (~0.63ms) and L5 (~0.2-0.3ms) both draw from the **GEMV (non-vocab, incl quantize)** class.
The class table says +0.44 ms (ours 4.16 vs llama 3.72), but the label hides the asymmetry:
llama's 3.72 ms includes its `quantize_q8_1` kernels (0.482 ms across 217 nodes), which
tinygrad does not pay. The like-for-like comparison excludes quantize from both sides: llama's
bare GEMV class is 3.543 - 0.304 (vocab) = **3.24 ms** vs ours 4.16 ms = **0.92 ms** of
headroom. L2+L5 (0.83-0.93 ms) fits under that cap. "Finishing ~0.5 ms ahead of llama on
GEMV" is exactly the quantize asymmetry: llama pays q8_1 activation quantization as separate
kernels, tinygrad's kernels consume packed storage directly. That is the point of the fused
storage, not an arithmetic violation.

This is the evidence-class discipline, applied correctly: per-kernel trace times are licensed
to compare like work. The cap that matters is **0.92 ms (quantize-excluded)**, and L2+L5 stay
under it. A measured quantize-excluded comparison (a DEBUG=2 trace of our GEMV class and a
node-filtered llama trace excluding `quantize_q8_1`) would harden it; that measurement is the
settling check, and it is cheap to run on the 5090.

### 8.2 Revised end-state

| lever | claimed | evidenced |
| --- | ---: | ---: |
| L1 plumbing | 0.9-1.0 ms | 0.9-1.0 ms (consistent with the 1.05 ms class delta) |
| L2 + L5 GEMV | 0.83-0.93 ms | <= 0.92 ms (bare-GEMV cap, 8.1) |
| L3 flash score | 0.15 ms | 0.16 ms (score delta 4.43us x 36; combine at parity) |
| L4 vocab | 0.2 ms | 0.093 ms values-only / 0.24 ms with substrate |
| **gross** | **2.05-2.25 ms** | **~1.98-2.18 ms** (values-only L4) / 2.13-2.33 with L4 substrate |
| after 60-80% haircut | 1.23-1.80 ms | **~1.19-1.75 ms** |
| end-state ms/token | 4.32-4.89 | **~4.37-4.93 ms** (values-only) / 4.26-4.84 with L4 substrate |

Against llama's 4.07 ms that is decode at roughly **1.07-1.21x** - parity is back in reach at
the optimistic end (4.37 vs 4.07 = 1.07x), not guaranteed, and not the old "not parity,
1.18-1.29x" verdict. L1 remains the largest single lever at ~45-50% of the realistic total
(0.9-1.0 of 1.98-2.18), so the SUBSTRATE work still belongs early; section 6's ordering
(L1 as P4) should be revisited. The 4.0-4.2 ms target is not supported by this budget; the
honest target is 4.37-4.93 ms with the values-only stack.

Every number in this table is node-sum-derived and inherits section 1's over-count. It is a
corrected *upper* bound, not a forecast.

---

## 9. Reviewer disagreement protocol (first principles)

Two model reviewers producing findings on the same doc will disagree, and there is no authority
that settles it by argument. This section derives the resolution rule from the repo's own
principle rather than inventing one.

### 9.1 The governing principle

`structure/Development/coding-principles.md`, **"Classify Evidence Before Fixing Mechanisms"**:

> classify what each piece of evidence is allowed to prove before changing code [...] Do not let
> a passing narrow gate authorize a wider claim.

Every disputed item in this campaign has been an instance of that rule, not a factual conflict:

| disputed number | evidence class | what it was licensed to prove | what it was used for |
| --- | --- | --- | --- |
| 92% GPU-efficiency | cold `nsys` busy / warm bench wall | nothing (two runs) | P3's success criterion |
| 0.85 busy/wall | cold capture / warm wall | nothing (two runs) | "P3 already met" |
| L1's 89% | share of a subtotal | share *of that subtotal* | share of total prefill |
| 24.1 ms busy | `GlobalCounters` estimate | tinygrad-internal accounting | comparison against llama's `nsys` |
| L3's 0.15 ms | one kernel's per-call delta | that kernel's delta | the whole flash class |
| L4's 0.2 ms | class-level delta | reduce + scatter + kernel | a values-only kernel tune |

None of these were wrong measurements. All were correct measurements used past their license.

### 9.2 The rule

**A finding is admissible only if it names (a) the claim it disputes, (b) the evidence class the
disputed claim rests on, (c) what that class is licensed to prove, and (d) the single command
that would settle it.** A finding without (d) is a question, not a defect, and is recorded as
such.

**Disagreements are resolved by running (d), never by argument.** Whichever reviewer's finding
names a runnable check wins by default; if both name checks, run the cheaper one first.

### 9.3 The dissolution case

Most reviewer disagreements are not conflicts - they are two reviewers licensing different
evidence classes and talking past each other. If reviewer A says "L4 has 0.2 ms of headroom"
from a class-level node-sum and reviewer B says "0.093 ms" from bandwidth arithmetic on the
kernel, both are right about their own class and the disagreement dissolves once the classes are
named. **Name the class before adjudicating; only a same-class conflict is a real disagreement.**

### 9.4 What this asks of each reviewer

- State the evidence class of every number you produce or dispute.
- Never divide two numbers from different runs or different instruments; if you must, label the
  result as unlicensed and do not let it become a criterion.
- Prefer the check that falsifies your own finding cheapest.
- A ceiling that a measurement has already exceeded is a falsified model, not a conservative
  one; say so rather than widening the model.

### 9.5 Open items under this protocol

Carried forward as *questions* because no settling command has been named yet:

1. What does node-sum double-count such that it exceeds wall (section 1)? Settles: sections 4
   and 8 recovery numbers.
2. Is there a quantize-excluded GEMV class comparison? Settles: whether the 8.1 cap is right or
   whether L2+L5 genuinely have more room.
3. Does the ~0.13 ms of non-score flash cost (L3) belong to a lever at all?

---

## 10. P1 results - L3 substrate verdict, L4 values row landed (2026-08-02)

RTX 5090, Qwen3-8B-Q4_K_M, d512 fixed-depth decode authority (`bench.py --decode`,
5 reps median), DEBUG=2 prime-token per-kernel times. Pins before/after: first-token digits
`[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080, 25, 576, 35264, 5624]`,
decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, census
`prefill_overlay_promotion: candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`,
and all 36 swept configs kept the fixed-depth token sha `5662f1cd...`.

### L3 - SUBSTRATE verdict (no code change)

36-config sweep of the G4 route values in-process (query_group_size x stage_width x split_size,
monkeypatched `FLASH_DECODE_G4`, never edited shared rows). Best score kernel
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`:

| qg (None=4) | stage_width | split_size | score us | combine us | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 48 | 7.41 | 3.65 | 163.25 |
| 4 | 8 | 48 | 7.46 | 3.63 | 163.41 |
| 4 | 2 | 48 | 7.48 | 3.63 | 163.12 |
| 4 | 1 | 48 (baseline) | 7.59 | 3.71 | 162.97 |
| 2 | 4 | 48 | 9.24 | 3.65 | 161.61 |
| 1 | 8 | 48 | 14.02 | 3.68 | 157.17 |

| split_size (qg=4) | score us |
| --- | ---: |
| 48 | 7.4-7.6 |
| 32 | 9.6-9.8 |
| 16 | 11.9-12.4 |

No row reaches the ~4us go/no-go bar (llama ext_vec 3.17us); the best value shaves only
0.18us (2.4%) off the baseline and the tok/s spread across all 36 rows is 149.1-163.4. L3 is
classified SUBSTRATE (whole-cache tile structure, per section 4's criterion) and value tuning
stops here.

### L4 - VALUES row landed: Q6K coop row_tile=2 on NV

`q6k_gen_coop` row_tile sweep (151936-row vocab head + 4096-row down, same emitter, in-process
`Q6K_DECODE_CANDIDATE` replace; lane_extent is pinned to 16 by the emitter's validate and the
coop emitter exposes no other staging/vector knob):

| row_tile | vocab coop us | down coop us | scalar reduce us | d512 tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 473.4 | 57.6 | 72.5 | 158.9 |
| 2 | 330.1 | 35.5 | 72.5 | 172.4 |
| 4 (baseline) | 397.4 | 49.7 | 72.5 | 163.2 |
| 8 | 383.7 | 54.2 | 72.5 | 161.4 |

row_tile=2 clears the ~340us bar and is implemented as per-target data (commit `ab3cb84c1`):
`Q6K_COOP_ROW_TILE_BY_TARGET = {("NV", "sm_120"): 2}` in `tinygrad/llm/decode_kernels.py`,
resolved at bind time from the installed primitive's TG3 admission capability
(`decode_routes.py::_Q6KDecodeCandidate.bind`); an undeclared target keeps the AMD search value
4. AMD pg3 hashes re-verified byte-identical after the change (all 10 rows, section 5.1 block).
Measured after: vocab coop 330.1us (1.55 TB/s, 86% of ceiling), down coop 35.5us, d512 decode
**163.5 -> 172.6 tok/s** (6.12 -> 5.80 ms/token), e2e growing-window decode 158.3 -> 166.8
tok/s, all NV pins unchanged.

---

## 11. P2 + P3 results - L5 lanemap verdict, L2 partial verdict (2026-08-02)

Same rig/protocol as section 10. Sweeps were in-process config patches only; no shared
route row was edited. The fixed-depth token sha for every legal sweep row in this section
is `9d6b3787ce...` (this harness: nmeas=20, reps=3; the sha is per-harness, like L4's
`5662f1cd...` at nmeas=40).

### L5 - SUBSTRATE verdict (q4k lanemap lanes sweep, no code change)

Swept `Q4KGateUpLaneMap.lane_extent` = 32/64/128 in-process (relaxed the wave-32
`LanePartition.validate` gate; the XOR-shuffle reduce ladder is wave-32, so lanes>32
runs are timing-only with garbage output - the per-kernel time is the diagnostic).
Median kernel us:

| kernel | lanes=32 | lanes=64 | lanes=128 |
| --- | ---: | ---: | ---: |
| q4k_g3_lanemap_gemv_12288_4096 (gate/up, 72x) | 20.98 | 19.97 | 20.95 |
| q4k_g3_lanemap_gemv_4096_4096 (q/o, 36x) | 9.36 | 8.72 | 9.28 |
| q4k_g3_lanemap_gemv_4096_12288 (down, 18x) | 26.74 | 28.74 | 21.81 |
| q4k_g3_lanemap_gemv_1024_4096 (k/v, 36x) | 4.86 | 4.28 | 4.19 |

No class-wide win: gate/up - the 72x mass of the class - is flat across all three widths,
and 64 lanes gains nothing anywhere. Only down (-18% at 128) and k/v (-14% at 128) move,
and those shapes are a minority of the class time. Even the optimistic partial win would
not move the class. Widening past WARP=32 requires a real cross-wave reduce (the wave-32
staged shuffle ladder cannot extend), which is shared substrate, not the "small emitter
generalization" the scope assumed. **L5 = SUBSTRATE; the lane-value hypothesis is
falsified.** No code change.

### L2 - partial route SUBSTRATE verdict (q6k opts sweep, no code change)

The partial spec's ONLY free value knob is `opts` (forwarded to
`KernelInfo.opts_to_apply`); `lane_extent`/`pos_axis`/`block_axis`/`reduction`/`storage`
are validate-pinned. Swept installed `Q6KPrimitiveLinear.opts` on the 18 Q6_K k/v layers
(`q6k_gen_partial_1024_4096_4`, parts=4), DEBUG=2 prime-token kernel time + d512 e2e:

| opts | partial us | d512 tok/s |
| --- | ---: | ---: |
| `()` (no opts) | 45.25 | 159.9 |
| `LOCAL:0:32` (installed policy row) | 17.15 | 172.97 |
| `LOCAL:1:4` | 18.47 | 172.1 |
| `LOCAL:0:64` | 21.68 | 170.2 |
| `UPCAST:0:2` / `0:4` / `1:2` / `1:4` (+/- LOCAL) | CRASH: shared devectorizer `fold_expanded_index` AssertionError | - |
| `UNROLL:0:2` / `0:4` (blk_part) | CRASH: same devectorizer AssertionError | - |
| `UNROLL:0:8` (blk_part, under LOCAL) | KernelOptError: 8 can't divide the size-4 reduce | - |
| `UNROLL:1:4` / `1:8` / `1:16` (pos) | nvcc fails: CUDA renderer emits `val50.y` on a scalar float | - |

The expansion defects were proven backend-independent where it matters: rendering the
partial route with `UPCAST:0:2` through HIPRenderer fails with the identical
devectorizer AssertionError (compile-only, no ROCm), so the UPCAST defect is shared late
codegen, not an NV row. `UNROLL:1:4` renders through HIPRenderer (src_len 23223) but
fails nvcc on the CUDA arm, so the unrolled-reduce vectorization is an NV-renderer bug.
Both are shared-code defects, i.e. substrate evidence under section 4's criterion.

The legal rows show the installed machine-search value is already optimal: 17.15us at
0.20 TB/s versus llama's v kernel 3.3us at 1.04 TB/s (5x). The route is structurally
bound: 4096 threads (rows x parts) each serially reduce 4 blocks x 16 pos; llama's
single-pass shape spreads the reduction over more threads. **L2 partial = SUBSTRATE**;
the in-kernel single-pass variant (parts merged in-kernel, or the reduce split across
threads) is the structural fix and is a separate scope - recorded here, not built. The
other L2 item, the Q6_K down coop route, was already improved by L4's row_tile=2
(35.5us vs llama 29.3us, 1.2x, values-saturated per the L4 sweep).

Budget consequence: L2+L5's values-only headroom was capped at 0.92ms (section 8.1). The
live values-only mass after these verdicts is only the coop-down residual
(18 x 6.2us ~= 0.11ms, already values-saturated). The partial 0.25ms and the L5 class
are now recorded as substrate; they remain recoverable only through the structural
scopes (in-kernel single-pass partial, cross-wave lane map), which this campaign does
not build. Section 8.2's end-state stands unchanged on its own terms, with the L2+L5
line now understood as substrate-parked rather than values-open.

## 12. P4 M2 result - Q6K in-kernel merges: coop lands, partial documented non-landing (2026-08-02)

Design: l1-decode-plumbing-fusion-design-20260802.md section 6 classes 9 (`r_8_8_16_2_4`,
partial v merge) and 10 (`r_32_32_4_2_8`, down coop merge). Landing commit: `[nn]` M2 (see
git log); the promotion record `decode-epilogue-fusion-route-policy.json` now names
`NV:sm_120` as the first fused consumer, and every other target stays closed.

### Landed - coop down merge (`q6k_gen_coop_4096_12288_inkernel`)

The coop emitter gains a `reduction="in_kernel"` variant: a REG accumulator plus a staged
shuffle ladder over the 16 pos lanes, writing `(rows,)` instead of `(rows,16)` and removing
the generic `partial.sum(axis=1)` merge kernel and its 4.6 MB/token fp32 round-trip.
Admission rides the M1 closed-default gate: `_Q6KDecodeCandidate.execute` selects it only
when `fusion_admitted` AND the binding is coop AND not the vocab head (the vocab head keeps
the scalar-reduce path; design Q9, L4 boundary).

Measured (d512 fixed-depth, DEBUG=2 prime-token medians, same rig as section 11):

| kernel | before | after |
| --- | ---: | ---: |
| down coop gemv + merge | 35.5 + 2.08 us | 34.9 us (fused) |
| partial v gemv + merge | 17.15 + 2.09 us | unchanged (17.2 + 2.11) |

Tokens: fixed-depth sha unchanged (`9d6b3787ce...`, 3/3 reps), first-token digits
unchanged (`151936`), d512 tok/s 172.97 -> 173.45 (the down-path launch-count win is small
at d512 because it is ~0.05 ms/token of a 5.8 ms budget; the kernel-level evidence above is
the primary row). Numeric parity probe (real weights, bitwise compare vs legacy merge):
2814/4096 last-bit order deltas, max_abs 7.2e-07, no digit movement - the same order-only
class the e2e sha gate already accepts.

### Non-landing - partial in-kernel merge (class 9), evidence

Two in-kernel partial shapes were built and timed on NV before the landing decision:

| shape | us | vs legacy 17.15 + 2.09 |
| --- | ---: | --- |
| 4-thread blocks (part LOCAL only) | 25.3 | +6.0 us/layer LOSS |
| 8-row x 4-part 32-thread blocks (this shape) | 466.6 | 27x SLOWER |

The 8-row shape is a reproducible 466.6 us in the real decode loop (token sha still
parity-clean at 3/3 reps while it was active). CUDA and HIP source diffs against legacy
(LOCAL:0:32, 128 blocks x 32 threads, identical loops and per-thread work) show only two
staged shuffles and a gated store added - no structural explanation found at source level,
and a standalone launch-config hypothesis (2D block 4x8 vs 1D 32) was not the driver in the
4-thread control. Pursuing this shape further was not justified: its ceiling is ~2.1 us
per layer (~0.04 ms/token) against the norm family's 0.58 ms, so the mystery is recorded
here, not chased. The partial in-kernel merge is a documented non-landing; the legacy
`external_sum` route is untouched and `Q6KGEMVRouteSpec.validate` now rejects
`reduction="in_kernel"` for the partial family with a pointer to this record. Revisit under
the substrate single-pass partial scope (section 11 L2 verdict), not under L1.

### Constraints and controls

- Single-warp constraint: the coop in-kernel ladder requires `row_tile * lane_extent <= 32`,
  enforced in `validate()`. NV's measured row_tile=2 is legal (2 x 16 = 32); AMD's row_tile=4
  is not (4 x 16 = 64) - AMD stays external_sum until a two-warp ladder exists. This is why
  the pg3 fused row renders at row_tile=2 through both renderers.
- pg3 HIP arm: all 10 legacy hashes re-verified byte-identical to section 5.1; new fused row
  `q6k_gen_coop_4096_12288_inkernel = add50a7aa43f` (src_len 9440, ds_bpermute=4).
- Unit gate updated: `test_decode_epilogue_fusion_gate.py` now pins the M2 promotion set
  (NV only), the partial in-kernel rejection, the single-warp constraint, and a compile-only
  render of the fused kernel through HIPRenderer and CUDARenderer.
- Census: 695 -> 677 E_/r_ kernels/token (the 18 `r_32_32_4_2_8` merges disappear; the 18
  `E_8_8_16_2` down-path companions remain, owned by M4).
