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

Every lever states: evidence, mechanism class, where the per-target value lives, recovery
estimate, controls, and the open question for review. Recovery estimates are node-sum based;
they are upper bounds until measured, and each lever re-measures in-place.

### L1 - plumbing fusion (E_/r_), ~0.9-1.0 ms. STRUCTURAL

- Evidence: 695 kernels (510 E_ + 185 r_) at 1.6-3.9us each = 1.56ms vs llama's 327 at
  1.3-3.4us = 0.51ms. llama has no separate add/silu kernels; the epilogue is folded into the
  GEMV kernels.
- Mechanism: structural. Two candidate shapes for review:
  (a) epilogue absorption: the q4k/q6k custom GEMV kernels and the flash block tile absorb
  their immediate per-layer epilogue (residual add, cast, norm, ffn activation mul), gated by
  a per-target declared capability (e.g. `decode_epilogue_fusion`) so AMD's admitted route is
  untouched;
  (b) graph-level fusion pass at JIT lowering, keyed by the same per-target fact, which would
  also cover the generic `partial.sum(axis=1)` merges without touching the custom emitters.
- Per-target value lives in: new `DeviceCapabilities` field or a decode route fact row.
- Recovery: node-sum 6.61 -> ~5.6ms; replay busy 5.83 -> ~4.9ms (estimate).
- Controls: NV digits/sha; legacy AMD route render equality (the capability defaults off for
  AMD).
- Open question: which shape fits the repo's migration pattern (additive, data-driven, no
  per-target branches)? (a) touches two custom emitters; (b) touches generic machinery with a
  wider blast radius but covers the partial merges too.

### L2 - Q6_K decode kernels, ~0.6 ms. VALUES-ONLY, with one structural variant option

- Evidence: `q6k_gen_coop_4096_12288` (down, 18x) 50.1us at 0.82 TB/s (kernel's own mem
  estimate, 46% of ceiling) vs llama w2 Q6_K same shape 29.3us (18x); `q6k_gen_partial_1024_4096_4`
  (k/v, 18x) 17.3us at 0.2 TB/s vs llama v 3.3us (18x).
- Mechanism: VALUES-ONLY for the emitter knobs (`row_tile`, lane extent, staging, vector
  width) as per-target rows in the route spec table; AMD's row keeps the current values.
  The partial route's parts count is fixed by the packed storage (`linear.parts`), so the
  llama-class single-pass shape requires either a new additive route family that consumes the
  same parts=4 storage with an in-kernel reduce (STRUCTURAL variant), or a load-time repack
  (expensive, not recommended).
- Recovery: coop 18x(50.1-29.3)us + partial 18x(17.3-3.3)us ~= 0.63ms.
- Controls: decode render equality (section 5); NV digits/sha.
- Open question: is the single-pass variant worth building, or is the values-only gain
  (coop at llama-class BW, partial staged wider) sufficient for this campaign?

### L3 - flash score kernel, ~0.15 ms. VALUES-ONLY

- Evidence: `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` 7.6us x 36 vs llama
  `flash_attn_ext_vec` 3.17us x 36 (+ combine 3.35us x 36). G4 runs `query_group_size=None`
  (QG=4 -> WARPS=4, THREADS=128), `stage_width=1`, `split_size=48`.
- Mechanism: per-target route config row (G4/G5 already are exactly that): NV row sweeps
  QG/stage_width/split_size; AMD G4 row unchanged.
- Recovery: ~0.15ms at d512; grows with depth (cache reads scale).
- Controls: decode render equality; NV digits/sha.
- Open question: at d512 the kernel is latency-bound on tiny data; is the sweep protocol
  (QG in {1,2,4}, stage_width in {1,2,4,8}, split_size in {16,32,48}) the right frame, or is
  the whole-cache tile structure itself the issue at larger depths?

### L4 - vocab head, ~0.2 ms. VALUES-ONLY, with a fused variant option

- Evidence: `q6k_gen_coop_151936_4096` 397.2us + `q6k_vocab_scalar_reduce` 72.5us + generic
  scatter chain ~0.07ms = 0.54ms vs llama single mmq 303.75us.
- Mechanism: VALUES-ONLY: tune the coop head kernel (row_tile, staging) toward llama-class
  BW. The scalar-reduce fusion into the coop kernel is a STRUCTURAL variant if pursued
  (emitter change, capability-gated).
- Recovery: ~0.2ms.
- Controls: decode render equality; NV digits/sha (token stream is the strongest pin here).
- Open question: does the coop head kernel's 397us come from occupancy/vector width, or from
  the row_tile=4 layout on a 151936-row shape (row count not divisible by useful tiles)?

### L5 - q4k lanemap bandwidth, ~0.2-0.3 ms. VALUES-ONLY

- Evidence: gate/up 21.2us (72x) and down q4k 27us (18x) vs llama's fused w1w3 37.9us for the
  pair and w2 q4k 19.2us; q/o at parity (9.5 vs 9.5-10.3us). `lanes` is hard-wired to WARP
  (32) in `q4k_g3_lanemap_gemv_kernel`; llama's mmq blocks are 128 threads.
- Mechanism: per-target `lanes`/vector-width fact row on the route spec; AMD keeps 32.
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
`flash_fused_gmax_combine` through `HIPRenderer` (render-only, no ROCm compiler needed,
exactly as pg2 does) and pin SHA-256 hashes in the campaign doc. Any VALUES-ONLY lever must
leave these hashes unchanged; any STRUCTURAL lever must keep the AMD-admitted route's hashes
unchanged.

### 5.2 NV pins (existing, re-run per lever)

- First-token digits `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080,
  25, 576, 35264, 5624]`.
- Decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`.
- Bench census row `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d...`.

### 5.3 Perf protocol

Fixed-depth W decode at d512/d2048/d4096 (same-session llama `tg10 @ d`), 5 reps median,
5.83ms busy baseline recorded per lever. AMD runtime measurement is NOT required for
VALUES-ONLY levers; it is required only if a future scope changes AMD's admitted values or
route.

---

## 6. Sequencing and delivery

1. P0: pg3 decode render-equality script + baseline re-measure of all pins (no code change).
2. P1: L3 (smallest surface, values-only) and L4 (values-only head tuning).
3. P2: L5 (lane-map generalization, if reviewed in).
4. P3: L2 values-only tuning (coop to llama-class BW; partial staging).
5. P4: L1 structural, capability-gated, additive (largest recovery, largest blast radius).

Each piece is a separate commit with one owning prefix (`[nn]`, `[test]`, `[docs]`), pushed
to `nvidia-bringup-20260731` only. No promotion to `dev`/`exp`/`master` is authorized by this
scope.

HARD STOP after this section. Nothing beyond this scope without review.

---

## 7. Review request

Please review with the repo's data-driven, additive-migration principles in mind. The five
open questions to settle:

1. L1 shape (epilogue absorption vs graph-level fusion pass), and whether the capability gate
   belongs in `DeviceCapabilities` or in a decode route fact row.
2. L2: is the parts=4 single-pass variant worth building, or is values-only sufficient?
3. L3: is QG/stage_width/split_size the right sweep frame, and is there a d512-vs-d4096
   tradeoff to name?
4. L4: does the vocab coop kernel's 397us point at row_tile on a 151936-row shape?
5. Controls: is pg3's kernel set complete, and is the claim "VALUES-ONLY levers need no AMD
   runtime measurement" the right call, or should any of these still carry a measured AMD
   number out of caution?
