# L4 vocab substrate fusion - variant-specific implementation scope

Date: 2026-08-03
Status: implementation scope for Scope B (L4 vocab substrate fusion) only. This document
authorizes no implementation and no GPU use; implementation proceeds only under its own
gates (section 3) and its own review, per the house rule that each variant-specific scope
carries its own settling command, legacy-hash controls, correctness pins, and fixed-depth
wall gate. Branch boundary: tinygrad `nvidia-bringup-20260731` at `d990ff87d` (HEAD).

Authority chain: `nv-decode-parity-final-20260802.md` (wall authority, harness protocol),
`nv-parity-and-beyond-forward-scope-20260803.md` (canonical umbrella: lifecycle states,
section 5 parity criterion, section 7 unresolved), and
`decode-gemv-efficiency-forward-scope-20260803.md` section 4 (Scope B) with sections 7-8
(endpoint discipline and pg3 controls). The Stage 2 fused-shape GO is the measurement
record `l4-vocab-substrate-fusion-measurement-record-20260803.md` (commit `05b1e9774`).

---

## 0. HARD BANS

- No dtype or precision changes of any kind. fp32 accumulate, fp16 activation/weight
  operands, and the existing RNE casts stay exactly as emitted today; the fused variant
  changes only the reduction location, never the arithmetic.
- Do not touch `prefill_routes.py` in any way. Decode evidence never authorizes prefill;
  prefill is a separate admission class.
- No new subsystem. The change is one shared emitter (`_emit_q6k_coop`,
  `tinygrad/llm/decode_kernels.py:447`, already implemented) plus the existing admission
  surface in `decode_routes.py`. No new route record, no new kernel family, no new
  builder, no new policy file.
- Do not touch `BOUNDED_PACKED_TILES`. That is the L2 partial storage surface, not the
  vocab coop path, and this scope has nothing to do with it.
- Do not touch the landed values row: commit `ab3cb84c1`
  (`Q6K_COOP_ROW_TILE_BY_TARGET = {("NV", "sm_120"): 2}` at `decode_kernels.py:215`). It
  stays the baseline this scope builds on; the fusion change is separate and additive.
- Do not reopen the M2 partial-family rejection. `Q6KGEMVRouteSpec.validate` rejects
  `reduction="in_kernel"` for the `q6k_partial` family with a pointer to the M2
  non-landing record; that rejection stays.
- Do not touch the M2 promotion record `decode-epilogue-fusion-route-policy.json`. It
  already promotes NV sm_120 and stays exactly as landed.
- AMD (row_tile=4) and Metal must remain on `external_sum`. Their rendered routes must be
  byte-identical to the pg3 pinned table; the AMD in-kernel shape is not legal under the
  single-warp constraint (`4 * 16 = 64 > 32`) and Metal has no fusion admission.
- Legacy rows byte-identical: all 10 pg3 HIP hashes unchanged, plus the two additive fused
  rows (`q6k_gen_coop_4096_12288_inkernel`, `flash_fused_gmax_combine_f16_32_128`)
  unchanged.
- No promotion to `dev`/`exp`/`master`. No push; the parent pushes after review.
- Never touch the untracked scratchpads (`extra/llm_research/microbench/dp4a_peak_cuda*`,
  `scratchpad/t6_metal_admission_probe.py`) or the user-owned docs
  (`docs/README.md`, `docs/beating-llama-first-principles-20260731.md`,
  `docs/what-makes-inference-fast.md`).

## 1. Established state (verified facts)

- `decode_routes.py:181` currently selects
  `reduction = "in_kernel" if fusion_admitted and not is_vocab and binding.use_coop else
  "external_sum"`. The `not is_vocab` exclusion keeps the vocab head off the in-kernel
  path; `is_vocab = binding.use_coop and binding.N >= Q6K_VOCAB_SCALAR_REDUCE_MIN_ROWS`
  (`decode_routes.py:178`, min rows `131072` at `decode_kernels.py:208`).
- `_emit_q6k_coop` (`decode_kernels.py:447`) already implements `reduction="in_kernel"`
  for the down path: the in-kernel branch (lines 457-462) runs
  `_q6k_coop_pos_reduce_sum` (line 252), a 4-step `__shfl_xor_sync` ladder
  (16, 8, 4, 2) reducing the 16 pos lanes within a single warp. No emitter change is
  needed; the vocab head is the same shape class (151936 rows, k=4096, row_tile=2 ->
  grid 75968 x (2,16)).
- `Q6KGEMVRouteSpec.validate` at `decode_kernels.py:406` already gates
  `row_tile * lane_extent > 32` for in-kernel: legal at NV row_tile=2 (`2 * 16 = 32`),
  illegal at AMD row_tile=4 (`4 * 16 = 64`). This is the fail-closed backstop; AMD must
  stay `external_sum`.
- The M2 promotion record promotes exactly `{("NV", "sm_120")}`; `fusion_admitted` on the
  vocab head is therefore NV sm_120-only today.
- Baseline stack (same-session, measurement record): `q6k_gen_coop_151936_4096` at
  330.1 us (row_tile=2, 1.55 TB/s, 86% of the 1792 GB/s ceiling) +
  `q6k_vocab_scalar_reduce` 72.5 us + scatter chain ~54.5 us
  (`E_1187_32_4` 3.46 + `r_32_4_1187` 38.37 + `r_128_16_8_1187` 10.82 + `r_16_8` 1.89)
  = ~457 us total vs llama's single mmq vocab kernel at 303.75 us.
- Stage 2 fused probe (record `05b1e9774`): emitted
  `q6k_gen_coop_151936_4096_inkernel` (grid 75968 x (2,16), 4 `shfl_xor_sync`) =
  315.9 us / 1616 GB/s / 90.1%; output bit-identical to the (N,16) partials reduced over
  pos on the host (max abs diff 0.0 over all 151936 rows); the scalar-reduce and scatter
  kernels are structurally gone. GO.
- Wall authority baseline (same-session, M2-open): d512 172.80 / d2048 161.50 /
  d4096 149.00 tok/s vs llama 248.20 / 235.14 / 225.95 (ratio 0.696x / 0.687x / 0.659x);
  1021 kernels/token; 6187 / 6576 / 7091 us/token at d512/d2048/d4096.
- LOC budget is 100000 (`sz.py`, raised at `d990ff87d`); no line compression is needed.

## 2. Scope of change

The exact code change surface is one admission expression in
`decode_routes.py` `_Q6KDecodeCandidate.execute` (line 181): allow the vocab head's coop
route to select `reduction="in_kernel"` when the existing single-warp constraint is
satisfied, i.e. when `fusion_admitted and binding.use_coop and
binding.row_tile * Q6K_POS_EXTENT <= 32`. On NV sm_120 that means row_tile=2 is legal
(`2 * 16 = 32`); AMD row_tile=4 and Metal stay `external_sum` by the same gate plus the
unchanged promotion record. The change is:

1. Capability-gated: it rides the already-promoted `decode_epilogue_fusion` record
   (`fusion_admitted`, NV sm_120-only); no record is modified or added.
2. Additive: the legacy `external_sum` + scalar-reduce + scatter path stays the default
   for every target and shape that does not admit; AMD and Metal render byte-identical.
3. Fail-closed: `Q6KGEMVRouteSpec.validate:406` remains the backstop; constructing an
   illegal in-kernel spec still raises, and the `q6k_partial` in-kernel rejection is
   untouched.

What must NOT change:

- No emitter change to `_emit_q6k_coop`; the in-kernel branch already renders the vocab
  shape.
- No dtype/precision changes; the fused output remains fp32 `(N,)` and the route's
  `reshape(1, 1, N)` return path already exists for in-kernel (`decode_routes.py:188-189`).
- No `prefill_routes.py`, no new subsystem, no `BOUNDED_PACKED_TILES`, no new route
  record or policy file.
- No change to the landed values row `ab3cb84c1`.
- No change to the 10 legacy pg3 rows or the two additive fused rows (section 4).

## 3. Gates (fixed-depth wall and sha, before any record change)

Isolated same-session measurement, protocol per `nv-decode-parity-final-20260802.md` and
`decode-gemv-efficiency-forward-scope-20260803.md` section 7:

1. Fixed-depth decode at d512 first: `/tmp/path3_e2e_probe.py`-style harness,
   `--depth 512 --nmeas 20 --reps 3 --no-fused-prefill`, median tok/s, Qwen3-8B-Q4_K_M,
   model `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, fused prefill attention disabled
   (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`), all GPU runs
   serialized with `flock /tmp/nv_gpu.lock` and 0% util at acquisition. Same-session
   llama rows: `llama-bench -ngl 99 -fa 1 -pg 0,10 -d <depth> -r 5`, decode row
   (`n_prompt=0, n_gen=10`).
2. Then d2048 and d4096 rows under the same protocol, each with its own llama row.
3. Per-depth parity qualification per canonical scope section 5: a target point is
   PARITY-QUALIFIED only when the same-session harness median satisfies
   `tinygrad >= llama` at that point (ratio >= 1.00), with the matching llama row from
   the same session family and the referencing harness's repetition protocol. A win at
   d512 qualifies d512 only; d2048 and d4096 each need their own rows. Campaign-wide
   "decode parity" phrasing is not allowed from a single depth.
4. Pins at every depth, all unchanged: token sha256
   `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3; first token
   `151936` 3/3; decode sha256
   `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`; bench census row
   `prefill_overlay_promotion: candidate_set:sha256:
   1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.
5. Census expectation at d512: kernel count 1021 -> fewer. The scalar-reduce kernel
   (`q6k_vocab_scalar_reduce_151936_4096`) and the scatter-chain kernels
   (`E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8` per the record's prime
   trace) disappear, for an expected 1021 -> 1016 (1 scalar-reduce + 4 scatter-chain
   kernels removed), asserted by the census run.

## 4. Test plan

1. Admission unit tests (CPU-only, no GPU):
   - NV sm_120 vocab in_kernel admits: vocab rows (>= 131072, e.g. 151936), row_tile=2,
     `fusion_admitted` -> reduction selects `in_kernel`; emitted kernel is
     `q6k_gen_coop_151936_4096_inkernel` and renders.
   - AMD row_tile=4 rejects and stays `external_sum`: AMD gfx1100 facts, row_tile=4,
     vocab rows -> `external_sum` even with `fusion_admitted`; and
     `Q6KGEMVRouteSpec(rows=151936, k=4096, row_tile=4, reduction="in_kernel").validate()`
     raises on the single-warp constraint.
   - Metal unchanged: Metal target has no fusion admission -> `external_sum`, render
     unchanged.
   - Existing gate tests stay green: `test_coop_in_kernel_single_warp_constraint` and
     `test_coop_in_kernel_renders_through_hip_and_cuda_without_gpu` in
     `test/unit/test_decode_epilogue_fusion_gate.py`.
2. pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU, no lock): all 10
   legacy rows byte-identical
   (`312422c73a49` / `27857cb8ca03` / `851760e2053c` / `39ddb717ddd4` / `cc38fbb3db92` /
   `5795e66a7292` / `344e1c388eeb` / `c708302aa2d2` / `66d4c4da3108` / `c78e4651ad35`),
   and the additive fused rows unchanged:
   `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f` (src_len 9440, ds_bpermute=4) and
   `flash_fused_gmax_combine_f16_32_128` = `94d73c1e9650`.
3. Fixed-depth token sha pin (section 3 item 4) at every depth.
4. Census assertion at d512: scalar-reduce and scatter kernels disappear (section 3
   item 5).

## 5. Deliverable + HARD STOP

Deliverable, one owning prefix per commit, no mixing:

- `[nn]` emitter/admission change (`decode_routes.py` reduction selection only).
- `[test]` admission unit tests + census assertion.
- `[docs]` implementation measurement record with the census, wall, sha, and verdict
  tables.

Report: commits, pytest result, pg3 hashes, census default vs fused, wall tok/s per
depth, sha pins, verdict, deviations, blocked on.

HARD STOP after this section. This document authorizes nothing; implementation proceeds
only under its own gates (section 3) and its own review. Nothing beyond this scope
without review. No promotion to `dev`/`exp`/`master`; no push; the parent pushes after
review.

## 6. One-line job

Admit the vocab coop head to `reduction="in_kernel"` on NV sm_120 (legal at row_tile=2
under the existing single-warp validate gate), keeping AMD and Metal on `external_sum`
byte-identical, and land it only after the fixed-depth sha pins, pg3 render-equality,
census (1021 -> fewer), and isolated same-session d512/d2048/d4096 parity-qualification
rows all pass.

## References

- `decode-gemv-efficiency-forward-scope-20260803.md` (section 4 Scope B; sections 7-8)
- `l4-vocab-substrate-fusion-measurement-record-20260803.md` (Stage 2 GO, commit `05b1e9774`)
- `m5-variant-reopen-boundary-p0-scope-20260803.md` (house format, pg3 table)
- `nv-parity-and-beyond-forward-scope-20260803.md` (canonical umbrella, sections 2.2, 3, 5, 7)
- `nv-decode-parity-final-20260802.md` (wall authority, harness protocol)
- `tinygrad/llm/decode_routes.py` (`_Q6KDecodeCandidate.execute`, line 181)
- `tinygrad/llm/decode_kernels.py` (`_emit_q6k_coop`, `Q6KGEMVRouteSpec.validate`,
  `Q6K_COOP_ROW_TILE_BY_TARGET`)
- `tinygrad/llm/generated/decode-epilogue-fusion-route-policy.json` (M2 promotion record)
- `scratchpad/pg3_decode_rendered_source_equality.py` (render-equality control)
