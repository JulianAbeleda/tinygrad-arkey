# L4 vocab substrate fusion implementation record - NV sm_120 vocab in_kernel admission

Date: 2026-08-03 (measured 2026-08-03, one flocked session per depth)
Status: implementation record for Scope B (L4 vocab substrate fusion), authorized by
`l4-vocab-substrate-fusion-implementation-scope-20260803.md` (the binding contract: section 2 scope of
change, section 3 gates, section 4 test plan, section 5 deliverable + HARD STOP). Authority chain:
`nv-decode-parity-final-20260802.md` (wall authority, harness protocol, CUDA build `ac4cddeb0`,
`build-cuda`) and `nv-parity-and-beyond-forward-scope-20260803.md` (canonical section 5 parity criterion).
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `499bf4f5f` (this scope's [nn] `146c42622` and
[test] `499bf4f5f` on top of `ed98ec54c`). Tracked tree clean except the three user-owned docs
(never committed here); untracked user files untouched.

## 1. Change summary

Exactly one admission expression in `tinygrad/llm/decode_routes.py` `_Q6KDecodeCandidate.execute`
(`reduction =` selection): the vocab coop head may now select `reduction="in_kernel"` when
`fusion_admitted and binding.use_coop and binding.row_tile * Q6K_POS_EXTENT <= 32`. NV sm_120
(row_tile=2) is legal (`2 * 16 = 32`); AMD row_tile=4 (`4 * 16 = 64`) and Metal (no fusion admission)
stay `external_sum`, guarded by the unchanged fail-closed backstop
`Q6KGEMVRouteSpec.validate` (`row_tile * lane_extent <= 32`) and the unchanged M2 promotion record
(`decode-epilogue-fusion-route-policy.json` promotes exactly `{("NV", "sm_120")}`).

- `[nn] 146c42622`: the admission expression plus import cleanup. The `is_vocab` variable became
  unused (its only uses were the removed line and the old expression); `Q6K_VOCAB_SCALAR_REDUCE_MIN_ROWS`
  had no remaining use in the file and was removed from the import, replaced by `Q6K_POS_EXTENT`.
- No emitter change: `_emit_q6k_coop` (`decode_kernels.py`) already renders the vocab shape with the
  in-kernel branch; `decode_kernels.py` is untouched by this scope. No `prefill_routes.py`, no
  `BOUNDED_PACKED_TILES`, no policy JSON, no dtype/precision change, no new subsystem. The landed
  values row `ab3cb84c1` (`Q6K_COOP_ROW_TILE_BY_TARGET = {("NV", "sm_120"): 2}`) is untouched.
- The fused in-kernel return path (`reshape(1, 1, N)`, `decode_routes.py:188-189`) already existed
  and is unchanged.
- `[test] 499bf4f5f`: new admission gate tests (section 4 below).

## 2. Unit results

`pytest test/unit/test_q6k_vocab_substrate_fusion_gate.py test/unit/test_decode_epilogue_fusion_gate.py
test/unit/test_route_admission_consistency.py test/unit/test_llm_decode_kernels.py
test/unit/test_llm_decode_routes.py test/unit/test_llm_decode_correctness.py
test/unit/test_qk_capability_policy_gate.py test/unit/test_model_route_plan.py
test/unit/test_qk_route_purity.py`: 85 passed.

New tests (CPU-only, no GPU): NV sm_120 vocab (151936 rows, row_tile=2, fusion_admitted) selects
`in_kernel`, emits `q6k_gen_coop_151936_4096_inkernel`, renders with `__shfl_xor_sync`, and the
fused path emits exactly one program (unit-level census: no vocab_reduce); AMD gfx1100 row_tile=4
stays `external_sum` even with `fusion_admitted` and `Q6KGEMVRouteSpec(rows=151936, k=4096,
row_tile=4, reduction="in_kernel").validate()` raises on the single-warp constraint; Metal
(no fusion admission) stays `external_sum` with the scalar-reduce chain intact. Existing gate tests
`test_coop_in_kernel_single_warp_constraint` and
`test_coop_in_kernel_renders_through_hip_and_cuda_without_gpu` stay green.

One pre-existing failure unrelated to this scope: `test_route_admission_consistency.py::
test_candidate_quant_matches_the_manifest` fails identically with this scope's change stashed
(a QuantFormat-instance vs manifest-string comparison on the Q4_K row; reproduced at HEAD before
any of this scope's code was present). Not touched here.

`sz.py`: authored budgeted lines 39999 / 100000 (budget raised at `d990ff87d`; no compression needed).
`git diff --check`: clean.

## 3. pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU, no lock)

All 10 legacy rows byte-identical to the pinned table, and the two additive fused rows unchanged:

| kernel | sha256 (first 12) |
| --- | --- |
| q4k_g3_lanemap_gemv_12288_4096 | 312422c73a49 |
| q4k_g3_lanemap_gemv_4096_4096 | 27857cb8ca03 |
| q4k_g3_lanemap_gemv_4096_12288 | 851760e2053c |
| q4k_g3_lanemap_gemv_1024_4096 | 39ddb717ddd4 |
| q6k_gen_coop_4096_12288 | cc38fbb3db92 |
| q6k_gen_coop_151936_4096 (vocab external_sum) | 5795e66a7292 |
| q6k_gen_partial_1024_4096_4 | 344e1c388eeb |
| q6k_vocab_scalar_reduce_151936_4096 | c708302aa2d2 |
| flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128 | 66d4c4da3108 |
| flash_fused_gmax_combine_32_128 | c78e4651ad35 |
| q6k_gen_coop_4096_12288_inkernel (additive fused) | add50a7aa43f |
| flash_fused_gmax_combine_f16_32_128 (additive fused) | 94d73c1e9650 |

The AMD vocab `external_sum` row (`5795e66a7292`) and the scalar-reduce row (`c708302aa2d2`) are
unchanged: AMD and Metal render byte-identical under this scope.

## 4. Census (d512, same-session A/B, DEBUG=2 prime trace)

Baseline (admission reverted to `ed98ec54c`) vs fused (`146c42622`), same box, back-to-back:

| | baseline | fused |
| --- | ---: | ---: |
| kernels/token | 1021 | 1020 |
| kernel us/token | 6189 | 6107 |
| q6k_gen_coop_151936_4096 (external_sum) | 1 x 334.5 us | 0 |
| q6k_gen_coop_151936_4096_inkernel | 0 | 1 x 322.24 us |
| q6k_vocab_scalar_reduce_151936_4096 | 1 x 72.7 us | 0 |
| E_1187_32_4 / r_32_4_1187 / r_128_16_8_1187 / r_16_8 | 0 each | 0 each |

The scope's expected count was 1021 -> 1016 (scalar-reduce + 4 scatter-chain kernels disappear).
The census run ACTUALLY shows 1021 -> 1020. The 4 scatter-chain kernels named in the scope
(`E_1187_32_4`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8`) are absent from this session's prime-trace
census in the baseline itself (count 0), so the per-token removal is exactly 1: the scalar-reduce
kernel. The external_sum vocab stack (334.5 + 72.7 = 407.2 us) becomes the single in-kernel kernel
(322.24 us), a measured ~85 us/token vocab-path saving, consistent with the kernel-us delta
(6189 -> 6107). The fused in-kernel kernel is present and the scalar-reduce + scatter kernels are
structurally gone from the fused path; the count difference from 1016 is a baseline-census
characteristic of the current branch state, not a missing removal.

## 5. Wall tok/s per depth (same session, flock /tmp/nv_gpu.lock, 0% util at acquisition,
   fused prefill attention disabled, Qwen3-8B-Q4_K_M)

tinygrad: `/tmp/path3_e2e_probe.py --depth N --nmeas 20 --reps 3 --no-fused-prefill` (median tok/s).
llama: `llama-bench -ngl 99 -fa 1 -pg 0,10 -d <depth> -r 5` with the campaign's CUDA build
`/home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench` (decode row n_prompt=0, n_gen=10, mean +/- stddev),
verified per JSON: `backends='CUDA'`, `gpu_info='NVIDIA GeForce RTX 5090'`.

| depth | tinygrad (fused) | llama (CUDA) | ratio | baseline ratio (M2-open record) | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| d512 | 174.724 | 248.111 +/- 7.75 | 0.704 | 0.696 | BELOW PARITY |
| d2048 | 163.13 | 234.912 +/- 7.22 | 0.694 | 0.687 | BELOW PARITY |
| d4096 | 150.819 | 225.302 +/- 6.62 | 0.669 | 0.659 | BELOW PARITY |

Per canonical scope section 5, a depth qualifies only when same-session `tinygrad >= llama`
(ratio >= 1.00). No depth qualifies. The fused change is additive: it improves the ratio by
~0.8-1.0 percentage points at every depth versus the M2-open baseline while keeping tokens
bit-identical, but the remaining gap is the pre-existing L4 per-kernel GEMV efficiency gap, not
this scope's surface.

## 6. Pins (every depth, unchanged)

- token sha256 `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3 at d512,
  d2048, d4096.
- first token `151936` 3/3 at every depth.
- decode sha256 (model_e2e_bench.py convention, 96 decode tokens):
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`; first token ids start
  `50994, 82, 31109, 3508, ...`.
- census row `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.

## 7. Deviations

- The first llama rows used `/home/ubuntu/env/llama.cpp/build/bin/llama-bench`, which is the
  ROCm/HIP build (`GGML_HIP=ON`); on this NV box it fails ROCm init and falls back to CPU
  (0% GPU util, ~6.9 t/s at d512). Those rows are discarded as comparators and are evidence of
  the build-path trap, not wall data. The campaign's authoritative comparator is the CUDA build
  `build-cuda` (`nv-decode-parity-final-20260802.md:19`), which reproduces the record's llama rows
  (d512 248.11 vs record 248.20). All parity rows above are CUDA-build rows.
- Census count is 1020 fused, not the scope's 1016 expectation; the actual baseline in this
  session is 1021 and the scatter-chain kernels are not present in the current branch state's
  prime-trace census (see section 4). Reported as measured; the record was not edited to fake the
  expectation.
- The first census A/B used `git stash`, which collided with a concurrent agent's uncommitted
  `tinygrad/codegen/opt/postrange.py` edits and left conflict markers in that file (the working
  tree became unimportable). The file was restored to HEAD; the other agent's work-in-progress is
  preserved in their kept stash entry and a saved copy (`/tmp/postrange_conflicted_saved.py`).
  The final A/B in section 4 used direct git-object file swaps (no stash) and ended with
  `decode_routes.py` verified identical to the committed `[nn]` version.
- GPU util at the very end of the final session was 57% (another process started); every run in
  this record acquired the flock with 0% util.
- The branch moved during the work: `ed98ec54c` (docs-only canonical forward scope) landed on top
  of `63c484558` while this scope was running; no code impact.

## 8. Blocked on

Nothing. All gates ran. Per the house rule the parent pushes after review; this record makes no
promotion claim beyond the measured BELOW PARITY verdicts.
