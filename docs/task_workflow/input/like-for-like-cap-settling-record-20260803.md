# Like-for-like cap settling record - Scope D measured quantize-excluded GEMV-class comparison

Date: 2026-08-03
Status: measurement record. Authorized by
`decode-gemv-efficiency-forward-scope-20260803.md` section 6 (Scope D) and section 8.2,
building on the evidence basis in `decode-gap-per-target-lever-scope-20260802.md` section
8.1: run the settling check - a measured quantize-excluded comparison of the GEMV-class on
both sides (a DEBUG=2 trace of our GEMV class and a node-filtered llama trace excluding
`quantize_q8_1`) on the RTX 5090 / sm_120 at d512 with Qwen3-8B-Q4_K_M - and state the
verdict on the ~0.92 ms cap. Measurement evidence only; no implementation code changed
(one new harness under `extra/llm_research/`, see section 2). Branch boundary: tinygrad
`nvidia-bringup-20260731` at `499bf4f5f` (the task-stated boundary `d990ff87d` moved during
this session; L4 vocab fusion landed ahead of it in `146c42622` + `499bf4f5f`).

## 1. The claim being settled

decode-gap scope section 8.1: the GEMV (non-vocab, incl quantize) class hides an asymmetry
in llama's favor - its 3.72 ms includes `quantize_q8_1` (0.482 ms across 217 nodes), which
tinygrad does not pay. Excluding quantize from both sides: llama's bare GEMV class is
3.543 - 0.304 (vocab) = **3.24 ms** vs ours 4.16 ms, i.e. **~0.92 ms of headroom** (the
cap). That cap bounds combined L2 + L5 GEMV-class claims. The settling check is the
measured quantize-excluded comparison this record runs. Per the scope, this record does
not claim L5 mass: the L5 lanemap lanes sweep was a SUBSTRATE verdict (flat at
lanes=32/64/128), and there is no live values-only mass in the q4k class beyond the
already-saturated coop-down residual.

## 2. Protocol

- Harness: `extra/llm_research/decode/gemv_class_census_nv.py` (new file, [test] commit).
  No existing harness classifies the GEMV-class: the flash probe
  (`flash_score_tile_nv_timing.py`) reports only the two flash kernels and
  `kernel_log_diff.py` only post-processes logs. The new harness captures the DEBUG=2
  prime token at a fixed depth exactly like the flash probe (prefill at DEBUG=0, then
  decode tokens 1-3, `marks[0]` = prime end), classifies kernels by name into GEMV-class
  vs everything else, and reports the class sum plus the house pins.
- Class rule (matches decode-gap scope section 8.1 and campaign section 14.3): GEMV-class =
  `q4k_g3_lanemap_gemv_*` + `q6k_gen_coop_*` + `q6k_gen_partial_*`, excluding the vocab
  head (`q6k_gen_coop_151936_4096*`, `q6k_vocab_scalar_reduce*`), the scatter chain
  (`E_1187*` / `r_*_1187*`), and quantize (none on our side; tinygrad consumes packed
  storage directly). Class sum = median x count per kernel (the house class convention,
  same basis llama's per-node avg uses).
- Session: same RTX 5090 / sm_120 box as the wall authority, driver 595.84, CUDA 13.2.
  All GPU runs serialized with `flock /tmp/nv_gpu.lock` and confirmed 0% util at lock
  acquisition. Fused prefill attention disabled
  (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`) in every decode run.
- Config: model `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, d512, max_context 4608
  (census harness) / 2048 (e2e control, matches prior records).
- Evidence classes: OBSERVED = measured in this session under the lock; INFERRED =
  arithmetic (subtraction) or prior-artifact attribution, flagged inline. Lifecycle
  vocabulary only.

## 3. Our side - GEMV-class census (OBSERVED, d512, HEAD `499bf4f5f`)

DEBUG=2 prime-token trace at d512. 1020 kernels/token, all-class kernel time 6095 us
(campaign census was 1021 kernels / 6187 us at the pre-fusion HEAD; the -1 kernel and
~92 us are the L4 vocab fusion landing: the vocab head renders as the single fused
`q6k_gen_coop_151936_4096_inkernel` and `q6k_vocab_scalar_reduce` is gone).

| kernel | n | median us | class |
| --- | ---: | ---: | --- |
| q4k_g3_lanemap_gemv_12288_4096 | 72 | 20.69 | GEMV |
| q4k_g3_lanemap_gemv_4096_4096 | 72 | 9.25 | GEMV |
| q4k_g3_lanemap_gemv_4096_12288 | 18 | 26.38 | GEMV |
| q4k_g3_lanemap_gemv_1024_4096 | 54 | 4.80 | GEMV |
| q6k_gen_coop_4096_12288_inkernel | 18 | 34.90 | GEMV |
| q6k_gen_partial_1024_4096_4 | 18 | 17.71 | GEMV |
| q6k_gen_coop_151936_4096_inkernel | 1 | 325.09 | vocab (excluded) |
| E_1187_32_4 (2 variants) / r_32_4_1187 / r_128_16_8_1187 / r_16_8 | 5 | 2.75/4.35/38.43/11.26/1.76 | scatter (excluded) |
| E_ / r_ plumbing | 690 | - | plumbing (excluded) |
| flash score/combine | 72 | - | flash (excluded) |

**GEMV-class sum: 252 kernels = 3.836 ms** (median x count; harness-reported
`gemv_class_us_sum` 3836 us). Fixed-depth tok/s 174.95 median (nmeas=20, reps=3).

## 4. llama side - quantize-excluded node-filtered trace (same shape/depth)

### 4.1 Same-session wall row (OBSERVED, CUDA build)

`llama-bench -m <model> -ngl 99 -fa 1 -p 0 -n 10 -d 512 -r 5 -o json` via
`/home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench` (CUDA build `ac4cddeb0`,
backends=CUDA, gpu_info=NVIDIA GeForce RTX 5090; GPU util peaked at 99% during the run):
**decode avg 245.50 +/- 12.69 tok/s**, samples [222.81, 250.53, 251.37, 251.37, 251.45],
median 251.37. This matches the parity record's same-session d512 llama row
(248.20 +/- 7.37, same CUDA build) within noise.

### 4.2 Per-kernel GEMV class (OBSERVED via reused CUDA-era node artifact)

Reused the prior campaign artifact `/tmp/llama_tg10_node.sqlite` (nsys
`--cuda-graph-trace=node` on the CUDA build, campaign 2026-08-02; TARGET_INFO_GPU =
NVIDIA GeForce RTX 5090 / GB202). Re-verified in this session that it reproduces the
campaign per-token node sums exactly: graphId 5, 762 nodes/token, total 4.774 ms; the
campaign doc section 14.3 lists exactly this node-sum as the d512 llama basis.

| kernel class | nodes | ms/token |
| --- | ---: | ---: |
| mul_mat_vec_q (incl vocab node) | 217 | 3.543 |
| quantize_q8_1 (excluded from both sides) | 217 | 0.482 |
| rms_norm_f32 | 145 | 0.308 |
| rope_neox | 72 | 0.127 |
| flash_attn_combine_results | 36 | 0.120 |
| flash_attn_ext_vec | 36 | 0.114 |
| k_set_rows / k_get_rows_float / k_bin_bcast | 39 | 0.079 |
| total | 762 | 4.774 |

Vocab node = the single mul_mat_vec_q node with avg 303.75 us (graphNodeId
21474837241). llama bare GEMV (quantize-excluded, vocab-excluded) =
3.543 - 0.30375 = **3.239 ms** (INFERRED by subtraction, same arithmetic as the claim).

## 5. The measured quantize-excluded comparison and verdict

| side | GEMV-class (quantize-excluded, vocab-excluded) | basis |
| --- | ---: | --- |
| tinygrad | **3.836 ms** (252 kernels) | OBSERVED, DEBUG=2 prime trace, d512, HEAD `499bf4f5f` |
| llama | **3.239 ms** (216 non-vocab mmq nodes) | OBSERVED per-node (reused CUDA-era artifact), 3.543 - 0.30375 |
| delta | **0.597 ms** | INFERRED arithmetic (3.836 - 3.239) |

The measured quantize-excluded delta is **~0.60 ms**, well under the **0.92 ms cap**
claimed in decode-gap scope section 8.1. The cap HOLDS with ~0.32 ms of margin. Note the
claim's "ours 4.16 ms" was the pre-row_tile=2 campaign number; at the current HEAD our
GEMV-class measures 3.836 ms (row_tile=2 + L4 fusion landed since), so the measured
headroom is smaller than the 0.92 ms estimate, not larger - the cap is not pressed.

Verdict on the cap's role:

- **L2 (Scope A) is CLOSED NO-GO** (l2-q6k-partial-singlepass-measurement-record,
  SUBSTRATE, no legal decomposition clears the llama-class floor). It consumes none of the
  cap.
- **L5 lanemap is SUBSTRATE-closed** with no live values mass (lanes flat at 32/64/128,
  no values-only knob remains). This record does NOT claim L5 mass and adds no recovery
  numbers across items.
- The cap's remaining role is bounding future structural GEMV-class claims: any new
  structural variant must stay under the measured ~0.60 ms quantize-excluded delta when
  its claims are combined, and its own isolated same-session measurement replaces this
  bound.

## 6. Controls (all re-run in this session)

- Fixed-depth token sha256 (census harness, nmeas=20, reps=3, d512, fused prefill
  disabled): `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` **3/3**.
- First token (same reps): **151936 3/3**.
- Decode sha256 (`model_e2e_bench.py`, d512 prefill, 96 decode tokens):
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`; first-token ids
  start `50994, 82, 31109, 3508, 692, 2, 11162, 100, ...` (matches section 8.2 pin).
- Census row: `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f` - matched by both the
  census harness and the e2e control.
- pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU, no lock): all 10
  legacy hashes byte-identical to the pin table
  (312422c73a49 / 27857cb8ca03 / 851760e2053c / 39ddb717ddd4 / cc38fbb3db92 /
  5795e66a7292 / 344e1c388eeb / c708302aa2d2 / 66d4c4da3108 / c78e4651ad35), and the M2
  promoted fused row `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f` holds.

## 7. Deviations

- The llama HIP build (`/home/ubuntu/env/llama.cpp/build/bin/llama-bench`, backends=ROCm,
  gpu_info empty) falls back to CPU on this 5090-only box (~6.9 tok/s at d512). The rows
  from that build (`/tmp/llama_hip_d512.json`) were **discarded** after the parent review
  flag; the authoritative llama side here is the CUDA build `ac4cddeb0`
  (`build-cuda/bin/llama-bench`), verified backends=CUDA + gpu_info=NVIDIA GeForce RTX
  5090 in the JSON before use, per `nv-decode-parity-final-20260802.md:19` and
  decode-gemv-efficiency-forward-scope section 6.
- The llama per-kernel node data is reused from the CUDA-era campaign artifact
  `/tmp/llama_tg10_node.sqlite` (not a fresh nsys trace); reuse is authorized by the scope
  and the artifact re-verified exactly against the campaign node-sum in this session.
- The `model_e2e_bench.py` prefill arm errors (fused prefill attention ABI broken on NV -
  the known `PACKED_FRAGMENT_LOAD` UOp verification failure); the decode arm and all
  controls completed, matching every prior record's handling.
- HEAD moved from the task-stated `d990ff87d` to `499bf4f5f` (L4 vocab fusion landed)
  during this session; the record reports the actual measured HEAD.
- A parallel agent left `tinygrad/codegen/opt/postrange.py` in an unmerged (UU) state and
  modified `tinygrad/llm/decode_routes.py` in the working tree after all measurements
  completed; neither file was touched by this record, and neither is part of its commits.

## 8. HARD STOP

Nothing beyond this scope. The cap is settled as HOLDS (measured ~0.60 ms quantize-excluded
delta, under the 0.92 ms estimate) with L2 CLOSED NO-GO and L5 SUBSTRATE-closed; this
record claims no L5 mass and adds no recovery numbers across items. No promotion to
`dev`/`exp`/`master`, no push; the parent pushes after review.

## 9. References

- `decode-gemv-efficiency-forward-scope-20260803.md` sections 6, 7, 8 (Scope D, controls)
- `decode-gap-per-target-lever-scope-20260802.md` section 8.1 (evidence basis)
- `nv-performance-campaign-scope-20260801.md` sections 12-14 (class table, method)
- `nv-decode-parity-final-20260802.md` (wall authority, llama CUDA build spec, pins)
- `l2-q6k-partial-singlepass-measurement-record-20260803.md` (Scope A NO-GO)
- `l4-vocab-substrate-fusion-measurement-record-20260803.md` (row_tile=2, L4 values row)
- Harness: `extra/llm_research/decode/gemv_class_census_nv.py`
