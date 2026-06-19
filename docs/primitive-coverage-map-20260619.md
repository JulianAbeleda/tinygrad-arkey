# Primitive coverage map - 2026-06-19

Purpose: PCG-0 result from `primitive-coverage-gap-scope-20260619.md`. This supersedes the stale parts of
`qk-machine-search-primitive-rows-20260618.md` by adding the latest decode integration diagnostic and prefill
transpose-free result.

Artifacts:

- `extra/qk_primitive_coverage.py`
- `bench/qk-primitive-coverage/rows.json`
- `bench/qk-primitive-coverage/summary.md`

No kernels were built, no routes were changed, and no defaults were touched.

## Verdict

The current primitive map is now explicit enough for the current Qwen3-8B/gfx1100 target.

There is no newly discovered cheap kernel. The missing coverage was mostly **row hygiene**: several important
lifecycle states were not represented in the old machine-search table, especially:

- decode B2 runtime/cache identity;
- decode MMVQ artifact/import discovery;
- prefill non-matmul overhead after the transpose-free Tensile refutation;
- long-context KV/attention as a separate target;
- serving, alternative quantization, CUDA portability, and tooling visibility as non-current-target rows.

## Priority Result

| priority | row | state | why |
|---:|---|---|---|
| 1 | `decode_mmvq_runtime_cache_identity` | open diagnostic | the only remaining bounded decode diagnostic after env knobs failed |
| 2 | `prefill_non_matmul_overhead` | open diagnostic | transpose-free Tensile is correct but `0.997x`; matmul is no longer the main prefill speed route |
| 3 | `decode_mmvq_artifact_import_family` | proposed | artifact/import could mirror the Tensile method, but only if a mature MMVQ family exists |
| 4 | long-context / serving / alternative-quant / CUDA rows | deferred / separate | target-regime changes, not current benchmark work |

## Decode

The decode map has two different classes of row:

1. **Current-target base decode.** `decode_mmvq_contract_preservation` is the real remaining large class:
   tinygrad `76%` standalone -> `~44%` in-model, while llama holds `57% -> ~54%`. The modeled target is `~1.187x`
   if `44% -> 54%` is recovered over the weight-GEMV bucket.
2. **Bounded next diagnostic.** `decode_mmvq_runtime_cache_identity` checks whether there is a wiring/cache/program
   identity mismatch before declaring the whole route renderer/scheduler or artifact/import only.

The q8 artifact route remains real but small: `decode_q8_artifact_lifecycle` is `pass_research_small`, with
`~5-6%` decode movement and dNLL inside gate. It is not the main parity path because reuse is capped at `2` and native
ownership is project-level.

Spec decode remains project-level closed for bounded builds. The PMU framing is right, but current verify is not
T-cheap.

## Prefill

The row map changed materially with the transpose-free result:

- `prefill_tensile_artifact_route` is now `refuted_for_e2e_speed`, not a live speed route. The extracted kernels are
  correct and fast, but as-built route is `0.999x` and transpose-free route is `0.997x`.
- `prefill_non_matmul_overhead` is the new active diagnostic row. If the matmul kernel is already around
  Tensile-class speed in-model, the remaining prefill gap is the non-matmul dilution: attention, norms, residuals,
  activation layout/casts, and lm_head.

That is a different conclusion from the old TPE-5 weighted model. TPE-5 remains valuable as backend-contract proof,
but not as a current e2e performance route.

## Deferred Target-Regime Rows

These are not implementation tasks unless the benchmark target changes:

- `long_context_kv_attention_lifecycle`: promote only after a clean ctx sweep proves KV/attention share and movement.
- `serving_overlap_scheduler_lifecycle`: requires an explicit serving workload/SLO.
- `alternative_quant_representation_lifecycle`: requires model-format and dNLL policy before kernels.
- `backend_portability_cuda_nvidia_lifecycle`: CUDA/NVIDIA needs its own audit; AMD refutations do not transfer
  blindly.
- `primitive_visibility_tooling`: supporting row only; it does not speed up inference by itself.

## Decision

Next work, if continuing decode, should be `decode_mmvq_runtime_cache_identity`.

Next work, if continuing prefill, should be a warm prefill non-matmul component atlas. Do not keep chasing fp16 GEMM
kernel swaps for pp512 unless that atlas contradicts the transpose-free result.
