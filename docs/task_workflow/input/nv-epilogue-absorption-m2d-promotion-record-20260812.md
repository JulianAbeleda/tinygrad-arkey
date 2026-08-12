# NV epilogue absorption M2d promotion record (NV sm_120, fp16 flash-combine store)

Status: **M2d PROMOTED. The fp16 flash-combine store is now default-on for
NV sm_120 decode via the `decode_flash_combine_route_policy.json`
`promoted_targets` entry. All correctness gates PASS (exact full-logit
SHA-256 `70838f5237ce2cf2` byte-identical to control, census net -36
kernels, 594 vs 630, no opaque copy, no unrelated program shift), and the
production timing harness confirms the promotion is bit-exact and faster:
median 5.3625 ms/token = 186.48 tok/s vs the pre-promotion production run
5.4268 ms/token = 184.27 tok/s (+64.2 us/token), with the generated token
stream identical (`9e6664fd...`). The +50 us promotion bar from the M2d
wall bracket was waived by the principal: that record itself holds that
the correctness gates are the booking authority for the route and the +50
us bar is wall evidence only.**

Scope: `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`.
Wall bracket record: `docs/task_workflow/input/nv-epilogue-absorption-m2d-wall-bracket-record-20260812.md`.
Branch `nvidia-bringup-20260731`. Run 2026-08-12 on the RTX 5090
(sm_120, Qwen3-8B-Q4_K_M, fixed depth 512, count 8, 5 reps) under the
shared GPU bench lock via
`extra/llm_research/decode/nv_predispatch_full_logits_qualification.py`.

## Promotion mechanics

The route policy file `decode-flash-combine-route-policy.json` (schema
`boltbeam.route_policy.v1`, route `decode_flash_combine_fusion`) now lists
`{"backend": "NV", "architecture": "sm_120"}` in `promoted_targets`, the
same target format used by the other promoted files. Consumption is
unchanged: `decode_routes.py` `admission.combine_fusion_admitted` ->
`binding.combine_fusion` -> `combine_fp16` default-on for NV sm_120
decode. The `_flash_combine_fp16_lease` stays harness-only (used by the
bracket and qualification harnesses); promotion is what makes the fp16
store the default for production decode on this target.

## Verification evidence

| gate | pre-promotion production | M2d promoted | result |
| --- | --- | --- | --- |
| generated identical (5 reps) | true | true | PASS |
| token stream sha256 | `9e6664fd...` | `9e6664fd...` | PASS (bit-exact) |
| median wall | 5.4268 ms/token (184.27 tok/s) | 5.3625 ms/token (186.48 tok/s) | +64.2 us/token |
| spread | 0.931% | 0.806% | PASS |

Raw artifacts: `/tmp/nv-e2e-production-20260812.json` (pre-promotion,
sha256 `0a43da1b95ae9b1f`), `/tmp/nv-e2e-m2d-promoted-20260812.json`
(promoted, sha256 `1fa4b7a8fb9b6f05`).

The wall delta measured in production (+64.2 us) is larger than the
same-session bracket delta (+35.8 us). The bracket compares fresh-process
arms with a smaller wall window; the production harness runs the full
predispatch decode path. Both directions agree (strictly faster) and the
token stream is byte-identical, so the promotion is fail-closed on
correctness regardless of which wall number is cited.

## Gate evidence (from the M2d wall bracket record, all PASS)

| gate | control | candidate | result |
| --- | --- | --- | --- |
| NV render smoke | - | survive, fp16 combine present, no cast/copy/add | PASS |
| exact logits (32 rows, fp32 sha256) | `70838f5237ce2cf2` | `70838f5237ce2cf2` | PASS |
| census | 630 kernels, cast 36, combine f32 36 | 594 kernels, cast 0, combine f16 36 | PASS |
| wall bracket | median 5.2075 ms | 5.1717 ms | +35.8 us (below +50 us bar) |

## Decision and scope limits

Promotion is restricted to NV sm_120 only. Every other backend and
architecture keeps the legacy `flash_fused_gmax_combine_*` kernel, which
is byte-identical and untouched. No model wiring changed; no default flip
for non-promoted targets; the eager JIT=0 baseline stays arm-invariant by
harness construction (`_without_flash_combine_fp16`), so the wedge that
originally diverged the candidate cannot re-enter through promotion.

## Ledger translation

The M2d row is now LANDED and BOOKED (promoted): expected production
credit ~+35.8 us/token at the wall bracket, observed +64.2 us/token in
the production timing harness. Production decode is now ~186.5 tok/s at
d512 vs the 08-08 working agreement 183.0 tok/s. Remaining path toward
~194 tok/s (5.1546 ms/token): the norm-epilogue row (M1) and then the
quant GEMV row (~4050 us census; a 10% win lands ~203 tok/s).
