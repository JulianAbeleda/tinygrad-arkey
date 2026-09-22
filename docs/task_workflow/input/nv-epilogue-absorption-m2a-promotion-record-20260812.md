# NV epilogue absorption M2a promotion record (NV sm_120, fp16 w1w3 store)

Status: **M2a PROMOTED. The fp16 w1+w3 store spelling is now default-on for
NV sm_120 decode via the `decode-q4k-w1w3-fp16-store-route-policy.json`
`promoted_targets` entry. All correctness gates PASS from the M2a wall
bracket (exact full-logit SHA-256 `70838f5237ce2cf2` byte-identical to
control, census net -36 kernels, `E_128_32_3` 36 -> 0, no unrelated
program shift), the wall bracket measured +57.2 us/token ABOVE the +50 us
promotion bar, and the production timing harness confirms the promotion is
bit-exact and faster: median 5.3474 ms/token = 187.01 tok/s vs the
pre-promotion production run 5.3625 ms/token = 186.48 tok/s (+15.1
us/token), with the generated token stream identical (`9e6664fd...`).**

Scope: `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`
row M2a. Branch `nvidia-bringup-20260731`. Run 2026-08-12 on the RTX 5090
(sm_120, Qwen3-8B-Q4_K_M, fixed depth 512, count 8, 5 reps) under the
shared GPU bench lock via
`extra/llm_research/decode/nv_predispatch_full_logits_qualification.py`.

## Promotion mechanics

The route policy file `decode-q4k-w1w3-fp16-store-route-policy.json`
(schema `boltbeam.route_policy.v1`, route `decode_q4k_w1w3_fp16_store`)
now lists `{"backend": "NV", "architecture": "sm_120"}` in
`promoted_targets`, deliberately SEPARATE from the w1w3 fusion record
(`decode-q4k-w1w3-fusion-route-policy.json`, which stays NV-promoted for
the fused kernel itself). The loader (`load_decode_q4k_w1w3_fp16_store_promotion`
in `model_route_plan.py`) resolves once at model load and ANDs the fp16
store admission with the w1w3 fusion admission, so the fused16 spelling
can never fire where the fused kernel is closed. The harness lease
`_q4k_w1w3_fp16_store_lease` still forces the spelling where the loader
record is closed, so the AB control/candidate contract is unchanged.

## Verification evidence

| gate | pre-promotion production | M2a promoted | result |
| --- | --- | --- | --- |
| generated identical (5 reps) | true | true | PASS |
| token stream sha256 | `9e6664fd...` | `9e6664fd...` | PASS (bit-exact) |
| median wall | 5.3625 ms/token (186.48 tok/s) | 5.3474 ms/token (187.01 tok/s) | +15.1 us/token |
| spread | 0.806% | 0.723% | PASS |

Raw artifacts: `/tmp/nv-e2e-m2d-promoted-20260812.json` (pre-promotion,
sha256 `1fa4b7a8fb9b6f05`), `/tmp/nv-e2e-m2a-promoted-20260812.json`
(promoted, sha256 `efb5cd300bb939bb`).

The production wall delta (+15.1 us) is smaller than the same-session AB
bracket delta (+57.2 us). The bracket compares fresh-process arms with a
different control composition; the production harness runs the full
predispatch decode path where small-kernel launch overhead is partially
hidden by queue depth (the phase-6 evidence). Both directions agree
(strictly faster) and the token stream is byte-identical, so the
promotion is fail-closed on correctness regardless of which wall number
is cited.

## Gate evidence (from the M2a wall bracket, all PASS)

| gate | control | candidate | result |
| --- | --- | --- | --- |
| exact logits (32 rows, fp32 sha256) | `70838f5237ce2cf2` | `70838f5237ce2cf2` | PASS |
| census | cast `E_128_32_3` 36, fused fp32 36 | fused16 36, cast 0 | PASS (net -36) |
| wall bracket | median 5.3295 ms = 187.64 tok/s | 5.2723 ms = 189.67 tok/s | +57.2 us (ABOVE +50 us bar) |

## Decision and scope limits

Promotion is restricted to NV sm_120 only. Every other backend and
architecture keeps the legacy fp32-store fused kernel
(`q4k_g3_lanemap_gemv_w1w3fused_*`), which is byte-identical and
untouched. No model wiring changed for closed targets; the Q6K
consumer-side typed-view request already carries the same fail-closed
validator as Q4K, and the fp32 spelling remains the default wherever this
record is closed.

## Ledger translation

The M2a row is now LANDED and BOOKED (promoted): expected production
credit ~+57.2 us/token at the wall bracket, observed +15.1 us/token in
the production timing harness. Production decode is now ~187.0 tok/s at
d512 vs the 08-08 working agreement 183.0 tok/s. Remaining parked rows:
the fp32 q/k norm route (booked +83.5 us, policy still empty pending the
full-row bracket) and the FFN-down owned-boundary layer-8 wall win
(-4.28 us, held by the precision gate decision). M1 norm absorption and
the Q6 direct/FFN-down shared-Q8 rows are measured NO-GO and stay closed.
