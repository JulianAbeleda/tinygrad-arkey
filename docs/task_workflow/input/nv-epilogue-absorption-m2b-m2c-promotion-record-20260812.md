# NV epilogue absorption M2b+M2c promotion record (NV sm_120, ffn-down residual add + block-output copy fold)

Status: **M2b+M2c PROMOTED. The ffn-down residual-add absorption is now
default-on for NV sm_120 decode via the
`decode-ffn-down-resadd-route-policy.json` `promoted_targets` entry, with
the M2c callify substrate enabled in the decode path. All correctness
gates PASS from the M2c AB record (exact full-logit SHA-256
`70838f5237ce2cf2` byte-identical to control, census net -85 kernels:
`E_32_32_4_02a9738c` 36 -> 0 and `E_32_32_4_fab82d40` 49 -> 0, all other
program counts byte-identical), and the production timing harness
confirms the promotion is bit-exact and faster: production timing median
5.2432 / 5.2365 ms/token = 190.72 / 190.97 tok/s vs a same-session
control (M2b/M2c record closed) 5.3721 ms/token = 186.15 tok/s
(+128.9 / +135.6 us/token, ABOVE the +50 us bar), with the generated
token stream identical (`9e6664fd...`) in every run.**

Scope: `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`
rows M2b/M2c. AB record:
`docs/task_workflow/output/nv-epilogue-absorption-m2c-ab-20260811.json`
(BOOKED). Branch `nvidia-bringup-20260731`. Run 2026-08-12 on the RTX
5090 (sm_120, Qwen3-8B-Q4_K_M, fixed depth 512, count 8, 5 reps) under
the shared GPU bench lock via
`extra/llm_research/decode/nv_predispatch_full_logits_qualification.py`.

## Promotion mechanics

The route policy file `decode-ffn-down-resadd-route-policy.json` (schema
`boltbeam.route_policy.v1`, route `decode_ffn_down_resadd`) now lists
`{"backend": "NV", "architecture": "sm_120"}` in `promoted_targets`. The
loader (`load_decode_ffn_down_resadd_promotion` in `model_route_plan.py`)
resolves once at model load and installs the flag on the model, every
block, and the Q4K/Q6K ffn_down linears after the primitive replacement
(the primitives are the objects that carry `route_role`). M2b: the
ffn_down GEMVs add the hidden-state residual h in-kernel
(`*_epi_ffnresadd` names), so the standalone fp32 add folds away 36x per
token. M2c: the declared epilogue-absorbing AFTER (fp32 block output)
gets its nested CALL rebound to the caller output slot, so the identity
copies fold away 49x per token; this requires the callify
owned-precompiled-output-redirect / typed-semantic-input-producer Context,
which production decode now applies via `_decode_callify_substrate()` when
any promoted policy needs it (M2b/M2c here; the reduce-output route
joins automatically if it ever promotes). The harness lease
`_ffn_down_resadd_lease` still forces the route where the record is
closed, so the AB control/candidate contract is unchanged.

## Gate evidence (from the M2c AB record, all PASS)

| gate | control | candidate | result |
| --- | --- | --- | --- |
| exact logits (fp32 sha256) | `70838f5237ce2cf2` | `70838f5237ce2cf2` | PASS |
| census | `E_32_32_4_02a9738c` 36, `E_32_32_4_fab82d40` 49 | both 0, all other counts identical | PASS (net -85) |
| wall bracket | median 5.2757 ms = 189.55 tok/s | 5.2031 ms = 192.19 tok/s | +72.6 us (ABOVE +50 us bar) |

## Decision and scope limits

Promotion is restricted to NV sm_120 only. Every other backend and
architecture keeps the legacy standalone-add spelling, which is
byte-identical and untouched. No model wiring changed for closed targets.
The fp32 q/k reduce-output route bracket re-ran today (2026-08-12) and
measured NO-GO (candidate 5.3089 ms vs controls 5.3531/5.2435 ms), so
`decode-reduce-output-rmsnorm-route-policy.json` stays empty and the
callify substrate is enabled only by the M2b/M2c record.

## Ledger translation

The M2b/M2c rows are now LANDED and BOOKED (promoted): expected production
credit ~+72.6 us/token at the wall bracket, observed +128.9 / +135.6
us/token in the same-session production timing harness (the production
delta includes the queue-depth hide of the folded copies). Production
decode is now ~190.7-191.0 tok/s at d512 vs the 08-08 working agreement
183.0 tok/s. Remaining parked rows: the fp32 q/k norm route (re-bracketed
NO-GO 2026-08-12, policy still empty) and the FFN-down owned-boundary
layer-8 wall win (held by the precision gate decision). M1 norm
absorption and the Q6 direct/FFN-down shared-Q8 rows are measured NO-GO
and stay closed.
