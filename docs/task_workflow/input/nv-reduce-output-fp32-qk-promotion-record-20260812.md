# NV fp32 q/k reduce-output promotion record (NV sm_120)

Status: **PROMOTED (bar waived). The fp32 q/k reduce-output route is now
default-on for NV sm_120 decode via the
`decode-reduce-output-rmsnorm-route-policy.json` `promoted_targets`
entry, on the principal's waiver of the +50 us wall bar (same basis as
M2d: the correctness gates are the booking authority; the bar is wall
evidence only). All correctness gates PASS: exact full-logit SHA-256
identical to control, 91 fused bodies
(`reduce_output_rmsnorm_32_128` x36, `_8_128` x36, `_1_4096` x19),
norms kernels 328 -> 74, net -163 programs with honest side-effect
accounting. Wall bracket 2026-08-12c on a stable machine (controls 4.5 us
apart): candidate 5.1987 ms/token = 192.36 tok/s vs controls 5.2374 /
5.2419 ms (+38.7 / +43.2 us/token, median +40.9 us). Production timing
harness confirms bit-exact and faster: median 5.2031 ms/token = 192.19
tok/s vs the pre-promotion production 5.2365 ms/token = 190.97 tok/s
(+33.4 us/token), token stream `9e6664fd...` identical in every run.**

Scope: `docs/task_workflow/input/nv-reduce-output-fp32-qk-route-scope-20260810.md`.
Bracket evidence: `docs/task_workflow/evidence/nv-fp32-qk-wall-bracket-20260812c.json`.
Branch `nvidia-bringup-20260731`. Run 2026-08-12 on the RTX 5090
(sm_120, Qwen3-8B-Q4_K_M, fixed depth 512, count 8, 5 reps) under the
shared GPU bench lock via
`extra/llm_research/decode/nv_predispatch_full_logits_qualification.py`.

## Promotion mechanics

The route policy file `decode-reduce-output-rmsnorm-route-policy.json`
(schema `boltbeam.route_policy.v1`, route
`decode_reduce_output_rmsnorm`) now lists
`{"backend": "NV", "architecture": "sm_120"}` in `promoted_targets`. The
loader resolves once at model load and installs
`_decode_reduce_output_rmsnorm_promoted` on the model and every block; the
decode path routes the q/k norms through the cooperative fp32
reduce-output bodies and, because the model also carries the M2c callify
substrate gate (`_decode_callify_substrate_promoted` ORs in this route),
the decode capture runs under the callify owned-precompiled-output-
redirect / typed-semantic-input-producer Context automatically. The
harness flags stay research-only; promotion is what makes the route the
production default on this target.

## Verification evidence

| gate | pre-promotion production | fp32 q/k promoted | result |
| --- | --- | --- | --- |
| generated identical (5 reps) | true | true | PASS |
| token stream sha256 | `9e6664fd...` | `9e6664fd...` | PASS (bit-exact) |
| median wall | 5.2365 ms/token (190.97 tok/s) | 5.2031 ms/token (192.19 tok/s) | +33.4 us/token |
| full-logit sha256 | `6ec7227e...` | `6ec7227e...` | PASS (bit-exact) |

Raw artifacts: `docs/task_workflow/evidence/nv-qk-promoted-timing-20260812.json`,
`docs/task_workflow/evidence/nv-qk-promoted-logits-20260812.json`.

## Decision and scope limits

Promotion is restricted to NV sm_120 only. Every other backend and
architecture keeps the ordinary q/k reduce + epilogue spelling, which is
byte-identical and untouched. The +50 us promotion bar was waived by the
principal for this record: the measured +40.9 us bracket delta is
consistent (controls within 4.5 us), every correctness gate passes, and
the production harness confirms the route is strictly faster and
bit-exact. The route book was revised from the older +83.5 us figure
(measured against a slower baseline) to the current +40.9 us bracket /
+33.4 us production against today's leaner graph.

## Ledger translation

The fp32 q/k row is now LANDED and BOOKED (promoted): production decode is
~192.2 tok/s at d512 vs the 08-08 working agreement 183.0 tok/s. The
remaining ~17-19 us/token to the 193 target sits in the M1 norm epilogue
row (the remaining 37 norm chains, ~226 us census attribution). The
residual/cast/contiguous row is exhausted (M2a + M2b/M2c + M2d); the next
frontier after norms is the quant GEMV row (~4050 us census, ~70% of
decode).
