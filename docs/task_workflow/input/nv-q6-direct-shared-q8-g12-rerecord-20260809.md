# NV Q6 direct shared-Q8 g12 re-bracket (post-promotion HEAD)

Status: **WALL_NO_GO; closed-default lease remains unpromoted.**

Date: 2026-08-09
HEAD: `092656818b409d372183f51cdd642e7e6b943486` (branch
`nvidia-bringup-20260731`)

The shared-Q8 attention lease is promoted for NV sm_120
(`b762bb67`, `record-decode-shared-q8-attention-route-policy.json`,
post-promotion record-vs-open equality re-run PASS).  Section 5.1 of
`nv-gemv-substrate-landing-scope-20260808.md` authorizes ONE fresh
same-session g12 bracket at this HEAD with the identical settled protocol of
the 08-05 wall NO-GO record, re-running the Q6 V direct-output lease
(`SharedQ8AttentionAdmission.q6_direct_output`) against the cooperative-Q4
g12 shared route.  Promotion follows ONLY if the fresh wall is negative.

## Settled g12 reverse bracket

All arms used `d512`, 32-token uninterrupted windows, five repetitions, two
feedback captures (composed ping-pong), the cooperative-Q4 g12 shared route,
and identical stream hashes.  Flags: `--mode fused-timing --fused-groups 12
--cooperative-q4 --q6-direct-output --composed --settled-continuous --depth
512 --count 32 --max-context 1024 --groups 0 --reps 5`, Qwen3-8B-Q4_K_M,
DEV=NV, each GPU child under `flock -w 600 /tmp/gpu-bench.lock`.

| arm | ms/token |
| --- | ---: |
| partial-Q6 control A | 5.2942739375 |
| direct-Q6 candidate B | 5.32763603125 |
| partial-Q6 control A2 | 5.30643471875 |
| control midpoint | 5.300354328125 |
| B minus midpoint | **+0.027281703125 ms/token** |

The fresh wall is **+27.281703125 us/token slower** than the existing shared
Q8 partial route.  This is a real-token included-cost reverse bracket at the
post-promotion HEAD, so the route receives zero recovery credit.  Do not
promote the lease; the 08-05 NO-GO stands and is reinforced.

## Verification

Exact token stream hash identical across all arms:
`f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`
(`all_token_hashes_equal: true`).  A separate census child under the same
lease flags reported `q6_direct_consumer_count: 12` (expected 12),
`q6_direct_expected_count: 12`, `finite: true`, `q8_provider_count: 24`,
`cooperative_q4_consumer_count: 60`, and the harness's expected-count check
raised no error.

## Raw artifacts

- `/tmp/nv-q6-direct-20260809-bracket.json` (aggregate reverse-bracket result)
- `/tmp/nv-q6-direct-20260809-bracket/control-0.json`
- `/tmp/nv-q6-direct-20260809-bracket/candidate-1.json`
- `/tmp/nv-q6-direct-20260809-bracket/control-2.json`
- `/tmp/nv-q6-direct-20260809-census.json` (census + finite-logits child)
- `/tmp/nv-q6-direct-20260809-census.npz` (full logits)
