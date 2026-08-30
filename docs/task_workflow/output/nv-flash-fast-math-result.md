# Flash fast-math translation result

## Decision

Program-scoped fast math is a real production Flash body win, but it is not a
strict token-wall pass yet. Keep the lease closed by default and classify the
branch as **body/graph proven, token wall unresolved**.

The earlier primitive estimate was too large because it mixed S6 and S8
geometry. The corrected equal-geometry result is much narrower: fast math
removes arithmetic work from the installed S8 score kernel, while the cold
kernel remains dominated by K/V load service. The production graph recovers
about 5 us/token, not the earlier roughly 16-us estimate.

## Why the primitive win is small when cold

The matched S8 counter comparison is:

| measure | ordinary | fast math | change |
|---|---:|---:|---:|
| dynamic instructions | 1,068,800 | 1,008,384 | -5.65% |
| registers/thread | 56 | 55 | -1 |
| cold DRAM bytes | 4,231,680 | 4,230,656 | effectively unchanged |
| cold duration under NCU | 6.752 us | 6.720 us | -0.47% |
| hot duration under NCU | 5.536 us | 5.312 us | -4.05% |
| cold long-scoreboard stall | 70.33% | 65.62% | still dominant |
| cold math-pipe throttle | 0.28% | 0.32% | not limiting |

Fast math removes instructions and one register, but it does not remove a K/V
load or a material byte. In the hot regime those arithmetic savings are
visible. In the cold regime the warps continue waiting primarily on dependent
loads, so much of the removed arithmetic was not on the elapsed-time critical
path. This is the same roofline distinction seen elsewhere in the ledger:
fewer instructions are useful only to the extent that instruction service,
rather than memory dependency, determines elapsed time.

## Production conversion

Three fresh graph profiles used the installed S8 route at depth 512. Each row
is the sum of all 36 layer instances per replay, with 61 steady replays per
arm.

| graph row | control A | fast math | control C | candidate versus control midpoint |
|---|---:|---:|---:|---:|
| score/PV bodies | 222.656 us | 217.632 us | 222.560 us | **-4.976 us/token** |
| combine bodies | 50.080 us | 50.208 us | 50.144 us | +0.096 us/token |
| all graph node service | 3,951.712 us | 3,946.464 us | 3,952.224 us | **-5.504 us/token** |

The score reduction repeats against both controls, and total node service
falls by the same amount. Graph overlap is approximately zero. Therefore the
win reaches the production graph and is neither swallowed by overlap nor lost
at the kernel-to-graph boundary.

## Why the token bracket did not close

The two unprofiled brackets are consistent with a signal smaller than their
inter-process drift, not with a missing production conversion:

- control/candidate/control, 144 timed tokens per arm: candidate was 8.783 us
  below the control midpoint, but 3.750 us above the faster control;
- candidate/control/candidate, 224 timed tokens per arm: candidate midpoint
  was 0.321 us below control, while the candidate endpoints differed by
  5.786 us.

The directly measured graph recovery is about 5 us/token, the same size as the
tighter bracket's endpoint spread and only about 0.12% of the full token. The
strict rule requiring the candidate to beat both flanks therefore fails even
though the on-path body delta is repeatable. This is a resolution wall, not an
overlap wall.

At the installed 4.094502-ms/token endpoint, a 4.976--5.504-us exact
translation would imply about **244.53--244.56 tok/s**, or **+0.30--0.33
tok/s**. This is a ceiling, not booked recovery.

## Promotion requirements

Do not enable the policy by default from these measurements alone. Fast math
changes floating-point contraction, division, reciprocal, square root, and
transcendental behavior; identical sampled token hashes and passing unit tests
are necessary but do not establish a general quality contract. Promotion
requires both:

1. a numerical/quality qualification broad enough for the dense-model
   contract, including logits or attention outputs rather than token IDs only;
2. a lower-noise same-process or device-timestamp endpoint experiment capable
   of resolving a roughly 5-us/token effect.

The closed research lease remains useful because it is program-scoped, changes
the source/cache identity, and leaves the default compiler path byte-identical.

## Evidence

- `docs/task_workflow/evidence/nv-flash-fast-math/primitive-ncu-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/bounded-s6-counters-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/profile-control-a.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/profile-candidate.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/profile-control-c.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/wall-r9.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/wall-cbc-r7.json`
- `extra/llm_research/decode/nv_flash_fast_math_wall.py`

