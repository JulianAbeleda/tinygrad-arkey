# NV reduce-output stage 3 outcome record: P2 lean NO-GO, P1 per-row grid BOOKED

Date: 2026-08-13. Branch `nvidia-bringup-20260731` (HEAD `dc58ae57f`).

Status: **Stage-3 geometry split verdict. The P2 lean single-row 1_4096 launch
is a NO-GO (the package measured -631.6 / -672.5 us vs the bracketing
controls; the 1_4096 body alone went 7.97 -> 45.63 us/launch) and its
geometry is reverted. The P1 per-row grid for the multi-row q/k bodies is
BOOKED: the P1-only package clears the +50 us bar against BOTH bracketing
controls (+55.31 / +66.55 us, bracket median +60.93 us) with the exact
full-logit sha `6ec7227e...` and the 91-body census intact.** The scope's
premise holds only for the q/k site: llama's grid-per-row `rms_norm_f32`
geometry transfers, but the extrapolation to a ONE 32-lane block for the
4096-dim chain does not (llama uses 1024-thread blocks there; our 16-lane
serial chain in a single warp serializes the whole reduce).

## A/B evidence

Package bracket (P1 + P2 geometry), `nv-reduce-output-stage3-p3-geometry-ab-20260813.json`:
verdict NO-GO. Smoke / exact logits / census all PASS; wall bracket
candidate 5.9636 ms vs control A 5.3320 ms and control B 5.2911 ms
(-631.6 / -672.5 us). Census attribution: `reduce_output_rmsnorm_1_4096`
median 45.63 us/launch x 19 = +867 us of body mass vs the old 7.97 us
(+715 us over the old geometry), matching the bracket regression; the q/k
bodies measured 3.185 / 3.07 us/launch, slightly better than the old
3.70 / 3.17.

P1-only bracket (P2 reverted), `nv-reduce-output-stage3-p3-p1only-ab-20260813.json`:
verdict BOOKED. All five gates PASS (smoke survive 594 programs; exact
full-logit sha `6ec7227eb9a481...` identical both arms, token stream
`9e6664fd...`, prelude 13876; 91-body census 19/36/36 with zero weight
materializations; reverse wall bracket; predicted-wall-delta EXPLAINED).
Body medians: `reduce_output_rmsnorm_32_128` 3.245 us (old 3.70),
`reduce_output_rmsnorm_8_128` 3.07 us (old 3.17),
`reduce_output_rmsnorm_1_4096` 7.94 us (old geometry restored).

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control A | 5.276792 | 189.51 |
| candidate | 5.221482125 | 191.52 |
| control B | 5.288030 | 189.10 |
| bracket median | 5.282411 | 189.31 |

Candidate minus control A = **+55.31 us**, minus control B = **+66.55 us**,
minus bracket median = **+60.93 us**. Bar +50 us vs both controls ->
**PROMOTED**. tok/s translation: 191.52 vs 189.31 = **+2.21 tok/s**.

## Decision and scope limits

- The P2 lean 1_4096 geometry is reverted to the booked 16-warp body
  (commit `dc58ae57f`); the q/k per-row grid (P1) stays as the measured
  improvement. No production default changed: the route policy file is
  untouched (the harness control arm now constructs the closed route-less
  graph in-process, so no temporary policy close is needed).
- The stage-3 target remains open on the 4096 side: llama's 1024-thread
  tree-block geometry would NOT be bitwise-equal to our serial association,
  so the remaining q/k gap (ours ~3.2 us/launch vs llama 1.30) is the
  bounded path to the +147 us / ~198 tok/s target; the 4096 side stays at
  parity and is closed unless a bitwise-preserving wider geometry is found.
- The harness carries the predicted-wall-delta contract now
  (COST_PREDICTION + validate_cost_prediction, llama `rms_norm_f32` floors):
  the P1-only bracket reconciled EXPLAINED (measured -60.9 us vs
  llama-shaped -136.1 us; residual causes in-kernel critical path +
  activation traffic), and any future bracket that contradicts the
  llama-shaped envelope fails closed.

Raw artifacts: `docs/task_workflow/evidence/nv-reduce-output-stage3-p3-geometry-ab-20260813.json`
(package NO-GO), `docs/task_workflow/evidence/nv-reduce-output-stage3-p3-p1only-ab-20260813.json`
(P1-only BOOKED). The per-arm children and timing rows live under the
harness `--out` directories (`/tmp/nv-stage3-p3-*-20260813.*`) and are not
committed.
