# NV gate/up four-warp regression closure (2026-08-23)

The Q4_K gate/up four-warp candidate regressed unprofiled wall by
`+28.36 us/token`. This run closes the regression against per-row
`PROFILE=1` device timestamps (control / candidate / control, token SHA
identical across all three arms).

## Findings

* [MEASURED] The regression is launch count, not body speed. The four-warp
  kernel body is bitwise identical and slightly *faster* than the installed
  fused control: `1352.384 us` vs `1359.424 us` per token, `-7.040 us/token`
  across 36 layers.
* [MEASURED] The candidate emits one extra `E_128_32_3` output-cast kernel
  per layer. Its node count is 632 vs 596 control nodes; the 36 added nodes
  are a per-layer `E_128_32_3` cast that the control's
  `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` absorbs in-kernel.
* [MEASURED] That cast costs `+39.520 us/token` (`~1.098 us/call`). Net
  gate/up swap is `+32.480 us/token` (`-7.040` body plus `+39.520` cast).
* [MEASURED] Downstream rows are flat. Down projections (`+6.24 us`),
  flash score/combine, Q/K/V projections, and norms all move `<= ~1 us`; the
  only other `>1 us` mover is the K/V cooperative kernel at `+6.064 us`.
  There is no cache/working-set damage spreading into the next layer.
* [MEASURED] The closure identity closes to `-2.472 us` (median-of-sums vs
  sum-of-medians noise), so the named rows account for the full node-sum delta.

## Closure table (profiled device domain, us/token)

| term | delta |
| --- | ---: |
| gate/up body (`four_warp` vs `w1w3fused16`) | -7.040 |
| added `E_128_32_3` cast | +39.520 |
| gate/up net | +32.480 |
| down Q4/Q6 | +6.240 |
| other rows | +7.344 |
| node_sum delta | +43.592 |
| overlap delta | +6.328 |
| union delta | +37.938 |

## Wall caveat

* [MEASURED] Profiled wall is noisy and is not the promotion signal. The
  first control arm measured `6949.84 us` against `6600.57 us` for the closing
  control (`349 us` spread), so the profiled `wall_delta_us = -164.883` is a
  first-arm-session artifact, not evidence the candidate is faster.
* [MEASURED] The authoritative wall remains the unprofiled reverse bracket:
  `+28.36 us/token` regression, or `~0.79 us/launch` for the 36 added casts.

## Verdict

The candidate stays `NO_GO_WALL`. The fix that would convert it is obvious and
small: absorb the `E_128_32_3` fp16 output cast into the four-warp kernel, as
the fused control already does. That would remove the `~28-39 us` cast cost
and leave the `~7 us/token` body advantage, a `~0.15%` win well below the
`+50 us` promotion bar and not a route toward 240 tok/s. No production model,
renderer, scheduler, runtime, or route code was changed.

## Evidence

* Driver result: `docs/task_workflow/evidence/nv-gateup-fourwarp-profile-closure-20260823/result.json`
* Per-arm profiles: `docs/task_workflow/evidence/nv-gateup-fourwarp-profile-closure-20260823/result/`
* Harness: `extra/llm_research/decode/nv_gateup_fourwarp_profile_closure.py`
