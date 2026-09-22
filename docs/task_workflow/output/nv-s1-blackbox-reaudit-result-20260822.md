# NV S1 “launch black box” re-audit (2026-08-22)

## Verdict

**MEASURED:** The claimed `~748 us` launch/timeline black box is refuted by two accounting errors. The NCU table sums **single-invocation** deltas as though they were full-token totals, and the S1 window uses llama O **grid start** as its endpoint even when PDL starts O early and leaves it waiting.

**MEASURED:** After correcting only invocation cardinality, the eight covered NCU rows sum to **400.700 us in the counter domain**, not 37 us, while still omitting many production invocations and support kernels.

**MEASURED:** After replacing llama's PDL-sensitive S1 boundary with its wait-free PDL-off trace, the S1 gap is **402.992 us**, decomposed as **379.963 us kernel-residence difference + 23.029 us idle-gap difference**. The arithmetic residual is zero.

**INFERRED:** The real still-unattributed device remainder is approximately **139-146 us**, not 748 us. A wait-exit trace is useful to validate semantic-ready boundaries, but it is no longer the first experiment needed to discover a 748-us mechanism.

## Finding 1 — the 37-us counter sum has the wrong unit

**MEASURED:** `docs/task_workflow/evidence/nv-catch-llama-ledger-20260822/04-row-authority.json:6-9` labels isolated counter-replay duration as `us per token`. Each bridge record launches one cubin once with one layer's buffer sizes. The result report then adds the per-launch deltas at `docs/task_workflow/output/nv-catch-llama-ledger-result-20260822.md:36-49` without multiplying by the full-token kernel census.

**MEASURED:** Production cardinality from `probe2-tinygrad-capture.json` changes the covered sum:

| Label | Covered symbol | Delta per launch us | Count/token | Cardinality-corrected delta us/token |
| --- | --- | ---: | ---: | ---: |
| MEASURED | gate/up | 2.47 | 36 | 88.92 |
| MEASURED | Q G3 subset | 2.27 | 19 | 43.13 |
| MEASURED | O | 1.72 | 36 | 61.92 |
| MEASURED | V Q4 G3 subset | 1.12 | 9 | 10.08 |
| MEASURED | V Q6 four-warp subset | 0.41 | 10 | 4.10 |
| MEASURED | down Q6 subset | 3.30 | 18 | 59.40 |
| MEASURED | vocab main | 22.63 | 1 | 22.63 |
| MEASURED | flash combine | 3.07 | 36 | 110.52 |
| MEASURED | **covered sum** | | | **400.70** |

**MEASURED:** This still excludes 17 Q-warp launches, all 36 K launches, 17 other V launches, 18 down-Q4 launches, all 36 flash-score launches, every Q/K/V completion reduction, q/k norms, rope/store, providers, and the vocab tail.

**UNMEASURED:** The 400.700-us value is not a wall result. NCU replay changes cache/timing, and the bridge itself records numeric replay equality as unmeasured. It is nevertheless decisive evidence that 37 us cannot be subtracted from a per-token union gap.

## Finding 2 — PDL makes the S1 endpoint non-invariant

**MEASURED:** The S1 tool defines exposure as `O.grid_start - Q.grid_end` and sets `llama_overlap = body_mass - exposure` at `extra/llm_research/decode/nv_s1_body_gap_split.py:26-42,67-81`.

**MEASURED:** Across the same 36 logical layers:

| Label | Oracle mode | Sum of `O.grid_start - Q.grid_end` us |
| --- | --- | ---: |
| MEASURED | PDL on A | 515.015 |
| MEASURED | PDL on B | 517.899 |
| MEASURED | PDL off | 749.258 |

**MEASURED:** PDL therefore shortens this timestamp-defined window by about **232.801 us**, while changing full-token resident union by only **+7.705 us** and DRAM traffic by 4352 bytes.

**INFERRED:** The missing window time is boundary migration: O becomes resident before its input is ready and waits. Work moving outside `Q.end -> O.grid_start` is not work removed from the device union. Consequently, the original 326.134-us `llama_overlap` term is a residence-multiplicity identity, not a causal wall saving.

## Finding 3 — corrected S1 ledger

**MEASURED:** Using the original tinygrad trace and the wait-free oracle trace:

| Label | Side | S1 exposure us | S1 kernel mass us | S1 idle gap us |
| --- | --- | ---: | ---: | ---: |
| MEASURED | tinygrad | 1152.250 | 1042.500 | 109.750 |
| MEASURED | llama PDL off | 749.258 | 662.537 | 86.721 |
| MEASURED | tinygrad − llama | **402.992** | **379.963** | **23.029** |

**MEASURED:** The kernel-mass difference further closes as:

| Label | S1 family comparison | Delta us |
| --- | --- | ---: |
| MEASURED | K/V GEMVs | +83.741 |
| MEASURED | flash score | +103.802 |
| MEASURED | flash combine | +68.693 |
| MEASURED | q/k norm group | +104.637 |
| MEASURED | generated `E_*` group versus oracle rope/store | +36.884 |
| MEASURED | generated `r_*` group versus oracle quant group | -17.794 |
| MEASURED | **sum** | **+379.963** |

**INFERRED:** The last two mappings are adjacency-based group comparisons, not final semantic names; role metadata must confirm them before changing code. Their aggregate arithmetic is exact.

## Finding 4 — the full-token row ledger already names most of the union gap

**MEASURED:** `04-row-authority.json:136-145` retains nine full-token semantic row deficits totaling **646.838 us**. These are gate/up, complete Q, O, down, vocab tail, flash combine, flash score, complete K, and complete V.

**MEASURED:** Against the reported PDL-on resident-union gap of 785.542 us, those disjoint profile rows leave **138.704 us**. Against the PDL-off union gap of 793.246 us, they leave **146.408 us**.

**INFERRED:** Those row values are not additive recovery promises because alternate paths and implementations interact. They are, however, valid evidence that the current device gap is largely named kernel residence, not an unnamed 748-us launch bubble.

## Finding 5 — flash wording and comparator are inconsistent

**MEASURED:** With the fresh isolated comparator `4.160 us tinygrad` versus `3.744 us llama`, tinygrad is **0.416 us slower**, not faster. Across 36 calls, body accounts for **14.976 us**, leaving the stated **49.564 us** from an installed 64.540-us row delta.

**MEASURED:** `04-row-authority.json:128-132` instead records llama as 4.480 us while retaining the 49.56-us residual. If 4.480 were used, the corresponding residual would be 76.060 us. These two comparator versions cannot share one residual.

## Finding 6 — the exact host advantage is not measured

**MEASURED:** `08-host-gap-ledger.json` explicitly constructs `-72.016 us` by subtracting imported profile unions from later unprofiled wall arms. `10-corrected-wall-ledger.json` labels this cross-domain.

**INFERRED:** Nearby runs support the direction that tinygrad's host term is not the primary 713-us loss. The exact `-72.016 us` magnitude and the statement that the entire gap is device time remain cross-domain inferences, not a paired same-token measurement. Zero residual is guaranteed after defining `host_gap = wall - imported_union`; it is not an independent closure test.

## Correct next measurements

1. **MEASURED prerequisite:** Correct the counter artifact units to `us per invocation`, attach production invocation counts, and separate counter-domain duration from installed-timeline duration.
2. **MEASURED prerequisite:** Rebuild every inter-anchor window with semantic-ready endpoints. For PDL consumers, use `wait_exit`, not `grid_start`; in the interim, use PDL-off as the wait-free oracle.
3. **INFERRED priority:** Build one exhaustive disjoint full-token role table that sums every tinygrad and llama-PDL-off kernel exactly once. Start with the approximately 139-146-us remainder: missing reductions/providers, q/k norm, rope/store, vocab tail, and true GPU-idle intervals.
4. **INFERRED priority:** Reopen gate/up, O, Q, down, and flash-combine as installed-row targets. The counter deltas are small per invocation but repeat 18-36 times; “near parity per call” is not a full-token closure.
5. **UNMEASURED validation:** Add wait-exit timestamps to PDL-on llama to prove that `O.wait_exit` reproduces the PDL-off semantic boundary. This names spin placement; it should not be sold as a new 748-us optimization ceiling.
6. **UNMEASURED mechanism tests:** For each large installed row, compare hot isolated, cold predecessor-conditioned, and production-timeline duration in one clock domain. This separates body, cache state, and launch-to-start gaps without subtracting NCU replay from wall.

## Bottom line

**INFERRED:** You are stuck because two ledgers accidentally make repeated per-layer cost disappear: one by omitting invocation counts, the other by ending S1 at an early-started waiting consumer. Correcting those choices moves the diagnosis back from “mysterious scheduler shadow” to mostly named repeated kernel/support residence, with a much smaller approximately 140-us remainder worth tracing.
