# Ranked wall tests and promotion result

## Decision

The ranked test-first pass produced one admissible token-wall win: extend the
installed shared-Q8 attention route through block 25. The current composed
reverse bracket recovered 23.738 us/token, and the corrected prompt-prefilled
semantic gate passed. The extension is now part of the default route policy.

The other four targets did not justify production investment under the current
exactness and topology contracts. These are measured closures of the tested
mechanisms, not claims that the corresponding ledger regions can never improve.

## Ranked results

| Rank | Territory and tested theory | Information obtained | Token-wall result | Disposition |
|---:|---|---|---:|---|
| 1 | Flash score/combine: exact single-stage and legal geometry sweep | All 432 legal geometries were numerically valid; the fast variants changed reduction order and failed exact token identity. The exact production-compatible variant had already bracketed neutral. The score-to-combine edge has no measurable idle delay. | No positive exact candidate | Closed for the tested exact geometry/fusion mechanisms |
| 2 | Ordinary Q/O: wider CTA/vector loads and Q8 service-rate translation | Weight bytes match llama. Q8/DP4A raises the service rate by changing the instruction grammar, but a standalone Q8 producer costs more than the consumer saves. Four-warp FP16 raises instruction count and loses. | Included Q8 path loses in both hot and cold gates | Closed for standalone packing and tested CTA widening; shared/fused production remains the admissible form |
| 3 | Native RMSNorm: retain each lane's inputs through reduction | Direct CUDA improved by 0.039 us/call without spills, but the full-token reverse bracket lost 7.114 us. | -7.114 us recovery | Rejected; research switch remains default-off |
| 4 | Quantized vocabulary: exact shared staging/service rate | The path streams the material weight bytes near the measured comparison rate. Exact-order staging had already lost 183.530 us; the structural tail was previously removed by native argmax. | No credible positive exact candidate | Closed for tested staging; future work needs byte or numerical-contract change |
| 5 | Shared-Q8 coverage: re-test block 25 on the completed producer/direct-output substrate | Prompt-prefilled logits and token decisions passed the route's semantic thresholds. The reps-9 C/C/C reverse bracket beat both controls. | **+23.738 us/token** | **Promoted** |

## Shared-Q8 admission evidence

The valid semantic run used a prompt prelude so the KV state was populated.
Across eight decode frames, tokens, argmax, top-10 membership, and top-10 order
matched. Relative L2 was 0.000367, maximum absolute error was 0.00889, and the
perturbation-to-margin ratio was 0.0801; all are inside the existing admission
contract.

The wall bracket was:

| Arm | Median token wall |
|---|---:|
| Control A | 4.175240 ms |
| Candidate | 4.148657 ms |
| Control C | 4.169550 ms |
| Control midpoint | 4.172395 ms |
| Booked recovery | **23.738 us/token** |

The standalone post-landing run measured 4.166708 ms/token, or 239.998 tok/s.
That run is the honest observed endpoint. Session drift means it must not replace
the matched bracket for causal attribution. Applying the booked recovery to the
previous canonical endpoint projects about 241.2 tok/s, but that is a projection,
not a new endpoint claim.

Against the current llama reference of 4.021721 ms/token (248.711 tok/s), the
observed endpoint is 144.987 us/token, or 8.713 tok/s, behind. The new win closes
about one sixth of that observed time gap.

## Refreshed device ledger

The post-landing device profile accounts for 4,029.536 us of node duration and
4,029.000 us of interval union, with only 0.536 us of overlap. Its host/profiler
wall field is not an endpoint measurement and is deliberately excluded.

| Region | Device time | Calls or route count |
|---|---:|---:|
| Gate/up GEMV | 1,272.592 us | dense FFN rows |
| Down Q6 | 491.584 us | dense FFN rows |
| Down Q4 | 362.000 us | dense FFN rows |
| Vocabulary | 313.936 us | 1 |
| O projection | 304.976 us | attention rows |
| Native norms | 232.112 us | 55 |
| Flash score | 230.112 us | attention rows |
| Ordinary Q | 148.880 us | 18 |
| Shared Q | 148.336 us | 18 |
| Flash combine | 102.192 us | attention rows |
| Q/K head work | 134.288 us | paired totals |
| Shared Q4/Q4 KV pair | 51.856 us | 10 routes |
| Shared Q4/Q6 KV pair | 48.256 us | 8 routes |
| Ordinary KV pair | 46.784 us | 8 routes |
| Shared-Q8 providers | 31.392 us | 18 |
| Native argmax | 8.480 us | 1 |

The installed route census is now ten shared Q4/Q4 blocks, eight shared Q4/Q6
blocks, and eight ordinary blocks. Block 25 moved from ordinary to shared Q4/Q4.

## Verification

The focused shared-Q8 landing and boundary suite passed 25 tests. The broader
norm, shared-Q8, qualification, and graph suite passed 52 of 53 tests in one
combined process. The sole failure asserted that an untouched KV-cache slot was
zero; the exact test then passed in isolation, identifying test-state/cache
initialization contamination rather than a route regression.

## Evidence

- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/shared-q8-extra25-semantic-valid.json`
- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/shared-q8-extra25-wall-r9.json`
- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/post-landing-endpoint.json`
- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/post-landing-ledger.json`
- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/norm-retain-input-cuda.json`
- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/norm-retain-input-wall-r9.json`
