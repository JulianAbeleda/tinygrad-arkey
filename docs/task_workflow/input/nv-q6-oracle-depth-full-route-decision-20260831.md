# NV Q6 persistent-depth and full-route decision

## Decision

`INVEST_DEPTH_TEST_PASS`; `NO_GO_FULL_BROAD_ROUTE`.

Do not promote the broad oracle route. The exact scoped-dA marginal depth test
passes the agreed 5% investment threshold, but the actual 170-owner full route
is materially slower than llama and fails the existing model-reference
tolerance. The measured full route, not a single-CTA projection, is
authoritative.

## Contract

- Shape: `M=512, N=4096, K=12288`, Q6_K FFN down.
- Body: straight-line 69-load/35-store Q6 publisher, rolling Q8 panel prefetch,
  scoped dA grouping, signed IMMA and FP32 output accumulators.
- Depth model: one runtime K256 loop; do not multiply `T(1)`.
- Depths: `1, 2, 4, 8, 16, 36, 37`, each measured R31 against an independently
  generated CPU fixture.
- Full ownership: 170 owners over 6,144 tile/K256 work units; each owner has 36
  or 37 units and at most two output-tile segments.
- Reduction: two deterministic partial slots per owner and slot-ordered fixup.

## Persistent scoped-dA depth sweep

All seven depths are bit-exact to their CPU fixtures.

Robust Theil-Sen fit:

```text
T(depth) = 3.6846 us + 5.7417 us * depth
```

- OLS slope: `5.7525 us/K256`.
- Maximum robust-fit residual: `1.3943 us`.
- Llama gross marginal reference: approximately `5.57 us/K256`.
- Investment threshold at +5%: `5.8485 us/K256`.
- Scoped-dA slope versus threshold: pass by `0.1068 us/K256`.

The persistent composition exposes a resource problem hidden by depth one:

| depth | median | registers | stack | LDL / STL |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.032 us | 255 | 0 B | 0 / 0 |
| 2 | 15.424 us | 255 | 256 B | 64 / 128 |
| 4 | 26.656 us | 255 | 256 B | 64 / 128 |
| 8 | 49.600 us | 255 | 256 B | 64 / 128 |
| 16 | 95.552 us | 255 | 256 B | 64 / 128 |
| 36 | 210.368 us | 255 | 256 B | 64 / 128 |
| 37 | 216.128 us | 255 | 256 B | 64 / 128 |

The 64 loop-carried FP32 output accumulators are the spill bank. Reusing four
dA scratch names does not change ptxas allocation or the slope.

## Pressure and barrier causals

These are diagnostic only. No-dA is semantically excluded from promotion.

| arm | robust slope | stack at depth 37 | result |
| --- | ---: | ---: | --- |
| scoped dA + prefetch | 5.7417 us | 256 B | investment pass within 5% |
| dA + serial Q8 | 6.8038 us | 280 B | reject |
| dA + combined initial publication | 5.6739 us | 256 B | reject composition |
| no-dA diagnostic | 5.2064 us | 32 B | diagnostic only |

The prior depth-one dA win does not compose into a spill-free persistent body.
Serializing Q8 is substantially worse, and removing one publication barrier
does not remove the accumulator spill.

## Actual 170-owner full route

The first two-launch segment diagnostic was rejected because it destroyed
owner load balancing. The authoritative route inlines both possible owner
segments in one 170-CTA launch and performs one deterministic fixup.

| timing | R9 median | R31 median | llama | ratio vs llama |
| --- | ---: | ---: | ---: | ---: |
| main | 284.160 us | 285.600 us | 201.216 us | 1.4194x |
| fixup | 25.792 us | 25.600 us | 8.640 us | 2.9630x |
| pair | 310.016 us | 311.360 us | 209.856 us | 1.4837x |

The llama +5% gates are `211.2768 us` main and `220.3488 us` pair. The route
misses them by `74.3232 us` and `91.0112 us`, respectively.

Full-main SASS/resource facts:

- 255 registers and 288 B stack.
- 251 `LDL`, 377 `STL`.
- 512 static `IMMA`, 64 `LDSM`, 11 `BAR`.
- 8,328 static instructions.

The GPU fixup is bit-exact to a CPU reduction over the same deterministic slot
order. Against the existing full compiler reference, maximum absolute
difference is `0.1871337890625`, mean absolute difference is
`0.01368770468980074`, and the existing `rtol=2e-5, atol=2e-3` gate fails.
This is consistent with changed segmented FP32 association, but it is still a
route failure under the current correctness contract.

## Correction to prior ledger

The publisher ledger's prior no-go was based on multiplying a launch-dominated
one-CTA time across owner work. That projection is invalid and withdrawn.
Fitting persistent depth shows the body was worth an actual-route investment.
The measured full route independently restores the no-go conclusion with the
correct reasons: segment composition, spill traffic, fixup cost, and model
reference failure.

## Evidence

- `docs/task_workflow/evidence/nv-q6-oracle-depth-scoped-da-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-depth-no_da-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-depth-serial-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-depth-combined-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-full-streamk-single-launch-20260831/result.json`
