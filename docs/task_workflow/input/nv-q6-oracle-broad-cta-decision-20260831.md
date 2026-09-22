# NV Q6 broad oracle CTA decision

## Question

Can a generated `128x128xK256`, eight-warp CTA reproduce the pinned llama
work unit closely enough to justify integration into the exact 170-owner
Stream-K route?

The investment bar was CPU-exact output, feasible shared memory, no stack or
local-memory traffic, and at least `23.5 us` projected recovery from the
current `318.8 us` main. A screening pass was required before full-route work.

## Matched package

The tested package uses canonical Q6_K and Q8_1 inputs and implements:

- one `128x128xK256` output work unit per 256-thread CTA;
- eight physical warps and llama's warp/lane output ownership;
- a 76-word expanded Q6 row in shared memory;
- one overwriteable 36-word-per-row Q8 panel;
- two fully unrolled K128 consumers;
- 64 persistent FP32 accumulators per lane;
- 16 preloaded Q6 fragments per K128 half;
- rolling pairwise IMMA folding;
- four lifecycle barriers; and
- an A/B for the second Q8 panel's 18-word register prefetch.

The launch uses `58,368 B` shared memory: `1,024 B` compiler static shared plus
`57,344 B` for the Q6 and Q8 arenas. This remains below llama's `58,880 B`.

## Initial lifetime control

The earlier broad generator materialized all 256 IMMA results before folding.
It was exact but not feasible:

| median | registers | stack | LDL | STL |
|---:|---:|---:|---:|---:|
| `31.456 us` | 255 | 2,944 B | 1,025 | 901 |

This proves broad ownership alone is insufficient; the rolling lifetime is a
required part of the oracle contract.

## Exact R9 results

One CTA:

| arm | median | exact | registers | stack | LDL / STL |
|---|---:|---:|---:|---:|---:|
| serial second panel | `13.856 us` | yes | 255 | 64 B | 18 / 16 |
| register-prefetched panel | `13.088 us` | yes | 255 | 0 B | 0 / 0 |

At 170 saturated CTAs, comparing `2,785,280` output values:

| arm | R9 median | exact | projected full main |
|---|---:|---:|---:|
| serial second panel | `19.744 us` | yes | `573.007 us` |
| register-prefetched panel | `18.688 us` | yes | `534.842 us` |

The prefetch is causal: it improves the saturated package by `1.056 us`
(`5.35%`), eliminates the stack and all local loads/stores, and reduces the
static instruction census from 4,056 to 3,560. It is not sufficient to make
the package competitive.

The projection holds the 170-CTA topology fixed, removes a `4 us` launch
floor, and scales the remaining one-unit cost by the exact
`6144/170 = 36.141176` K256 work units per owner. It is a screening projection,
not a substitute for a full-route measurement. Its margin is decisively
negative: the prefetch arm projects a `216.042 us` regression from the current
main, rather than the required `23.5 us` recovery.

## Static normalized comparison

| family per K256 body | generated prefetch | pinned llama |
|---|---:|---:|
| IMMA | 256 | 256 |
| LDSM | 32 | 32 |
| BAR | 4 | 4 |
| LDS | 172 | 176 |
| I2FP | 512 | 512 |
| FFMA | 512 | 640 |
| IMAD | 1,043 | 1,083 |
| PRMT | 26 | 80 |

The generated package now matches the oracle's principal structural counts
and is spill-free. Its remaining timing loss therefore cannot be attributed
to missing broad reuse, fragment count, barrier count, or panel-prefetch
lifetime. Static census similarity does not prove instruction scheduling,
dependency latency, or achieved issue overlap.

## Decision

`NO_GO_BROAD_CTA`

Do not integrate this package into the exact 170-owner route. The Q8 register
prefetch is a proven local transformation, but the complete package misses the
full-route investment bar. The next oracle comparison must account for the
physical instruction schedule and dependency chains inside one normalized
body, rather than adding more topology features already present in this test.

## Evidence

- `docs/task_workflow/evidence/nv-q6-oracle-cta-broad-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-broad-rolling-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-broad-rolling-170-20260831/result.json`
- `extra/llm_research/prefill/nv_q6_oracle_broad_cta.py`
- `extra/llm_research/prefill/bench_nv_q6_oracle_broad_cta.py`
