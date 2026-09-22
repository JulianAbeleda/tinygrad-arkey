# NV Q6 oracle-shaped CTA topology decision

## Question

Does llama-shaped 8-warp CTA ownership become an investable full-route lever
when the generated Q6_K kernel increases output-column reuse?

The screening bar was an exact, spill-free candidate whose normalized
single-CTA work projected at least `23.5 us` recovery from the current
`318.8 us` wide Stream-K main. A screening pass required an exact full-shape
170-owner measurement before promotion.

## Exact R9 CTA sweep

All six arms use canonical packed Q6_K weights, packed signed Q8 values,
power-of-two scales, an independent CPU reference, one 256-thread CTA, and
nine synchronized timing samples.

| K blocks | columns | median us | normalized us/(output*K) | registers | stack bytes | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 4.704 | 8.9722e-6 | 138 | 0 | feasible |
| 1 | 32 | 6.336 | 6.0425e-6 | 255 | 0 | screening winner |
| 1 | 64 | 14.656 | 6.9885e-6 | 255 | 1112 | reject: spills |
| 2 | 16 | 8.192 | 7.8125e-6 | 254 | 0 | fresh pinned control |
| 2 | 32 | 11.648 | 5.5542e-6 | 255 | 64 | reject: spills |
| 2 | 64 | 30.176 | 7.1945e-6 | 255 | 1408 | reject: spills |

The fresh pinned `128x16xK512` result is `8.192 us`, consistent with the
historical `8.176 us`. The feasible `128x32xK256` arm has a `0.7734375`
normalized ratio and therefore projected `72.228 us` recovery. It passed the
screening bar and triggered full-route integration.

## Full-shape 170-owner result

The integrated candidate preserves 170 CTAs. A 32-column tile expands the
full output grid to 512 tiles, so every owner needs up to four ordered tile
segments. The exact route uses 680 deterministic partial slots and compares
against an independent descriptor-driven slot producer.

| Measurement | Result |
|---|---:|
| compared values | 2,785,280 |
| output exact | true |
| tile IDs exact | true |
| maximum absolute error | 0 |
| R9 minimum | 851.360 us |
| R9 median | 852.704 us |
| current wide main | 318.800 us |
| measured recovery | -533.904 us |
| registers | 134 |
| shared memory | 58,880 bytes |
| stack / LDL / STL | 0 / 0 / 0 |

## Decision

`NO_GO_ORACLE_CTA_TOPOLOGY`

The single-CTA projection does not survive full-route scaling. Narrowing the
output tile improves isolated column reuse but makes each of the fixed 170
owners traverse four sequential output-tile segments. Each segment repeats
Q6 producer work, Q8 record staging, synchronization, and writeback. The
exact full route is `2.675x` slower than the current wide main and misses the
investment bar by `557.404 us`.

Do not promote the 32-column topology. The useful oracle constraint is the
opposite: retain broad output reuse per packed Q6 publication while reducing
decode/address instructions inside that broad tile. A future candidate must
be screened under saturated full-route ownership, not inferred from one-CTA
normalization alone.

## Evidence

- `docs/task_workflow/evidence/nv-q6-oracle-cta-sweep-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-cta-m32-full-20260831-r2/qualification.json`
- `extra/llm_research/prefill/bench_nv_q6_oracle_cta_sweep.py`
- `extra/llm_research/prefill/bench_nv_generated_q6k_streamk_owner_m32.py`
