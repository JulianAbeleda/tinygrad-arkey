# NV full ledger audit versus llama — post-campaign rebuild

> Superseded for the current tinygrad endpoint and device ledger by
> `nv-dense-ffn-composition-reopen-result-20260824.md`. This document remains
> the retained pre-Q6-unroll campaign comparison.

Date: 2026-08-24
tinygrad tested commit: `7a18a43ff`
GPU: RTX 5090, graphics 2790 MHz / memory 14001 MHz
Depth: 512

## Verdict

The fresh installed tinygrad endpoint is **4355.023 us/token = 229.620
tok/s**. It exceeds the 227 checkpoint by **50.263 us/token**, or **2.620
tok/s**. Llama parity is not reached: the retained same-binary llama authority
is **4048.325 us/token = 247.016 tok/s**, leaving **306.699 us/token** or
**17.396 tok/s**.

The old ledger displayed llama as `247.061 tok/s`; that did not reciprocate
its recorded `4048.3246 us/token`. This rebuild computes throughput directly
from latency and corrects it to `247.0158 tok/s`.

## Wall and device comparison

| metric | tinygrad current | llama retained | tinygrad - llama |
| --- | ---: | ---: | ---: |
| unprofiled wall | 4355.023 us | 4048.325 us | +306.699 us |
| throughput | 229.620 tok/s | 247.016 tok/s | -17.396 tok/s |
| profiled device union | 4187.750 us | 3888.240 us | +299.510 us |
| profiled node sum | 4191.232 us | 5011.035 us | -819.803 us |
| resident overlap | 3.482 us | 1122.329 us | -1118.847 us |
| nodes | 452 | 762 | -310 |

The tinygrad profile identity closes exactly:

```text
4191.232 node sum - 3.482 overlap = 4187.750 us union
```

Unprofiled wall and profiled union are deliberately separate authorities.
The earlier `wall - profiled union` construction mixed domains and falsely
suggested an `82.557 us` host opportunity. A same-token marker run measured a
median `4.576 us` outside the marked device interval after subtracting marker
submission cost. This ledger therefore makes no host-gap subtraction claim.

## Current tinygrad census

The composed replay is `32 + 64 + 128 + 228 = 452` nodes. Current promoted
route populations are:

| route | population |
| --- | ---: |
| Q4 gate/up four-warp vector | 36 |
| Q6 down packed-lane | 18 |
| ordinary Q4/Q4 K/V pair | 9 |
| shared-Q8 Q4/Q4 K/V pair | 9 |
| shared-Q8 mixed Q4/Q6 K/V pair | 8 |
| Q6 V direct | 10 |
| producer-owned K/V cache sink | 36 |
| native vocab argmax | 1 |

Largest disjoint program rows:

| pool | us/token | calls | us/call |
| --- | ---: | ---: | ---: |
| gate/up Q4 | 1284.352 | 36 | 35.676 |
| down Q6 | 535.744 | 18 | 29.764 |
| down Q4 | 363.808 | 18 | 20.212 |
| vocab main | 312.608 | 1 | 312.608 |
| O projection | 306.816 | 36 | 8.523 |
| Q projection, ordinary + shared | 302.144 | 36 | mixed |
| flash score | 238.944 | 36 | 6.637 |
| physical K/V projection producers | 237.568 | 46 | mixed |
| flash combine | 102.208 | 36 | 2.839 |
| Q/K norm+RoPE and cache sink | 138.464 | 72 | mixed |
| Q8 providers | 30.464 | 17 | 1.792 |
| native vocab argmax | 8.800 | 1 | 8.800 |

Pair producers are counted by physical launches in this table; paired kernels
each produce two logical K/V outputs.

## Movement during this ranked campaign

| accepted action | conservative booked recovery |
| --- | ---: |
| typed four-warp vector gate/up | 53.329 us/token |
| Q6 down packed lane map | 10.607 us/token |
| shared-Q8 mixed Q4/Q6 pair | 3.126 us/token |
| total new conservative booking | 67.062 us/token |

The conservative campaign endpoint is `4448.334 us/token = 224.803 tok/s`,
still `43.048 us` short of 227. The fresh installed endpoint is faster and is
the current performance authority, but same-session candidate bookings remain
separate from cross-session endpoint movement.

## Remaining gap

| target | latency target | recovery from fresh tinygrad |
| --- | ---: | ---: |
| 227 tok/s | 4405.286 us | already ahead by 50.263 us |
| 240 tok/s | 4166.667 us | 188.357 us |
| retained llama | 4048.325 us | 306.699 us |

The remaining gap is overwhelmingly device residence. The largest current
surfaces are gate/up, down, flash, vocab, and projection bodies/boundaries.
Closed constructions must not be retried without new information: flash
single-stage and wider combine, exact-order vocab shared staging, row-packed
Q6 down, and tested queue-placement variants all have complete negative gates.

Evidence is under
`docs/task_workflow/evidence/nv-ranked-parity-campaign-20260824/`, especially
`07-current-default-wall-r15.json`, `07-current-default-profile.json`, and
`07-final-ledger.json`.

Verdict: `FRESH_DEFAULT_229_620_TOK_S_227_PASSED_LLAMA_GAP_306_699_US`.
