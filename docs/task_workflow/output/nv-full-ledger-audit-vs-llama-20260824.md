# NV full ledger audit versus llama

Date: 2026-08-24  
tinygrad HEAD tested: `dc03370a7`  
GPU: RTX 5090, pinned 2790 MHz graphics / 14001 MHz memory  
Depth: 512

## Verdict

tinygrad has not reached llama.  The fresh same-session wall result is
**4.503391 ms/token = 222.055 tok/s** for tinygrad and **4.048325 ms/token =
247.061 tok/s** for llama.  The measured parity gap is therefore **455.067
us/token = 25.006 tok/s**.

The immediate 227 checkpoint is close: the fresh wall needs **98.105
us/token**, while the conservative booked endpoint still needs **110.109
us/token**.  The 240 checkpoint needs **336.725 us/token** from the fresh wall.
Neither 240 nor fresh llama parity may be claimed from current candidates.

## Full wall reconciliation

| term | tinygrad | llama | tinygrad - llama |
| --- | ---: | ---: | ---: |
| unprofiled wall, us/token | 4503.391 | 4048.325 | +455.067 |
| profiled resident union | 4260.750 | 3888.240 | +372.510 |
| inferred host/outside-union gap | 242.641 | 160.085 | +82.557 |
| profiled node sum | 4272.176 | 5011.035 | -738.859 |
| resident overlap | 11.426 | 1122.329 | -1110.903 |

The closing identity is:

```text
wall delta = union delta + host-gap delta
455.067    = 372.510     + 82.557 us/token
```

Node sum is not the parity objective.  llama carries about 1.12 ms of
overlapped PDL wait residence, so its node sum is much larger while its union
and wall are smaller.  Treating the `-738.9 us` node-sum difference as a
tinygrad advantage would be a category error.

## Current tinygrad device census

The corrected current replay contains 462 nodes.  Its largest disjoint pools
are:

| pool | us/token | calls | us/call |
| --- | ---: | ---: | ---: |
| gate/up Q4_K | 1323.632 | 36 | 36.768 |
| down Q6_K | 549.792 | 18 | 30.544 |
| down Q4_K | 362.448 | 18 | 20.136 |
| O projection | 307.392 | 36 | 8.539 |
| vocab main | 312.576 | 1 | 312.576 |
| flash score | 240.416 | 36 | 6.678 |
| Q projections, ordinary + shared | 303.952 | 36 | 8.443 |
| 4096 RMS/reduction rows | 263.072 | 56 | mixed |
| flash combine | 102.496 | 36 | 2.847 |
| K/V projections, all installed forms | 201.056 | 36 | mixed |
| Q/K norm+RoPE and KV sink | 137.216 | 72 | 1.906 |
| activation epilogues | 54.176 | 38 | 1.426 |
| Q8 providers | 30.912 | 17 | 1.818 |
| vocab argmax/tail | 8.864 | 1 | 8.864 |

The current graph closes at `node_sum=4272.176`, `union=4260.750`, and
`overlap=11.426 us/token`.  The earlier parser's 910 ms result was invalid: it
concatenated many 462-node token replays.  The parser now recognizes the
current `32+64+128+238` replay grouping.

## Lever audit

1. **Kernel launch-shape variants are closed as a broad strategy.** The Q6
   packed lane map improved the profiled body by 13.848 us/token but was wall
   neutral.  Gate/up four-warp is 7.13 percent slower than the installed vector
   route.  Flash-combine width, K/V concurrency, and queue-placement probes
   are also closed no-go results.
2. **The main measured deficit is still device union.** Recovering 372.5 us of
   serialized residence requires either lower streamed bytes, a substantial
   fixed-byte rate improvement that survives cold production, or a new
   overlap construction with a causal wall result.
3. **Host/outside-union time has reopened as a secondary lever.** The fresh
   audit attributes 82.6 us of the parity gap to tinygrad's larger outside-
   union term.  This is an inferred aggregate, not yet assigned to submission,
   synchronization, graph replay, or measurement-boundary components.
4. **227 is not yet booked.** The fresh wall is only 98.1 us from 227, but the
   campaign endpoint remains the conservative 4515.396 us/token until a
   same-session reverse bracket promotes a mechanism.

## Next process

The next clean phase should split the two open terms rather than reopen a
closed microkernel spelling:

1. capture matched host/API and GPU timestamps around one settled current
   token to assign the **82.557 us host/outside-union** term;
2. rank the **372.510 us union excess** using current cold production bodies,
   with gate/up, down, flash, vocab, and handoff boundaries kept disjoint;
3. implement only the highest causal candidate, require bit-exact output and a
   fresh A/B/A wall bracket, then rebuild this ledger.

Evidence: `docs/task_workflow/evidence/nv-full-ledger-audit-20260824/`.
