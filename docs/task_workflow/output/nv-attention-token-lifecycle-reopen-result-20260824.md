# NV attention token-lifecycle reopen result

Date: 2026-08-24

## Verdict

`ATTENTION_BOUNDARIES_ACCOUNTED_NO_PROMOTION`.

The installed attention lifecycle is already a direct chain after score
admission: flash score to combine and combine to O have zero measurable edge
wait across every steady within-group layer. Q/K readiness to score has a
visible command-timeline interval, but the legal split-phase construction
does not recover token wall time. No attention route is promoted.

## Fresh installed control

The fresh profiled graph remains the composed 452-node production route:

| metric | us/token |
| --- | ---: |
| node sum | 4114.592 |
| device union | 4110.750 |
| overlap | 3.842 |
| flash score, 36 calls | 238.528 |
| flash combine, 36 calls | 99.872 |
| O projection, 36 calls | 305.056 |

The profile's wall measurement is discarded because profiling perturbs token
delivery. The unprofiled reverse bracket supplies the endpoint authority.

## Exact lifecycle accounting

The same-clock profile contains 289 steady complete tokens. Thirty-five
attention chains per token live wholly inside graph groups; the remaining
layer crosses the 64-to-128 group boundary. A fresh pre-split full-token DAG
restores that layer and proves two direct RAW readiness dependencies into its
score node: Q norm/RoPE and K norm/RoPE/cache.

All 36 layers are therefore accounted without assuming independence.

| edge | median wait/call | median summed wait/token | verdict |
| --- | ---: | ---: | --- |
| Q/K ready -> score (35 timestamp-visible layers) | 2.750 us | 98.750 us | attribute/test |
| score -> combine | 0.000 us | 0.000 us | closed |
| combine -> O | 0.000 us | 0.000 us | closed |

The readiness sum is an observed command-timeline interval, not a recovery
claim. The production wall test below determines how much of it is real token
latency that the tested construction can remove.

## Candidate: KV-ready to score split-phase launch

The narrow candidate arms the immediate
`reduce_output_rmsnorm_rope_kv_cache_8_128` producer and flash-score consumer.
It triggers launch readiness at producer start and retains an in-kernel
consumer dependency wait, preserving the data dependency. Score arithmetic,
combine geometry, O code, buffers, and graph cardinality do not change.

Per-kernel PMA profiling cannot qualify this construction because profiling
commands break the active-QMD adjacency required by the native arm. That is an
instrumentation wall, not a negative mechanism result, so qualification moved
to an unprofiled fresh-process control/candidate/control bracket.

| arm | ms/token |
| --- | ---: |
| control A | 4.250508 |
| candidate | 4.251160 |
| control C | 4.254850 |

All token hashes match. The candidate appears 1.519 us/token faster than the
control midpoint but is 0.652 us/token slower than the faster control. It
therefore fails the conservative wall gate: `NO_GO_WALL`.

The result also prevents a bad accounting inference: the roughly 99 us/token
timestamp interval is not an approximately 99 us recoverable token pool.

## Current position versus llama

Using the control midpoint as this session's unprofiled endpoint:

```text
4252.679 us/token = 235.146 tok/s
```

| target | remaining latency |
| --- | ---: |
| 240 tok/s | 86.012 us/token |
| retained llama, 4048.325 us/token | 204.354 us/token |

The fresh device-union gap to the retained llama authority is 222.510
us/token. The remaining attention rows are real accounting surfaces, but the
known legal constructions are now closed: wider combine is wall-negative,
single-stage score/combine is negative, score-to-combine and combine-to-O
have zero edge delay, Q/K norm+RoPE/cache is already fused, O's isolated body
is near parity, and this exact readiness split-phase arm is wall-neutral.

The next campaign should not retry attention topology without new
information. It should refresh cold-rate and byte accounting for the current
Q and O projection cubins, then test only a construction that changes their
production rate or compulsory traffic. If those rows are also rate-closed,
the ledger should move outside attention rather than mining timestamp gaps.

## Evidence

Controlling evidence is under
`docs/task_workflow/evidence/nv-attention-token-lifecycle-reopen-20260824/`:

- `control-profile.json`
- `attention-edge-ledger.json`
- `kvready-score-pdl-wall-r9.json`

The raw full-token DAG and per-replay profile JSONL files remain local. The
committed edge ledger retains the DAG/profile hashes, semantic census, and the
two controlling cross-group RAW edges.
