# NV dense FFN composition reopen result

Date: 2026-08-24

## Verdict

`PROMOTE_FFN_COLD_RATE_CONSTRUCTION`.

The producer-composition hypothesis is closed: the installed gate/up output
already feeds down through one direct dependency with no measurable edge
delay. The cold-rate branch produced one promotion. Four-block unrolling of
the current packed-lane Q6_K down loop is bit-exact, reduces long-scoreboard
stalls, passes the full production wall, and is now the NV `sm_120` default.

Rollback:

```text
TINYGRAD_Q6K_FFN_DOWN_UNROLL_DISABLE=1
```

## H1: producer boundary closes

The fresh profile reconstructed 289 steady complete tokens and 10,404 direct
gate/up-to-down pairs. Every token had 36 pairs, every down program named its
immediately preceding gate/up program as a dependency, and every measured
edge interval was exactly zero:

| edge statistic | us |
| --- | ---: |
| minimum | 0.000 |
| median | 0.000 |
| p95 | 0.000 |
| maximum | 0.000 |
| median sum per token | 0.000 |

There is no admission or idle interval between the two bodies to remove.
Together with the earlier structural result—no intervening activation, cast,
copy, quantization, or packing kernel—this closes another producer fusion as
an FFN lever. A monolithic gate/up-plus-down construction would have to change
the algorithm or duplicate work; it is not justified by the current ledger.

## H2: Q6 cold-rate mechanism

Gate/up is already near the measured streaming ceiling: its fresh cold probe
reads the full 56.64 MB stream, spills nothing, and reaches 84.94% of reported
peak DRAM throughput. Q4 down's already-promoted vector path reaches 73.85%.

Current Q6 down was the remaining supported target. Its cache-cold profile
showed:

| metric | packed control | four-block unroll |
| --- | ---: | ---: |
| DRAM reads | 41.334 MB | 41.338 MB |
| duration | 31.392 us | 30.272 us |
| reported DRAM peak | 74.74% | 77.50% |
| long-scoreboard stalls | 75.05% | 41.46% |
| executed instructions | 18.940 M | 17.908 M |
| registers/thread | 38 | 40 |
| spills | 0 | 0 |

The byte difference is negligible counter granularity. The candidate groups
four consecutive Q6 blocks inside each iteration so their independent loads
are visible together, while retaining left-to-right accumulation. This attacks
the measured long-scoreboard limit without changing compulsory traffic.

## Kernel lifecycle gate

All tested groupings (`2`, `3`, `4`, `6`, and `12`) are elementwise bit-exact
against the packed control. Four blocks was the best bounded CUDA-event
variant; full unrolling increased register pressure and regressed.

The selected four-block candidate:

- preserves Q6_K weights, fp16 activation, fp32 residual, and fp32 output;
- preserves left-to-right block accumulation;
- adds no copy, cast, synchronization, graph break, or node;
- applies to the same 18 Q6 FFN-down calls;
- is selected only by the generated NV `sm_120` route policy.

## Token lifecycle gate

The production depth-512, reps-9 control/candidate/control bracket is exact:

| arm | us/token |
| --- | ---: |
| control A | 4287.914 |
| candidate | 4253.360 |
| control C | 4296.131 |

The candidate recovers 38.662 us/token against the control midpoint and
34.554 us/token against the faster control. All arms have the same token-stream
hash. The conservative booked recovery is therefore **34.554 us/token**.

The installed device profile explains the wall movement:

| metric | control | promoted | recovery |
| --- | ---: | ---: | ---: |
| Q6 down row | 533.792 us | 491.232 us | 42.560 us |
| node sum | 4150.624 us | 4113.728 us | 36.896 us |
| device union | 4147.750 us | 4110.000 us | 37.750 us |

The promoted graph retains 452 nodes and the same topology. The profiled run's
wall field is discarded because profiling perturbed token delivery; device
times and unprofiled wall remain separate authorities.

## New installed endpoint

The fresh unprofiled installed endpoint is:

```text
4250.643 us/token = 235.259 tok/s
```

This is cross-session endpoint state, not the causal booking. From this fresh
authority:

| target | remaining latency |
| --- | ---: |
| 240 tok/s | 83.976 us/token |
| retained llama | 202.319 us/token |

FFN composition H1 is exhausted with complete information. The admitted H2
candidate is promoted and booked. The next campaign should move to the
attention lifecycle—projection/norm/cache/flash/O boundaries and only those
independent edges supported by a same-clock DAG.

## Evidence

Machine-readable evidence is under
`docs/task_workflow/evidence/nv-dense-ffn-composition-reopen-20260824/`.
The controlling files are:

- `ffn-edge-ledger.json`
- `ffn-cold-counters.json`
- `q6-down-unroll-microgate.json`
- `q6-down-u4-cold-counters.json`
- `q6-down-u4-wall-r9.json`
- `candidate-profile.json`
- `installed-wall.json`
- `final-ledger.json`
