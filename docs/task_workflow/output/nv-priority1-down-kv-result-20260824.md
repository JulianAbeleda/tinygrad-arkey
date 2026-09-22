# NV priority 1: down and Q/K/V topology result

Date: 2026-08-24  
Repo: `/home/ubuntu/tinygrad-arkey`  
Base HEAD: `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090 (`sm_120`), model: Qwen3-8B-Q4_K_M, single-token decode

## Outcome

Priority 1 produced two promotions and one useful rejection:

1. Q4_K FFN-down vector loads are bit-exact and promoted on NV `sm_120`.
2. Packing multiple Q6_K FFN-down rows into one CTA is bit-exact but slower in
   the production graph, so production remains one row per CTA.
3. The cooperative shared-Q8 Q4 projections now merge their four warp totals
   inside the producer CTA. This removes all 43 standalone Q/K/V completion
   kernels and is bit-exact.

The conservative current plain-wall endpoint is `4.635946 ms/token`, or
`215.706 tok/s`. The gap to 240 is now `469.279 us/token`. Priority 1 improved
the fresh installed device snapshot from 560 to 517 nodes and from
`4463.360` to `4393.776 us` of node sum. It did not reach 240.

## 1. Q4_K FFN-down vector loads: promoted

This is the same fixed-byte, higher-rate mechanism established for Q/K/O, now
applied to the 18 Q4_K down rows.

| measurement | scalar | vector | delta |
| --- | ---: | ---: | ---: |
| hot isolated CUDA event | 18.3664 us | 12.3888 us | -5.9776 us/launch |
| cold NCU duration | 26.112 us | 21.792 us | -4.320 us/launch |
| cold DRAM reads | 28,360,960 B | 28,360,448 B | -512 B |
| executed instructions | 16,179,200 | 14,852,096 | -8.20% |
| registers/thread | 40 | 38 | -2 |
| production device row | 382.400 us/token | 364.160 us/token | -18.240 us/token |

The native outputs are bit-identical (`max_abs_diff=0`). The cold replay reads
the same matrix bytes, but the vector kernel finishes 16.5% sooner and raises
reported DRAM utilization from 61.66% to 73.85%. This is further evidence that
fixed DRAM bytes do not make load/codegen changes irrelevant.

Plain reverse brackets also pass:

| depth | reps | control A ms | vector ms | control C ms | recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 15 | 4.693903 | 4.665045 | 4.701300 | 32.556 us/token |
| 128 | 9 | 6.635143 | 6.614302 | 6.632755 | 19.647 us/token |

The normal-load structural census sees 18 vector Q4 down programs and zero
scalar programs. `TINYGRAD_Q4K_SCALAR_LOAD=1` restores the scalar control.

## 2. Q6_K FFN-down row packing: rejected

The Q6 emitter already lets the renderer eliminate repeated weight reads, so
the bounded alternative tested here was CTA topology: 2, 4, or 8 independent
rows per block while preserving four warps and arithmetic order per row.

All three shapes were bit-exact. The host-synchronized microgate misleadingly
favored four rows per block by `1.586 us/launch`, but the production profile
reversed the result:

| production row | one row/CTA | four rows/CTA | delta |
| --- | ---: | ---: | ---: |
| Q6 down, 18 calls/token | 552.704 us | 563.360 us | +10.656 us/token |
| full node sum | 4443.488 us | 4454.240 us | +10.752 us/token |
| device union | 4440.125 us | 4451.000 us | +10.875 us/token |

The profiled wall midpoint is invalid for promotion because control C was a
large outlier; the causal device rows consistently regress. No loader or route
policy enables row packing, and production remains `rows_per_block=1`.

## 3. Shared-Q8 Q4 completion: promoted

The old cooperative Q4 route produced four fp32 warp totals per row and then
launched one reduction per projection. Across the 17 admitted shared-Q8 blocks
that meant 17 Q completions and 26 K/V completions per token. The new route
publishes those four totals to shared memory and merges them left-to-right in
the same CTA.

The numerical gate compares the old `partial + sum` result to the new direct
result using the same Q4 weights and packed Q8 activation:

| shape | float32 outputs checked | mismatched words | max abs delta |
| --- | ---: | ---: | ---: |
| Q, 4096x4096 | 4096 | 0 | 0 |
| K/V, 1024x4096 | 1024 | 0 | 0 |

Both output SHA-256 pairs match. This is full elementwise bit equality, not
only an equal generated-token stream.

The causal production profile is equally clear:

| bucket | old partial + completion | direct | delta |
| --- | ---: | ---: | ---: |
| Q cooperative Q4 | 160.480 us | 141.808 us | -18.672 us |
| K/V cooperative Q4 | 125.856 us | 97.024 us | -28.832 us |
| combined | 286.336 us | 238.832 us | -47.504 us |

Full-profile node sum falls `51.880 us/token` and union falls
`52.312 us/token`. The additional difference is scheduler/install cost from
removing the 43 nodes.

Plain wall qualification passes three times:

| depth | reps | control A ms | direct ms | control C ms | recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 9 | 4.640907 | 4.614866 | 4.618902 | 15.038 us/token |
| 512 | 15 | 4.662723 | 4.635946 | 4.655128 | 22.979 us/token |
| 128 | 9 | 6.583754 | 6.534462 | 6.608447 | 61.639 us/token |

The first depth-512 pass was narrow against the faster control, which is why
the reps=15 confirmation and the separate depth-128 bracket were required.
All three arms in each bracket have matching token-stream hashes.

The production census reports 43 direct Q4 programs, zero partial programs,
and zero completion programs. A fresh rollback process using
`TINYGRAD_SHARED_Q8_Q4_PARTIAL_OUTPUT=1` reports exactly 43 partial programs
and 43 completion programs. The promotion is target-scoped to NV `sm_120`.

## Updated installed and llama ledgers

The current no-override profile closes as:

```text
scheduled nodes  517
node sum          4393.776 us/token
device union      4390.750 us/token
overlap              3.026 us/token
```

Relative to the installed snapshot at the start of priority 1, node sum is
`69.584 us` lower and union is `70.125 us` lower. The independent llama
reconciliation reports a remaining node-sum delta of `515.506 us/token`.

| role | current tinygrad - llama, us/device-token | interpretation |
| --- | ---: | --- |
| attn/FFN/final 4096 norms | +115.42 | accounting caveat; paired with quant advantage |
| flash score | +82.24 | production/cache boundary; body already at parity/faster |
| K/V projections | +73.12 | completion gone; retained evidence says bodies are at parity, residual is install/boundary |
| flash combine | +66.69 | single-warp body/topology |
| vocab main + tail | +65.44 | serial reduction tail |
| down GEMV | +57.91 | Q4 improved; Q6 row packing rejected |
| gate/up GEMV | +53.65 | remaining streaming-rate residual |
| Q projection | +51.42 | completion gone; install/boundary residual |
| O projection | +48.86 | remaining streaming-rate residual |

The norm line must not be ranked alone: tinygrad's activation quant row is
`-113.98 us` versus llama, and the Q/K norm rows fuse RoPE work that llama
reports separately. The clean actionable ranking therefore starts with flash,
vocab, and remaining projection boundary work.

## Plan after priority 1

1. **Flash combine topology.** Build the already-scoped wider parallel
   reduction over 48 splits/head. Its body is `2.304 us/call` versus llama's
   `1.024 us/call`; the measured body-parity ceiling is about `46.1 us/token`.
   Require exact online-max/sum output, unchanged 36-node cardinality, a device
   profile, and a plain reverse bracket.
2. **Vocab tail topology.** Replace the latency-bound single-warp
   `r_32_4_1187` reduction with a wider or two-stage exact reduction. Do not
   retry the closed naive top-1 fusion. The current entire vocab residual is
   `65.44 us`, with roughly `39 us` historically concentrated in this row.
3. **Production-conditioned handoffs.** Revisit score-to-combine and Q/K/V
   installed residuals only with a causal boundary construction. Isolated body
   rewrites are not justified: flash score and K/V bodies are already at or
   below llama body time.
4. **Re-ledger after each accepted change.** The current wall gap is
   `469.279 us`; flash combine and vocab together cannot reach 240, so later
   work will still need multiple independent wins from the remaining
   down/gate-up/Q/O and boundary pools.

## Verification and evidence

The implementation includes target-scoped route policy, explicit rollback,
admission validation, structure tests, production profilers/brackets, and the
native bit-exact gate. The combined structured ledger is in
`docs/task_workflow/evidence/nv-priority1-down-kv-20260824/ledger.json`.
Detailed raw evidence is under:

- `docs/task_workflow/evidence/nv-q4k-down-vector-load-20260824/`
- `docs/task_workflow/evidence/nv-q6k-down-row-packing-20260824/`
- `docs/task_workflow/evidence/nv-shared-q8-q4-direct-20260824/`

Verdict: `PRIORITY1_PARTIAL_PARITY_TWO_PROMOTIONS_240_NOT_REACHED`.
