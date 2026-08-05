# NV Q4_K exact native substrate Rank-1 record — 2026-08-05

## Verdict

**NO-GO for both exact constructions measured here.**  Neither is promoted,
route-bound, or default-enabled.  The four-warp construction is 2.42% slower
than installed; the one-warp algebraic factorization is wall-neutral.  This
does not prove that every possible exact Q4_K kernel is closed.  It does prove
that copying llama's observed four-warp ownership without its Q8_1 activation
representation does not recover llama's MMVQ advantage.

The causal source correction is load-bearing: the pinned d512 llama trace runs
`mmvq.cu::mul_mat_vec_q<Q4_K,1,...>`, which uses Q8_1 + DP4A and four warps per
output.  It does **not** run the `mmq.cuh` integer-MMA path.  CUDA integer-MMA
descriptor work therefore remains generic/future substrate evidence and is
not an explanation of the measured decode gap.

## Constructions

Both candidates consume the production packed Q4_K `uint32` words and the
production fp16 activation directly.  There is no Q8 provider, DP4A, integer
MMA, alternate weight layout, or hidden preparation kernel.

1. `q4k_exact_four_warp`: one 128-thread CTA/output; four disjoint K stripes;
   four-lane subgroup/group; two packed words/lane/block; exact group-factorized
   fp32 arithmetic; 16-byte shared rendezvous; one final output; one program.
2. `q4k_exact_group_factorized`: installed one-warp/output geometry, but changes
   the arithmetic from 32 per-value scale/min subtractions to
   `d*sc*sum(q*x) - dmin*mn*sum(x)` once per quant group.  This is not the
   rejected vector-carrier spelling.

The independent oracle decodes Q4_K from its byte layout and accumulates in
fp64; it does not import the emitter's lane mapping or arithmetic expression.

## Static PTX/resource gate, production 4096 x 4096

| construction | threads | PTX bytes | static global loads | shuffles | physical registers | shared | spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| installed G3 | 32 | 14,110 | 40 | 5 | 61 | 0 | 0 |
| exact one-warp factorized | 32 | 12,429 | 44 | 8 | 63 | 0 | 0 |
| exact four-warp runtime-loop | 128 | 6,315 | 14 | 7 | 36 | 16 B | 0 |

The compact four-warp static body was not treated as a speed claim: its loop
executes four times and it launches four times as many threads per output.
This is why the native included-cost gate was mandatory.

## Native RTX 5090 gate

Command:

```sh
flock /tmp/gpu-bench.lock env DEV=NV PYTHONPATH=. python3 \
  extra/llm_research/decode/q4k_exact_four_warp_microgate.py \
  --replays 200 --reps 7
```

Four-warp correctness passes: selected-row independent-oracle max absolute
error `1.19209e-7`; full primitive relative L2 versus installed `2.39831e-7`.
The timed A/B/A result is:

| arm | median us/kernel |
|---|---:|
| installed control midpoint | 118.578843 |
| exact four-warp | 121.447340 |

Ratio `1.0241906x`; delta `+2.868498 us`; material-win gate **FAIL**.
The canonical payload is
`docs/task_workflow/output/nv-q4k-exact-four-warp-microgate-20260805.json`.

The exact one-warp factorized follow-up used the same payload, oracle rows,
200 replays, seven samples/arm, reverse A/B/A, and GPU lock.  Correctness was
`2.32753e-7` relative L2 versus installed.  Its installed midpoint was
`121.799850 us`, candidate `121.592310 us`: ratio `0.9982961x`, delta
`-0.207540 us`.  That is noise-scale/wall-neutral and fails the same 5%
material-win gate.

## What this resolves

- Four-warps/output is not independently sufficient under exact fp16 input.
- Group-factorizing the exact dequant arithmetic reduces subtract/conversion
  pressure, but added loads/adds/shuffles erase the static benefit at wall.
- The observed llama MMVQ mechanism's large isolated win is a coupled package:
  Q8_1 activation representation + DP4A + four-warp ownership.  The checked-in
  shared-Q8 progression already measures that package and stops at g18 on its
  cumulative full-token semantic threshold; this record does not reopen it.
- Integer MMA is not observed d512 decode causality.  More importantly, an
  s8/s8 MMA cannot consume the exact fp16 activation without quantization or a
  multi-bitplane/exponent decomposition.  The former reopens the prohibited
  approximation; the latter needs many MMA passes and is not a credible next
  primitive absent a cheaper exact encoding proof.

## Rank-1 disposition

Close these two exact candidates with zero parity credit.  Do not run a model
family or full-logit promotion gate: the primitive material-win prerequisite
failed.  A future exact-Q4 reopen must name a third physical representation or
instruction mapping, preserve packed-Q4 + fp16 semantics, and beat installed
by at least 5% on this identical-shape one-program gate before consuming
full-token GPU time.
