# NV exhaustive full-token role census result (2026-08-22)

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

## Verdict

`CLOSED`. The device gap is now fully named. Every kernel in the retained
tinygrad control capture (596 nodes) and the llama PDL-off wait-free oracle
(762 nodes) is summed exactly once into a common role. The node_sum delta
closes at `799.67 us`, and it decomposes into the nine previously named rows
(`646.84 us`) plus a `152.83 us` remainder that is **kernel residence in the
norm/rope/quant group, not an unidentified launch bubble.**

The two accounting errors from the re-audit are confirmed: the `~37 us` body
sum omitted invocation cardinality, and the S1 window ended at an early-started
waiting consumer. After correcting those, there is no `748 us` scheduler
shadow left to discover.

Accounting is frozen at census SHA
`0326f0d21e10059a92196a439431f5bd58fb04353a6b20d972e94b3cece494cf`.

## Closure

```text
node_sum delta   = tinygrad 4677.92 - llama 3878.25  = 799.67 us
union delta      = 793.25 us            (PDL-off wait-free boundary)
tinygrad overlap =   6.42 us            (tinygrad's own small overlap)

799.67 = 793.25 + 6.42
```

## Full role table

Counts are kernels per token. Delta is tinygrad minus llama, node_sum domain.

| role | tinygrad us | llama us | delta us | tiny us/kernel | llama us/kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| attn/ffn/final 4096 norm | 328.80 | 203.36 | **+125.44** | 3.50 (94) | 2.79 (73) |
| K/V projections + completion | 327.23 | 215.01 | **+112.22** | 3.34 (98) | 2.99 (72) |
| gate/up GEMV | 1369.70 | 1268.37 | **+101.33** | 38.05 | 35.23 |
| Q projection + completion | 333.50 | 249.09 | **+84.41** | 6.29 (53) | 6.92 (36) |
| O projection | 335.04 | 259.81 | **+75.23** | 9.31 | 7.22 |
| down GEMV | 930.37 | 855.82 | **+74.55** | 25.84 | 23.77 |
| vocab main + tail | 371.52 | 303.91 | **+67.61** | 74.30 (5) | 151.95 (2) |
| flash combine | 104.00 | 37.06 | **+66.94** | 2.89 | 1.03 |
| flash score | 227.49 | 162.95 | **+64.54** | 6.32 | 4.53 |
| Q head norm | 96.74 | 41.28 | **+55.45** | 2.69 | 1.15 |
| K head norm | 90.21 | 40.83 | **+49.38** | 2.51 | 1.13 |
| rope + K/V store | 121.47 | 92.87 | **+28.61** | 1.66 (73) | 0.86 (108) |
| misc / embedding | 11.17 | 2.78 | **+8.38** | 2.79 (4) | 0.93 (3) |
| activation quant | 30.69 | 145.12 | **-114.43** | 1.81 (17) | 0.67 (216) |

The nine previously named rows (gate/up, complete Q, O, down, vocab,
flash combine, flash score, complete K, complete V) reproduce exactly from
this census: `646.84 us`.

## The remainder, named

The previously unattributed `152.83 us` (node_sum domain; about `146.41 us`
union domain) is:

```text
attn/ffn/final 4096 norm   +125.44
Q head norm                 +55.45
K head norm                 +49.38
rope + K/V store            +28.61
misc / embedding             +8.38
activation quant           -114.43   (tinygrad advantage)
sum                        +152.83
```

The cleanest, highest-confidence recovery target is the Q/K head-norm pair:

```text
Q head norm   +55.455 us   (36 calls each side)
K head norm   +49.376 us   (36 calls each side)
combined      +104.831 us
```

These two rows have matching semantic cardinality (36 vs 36) and no
provider-attribution ambiguity. Per kernel, tinygrad is slower than llama: q
norm `2.69` vs `1.15 us` (2.34x), k norm `2.51` vs `1.13 us` (2.22x).

The 4096-norm row is not an independent claim:

```text
4096 norm +125.44  +  activation quant -114.43  =  +11.01 us net
```

The 17 `rmsnorm_q8_1_llama_provider_4096` nodes fuse the attention input norm
with q8_1 activation quant, so the `+125.44 us` deficit is partially paid for
by the `-114.43 us` quant advantage. Recovering the 4096 norm without
preserving the provider path would consume that advantage. Tinygrad's 94 count
is physical kernel invocations while llama's 73 count is semantic norm
operations (56 standalone logical norms plus 17 fused providers on the tinygrad
side), so a per-call average over physical counts is not a like-for-like
comparison.

The remainder is kernel residence, not an unassigned nonresident launch gap.
Whether that residence is body arithmetic, occupancy, cache state, or
predecessor-conditioned launch elongation is not yet established and needs
counters plus conditioned timing.

## Attribution caveats

- The `E_*` / `r_*` roles are mapped by exact canonical kernel name plus shape.
  The `E_16_32_4_2`, `E_8_8_16_2`, and `E_16_4_2_8_16_2_4_4` group is assigned
  to rope/store by position and shape; semantic metadata is absent on those
  nodes, so that bucket needs one confirmation pass before code changes.
- The 17 `rmsnorm_q8_1_llama_provider_4096` nodes couple the 4096-norm row to
  the activation-quant row. The two must be treated together (`+11.01 us` net)
  unless the provider advantage is independently preserved.
- Durations are profiled node residence, never an unprofiled wall claim.

## Recommended order

1. Q/K head norms together: `104.831 us` attribution ceiling, clean 36-call
   semantic match, no provider ambiguity.
2. Standalone 4096 norms, while freezing the 17 provider path.
3. gate/up, Q, O, down, then flash-combine predecessor-conditioned reopenings.
4. PDL wait-exit timestamps as semantic validation, not optimization work.

## Evidence

- Census tool: `extra/llm_research/decode/nv_full_token_role_census.py`
- Machine-readable: `docs/task_workflow/evidence/nv-full-token-role-census-20260822/role-census.json`
- Inputs (retained, SHA-pinned):
  - `probe2-tinygrad-capture.json`
  - `probe2-llama-pdl0-dag.json`
