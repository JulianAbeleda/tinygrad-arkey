# NV Q6_K 128-CTA full-K Gate 1 decision

Date: 2026-08-31  
Shape: `M=512, N=4096, K=12288`  
Evidence: `docs/task_workflow/evidence/nv-q6-oracle-fullk-tiles-gate1-20260831/result.json`

## Deterministic verdict

```text
verdict = FAIL_NUMERICAL_FINITE_SWEEP_EXHAUSTED
trusted_reference = PASS
tile_aligned_direct = FAIL
tile_aligned_factored = FAIL
controlled_sass_proofs = 24/24
numerically_passing_arms = 0/26
selected_parenthesization = none
```

No arithmetic association is admitted. The residual is already present in a 128-CTA route with one CTA per output tile, all 48 K256 epochs in that CTA, direct destination output, no partial workspace, and no fixup. Removing Stream-K subtotal formation and cross-CTA reduction therefore does not repair the `~0.187` error.

The complete order-preserving FP32 scale/reduction/contraction space defined for this gate was then swept. All 24 controlled arms emitted their requested FMUL/FADD/FFMA census exactly, but none materially changed the error envelope. The fault is not selectable FP32 parenthesization at the post-dot scale/reduction hop.

The next gate is a real-scale K256 prefix/hop localization. It must locate the first mismatch among packed input addressing, Q6/Q8 shared publication, integer IMMA/correction terms, scale extraction, and epoch-to-accumulator mapping before further performance work.

## Command

Executed from `/home/ubuntu/tinygrad-arkey` under the exclusive GPU lock:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_fullk_tiles.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-fullk-tiles-gate1-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-fullk-tiles-gate1-20260831/artifacts
```

The final sweep compiled 26 kernels, ran each arm once in every round using round-index rotation and odd-round reversal, captured normalized binary artifacts, and returned exit code 1 because no numerical arm qualified.

## Isolation contract

| Property | Fixed value |
|---|---|
| Grid / block | `(128,1,1)` / `(256,1,1)` |
| Tile grid | `(4,32)` |
| Work per CTA | one 128x128 output tile, 48 ascending K256 epochs |
| Partials / fixup | none / none |
| Output | direct row-major `512x4096` |
| Q6 publisher | oracle publisher |
| Q8 schedule | second-panel prefetch enabled |
| Initial publication | separate, unchanged |
| Fixture | `blk.0.ffn_down.weight` from Qwen3-8B Q6_K model |
| Reference | independently qualified compiler-wide direct output |
| Operand order | `wd`, `yscale`, `dot`; unchanged |
| `p` order | `p0,p1,p2,p3`; unchanged |

The kernel's new `tile_grid` routing option is default-off and mutually exclusive with Stream-K. Existing callers retain their prior addressing.

## Reference qualification

| Metric | Value |
|---|---:|
| Reference passed | true |
| Maximum absolute error against its independent oracle | `0.001220703125` |
| Mean absolute error | `0.00003766872760024853` |
| Finite | true |

## Exhaustive association space

The two implicit baselines were retained:

```text
direct:   acc + ((wd * yscale) * dot)
factored: acc + (left_fold_zero(yscale_p * dot_p) * wd)
```

The controlled direct space contains both order-preserving multiplication trees and both relevant contraction regimes:

```text
((wd * yscale) * dot): none, final-FMA
(wd * (yscale * dot)): none, final-FMA
```

The controlled factored space contains all five Catalan, order-preserving binary trees over `u0,u1,u2,u3`, where `up=yscale_p*dot_p`:

```text
left:        (((u0 + u1) + u2) + u3)
inner_left:  ((u0 + (u1 + u2)) + u3)
inner_right: (u0 + ((u1 + u2) + u3))
right:       (u0 + (u1 + (u2 + u3)))
balanced:    ((u0 + u1) + (u2 + u3))
```

Each tree was run with `none`, `tmp_only`, `final_only`, and `both` contraction regimes. Controlled CUDA uses `__fmul_rn`, `__fadd_rn`, and `__fmaf_rn`; exact SASS-family counts are an admission condition. Parentheses alone were not treated as proof of contraction.

## Every arm

Tolerance is `rtol=2e-5, atol=2e-3` over 2,097,152 outputs. Every arm is finite and every arm fails allclose.

| Arm | Max abs | Mean abs | Failures | R31 median us | Stack B | LDL/STL | FFMA/FMUL/FADD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `direct_da` | 0.187255859375 | 0.0136876823381 | 1,758,882 | 284.320 | 64 | 15/30 | 512/512/0 |
| `factored_da` | 0.187255859375 | 0.0136876869947 | 1,758,819 | 349.056 | 264 | 74/132 | 640/0/0 |
| `direct_wd_yscale_then_dot_none` | 0.187255859375 | 0.0136876944453 | 1,758,878 | 264.416 | 0 | 0/0 | 0/1024/512 |
| `direct_wd_yscale_then_dot_final` | 0.187255859375 | 0.0136876823381 | 1,758,882 | 284.192 | 64 | 15/30 | 512/512/0 |
| `direct_wd_then_yscale_dot_none` | 0.187255859375 | 0.0136876944453 | 1,758,878 | 277.920 | 16 | 5/6 | 0/1024/512 |
| `direct_wd_then_yscale_dot_final` | 0.187255859375 | 0.0136876823381 | 1,758,882 | 267.808 | 16 | 4/8 | 512/512/0 |
| `factored_left_none` | 0.187255859375 | 0.0136876925826 | 1,758,810 | 265.696 | 16 | 4/8 | 0/640/512 |
| `factored_left_tmp_only` | 0.187255859375 | 0.0136876925826 | 1,758,810 | 265.568 | 80 | 25/38 | 384/256/128 |
| `factored_left_final_only` | 0.187255859375 | 0.0136876869947 | 1,758,819 | 254.176 | 0 | 0/0 | 128/512/384 |
| `factored_left_both` | 0.187255859375 | 0.0136876869947 | 1,758,819 | 323.776 | 264 | 75/132 | 512/128/0 |
| `factored_inner_left_none` | 0.187255859375 | 0.0136876888573 | 1,758,811 | 267.904 | 16 | 4/8 | 0/640/512 |
| `factored_inner_left_tmp_only` | 0.187255859375 | 0.0136876888573 | 1,758,811 | 285.216 | 136 | 39/66 | 384/256/128 |
| `factored_inner_left_final_only` | 0.187316894531 | 0.0136876888573 | 1,758,816 | 253.632 | 0 | 0/0 | 128/512/384 |
| `factored_inner_left_both` | 0.187316894531 | 0.0136876888573 | 1,758,816 | 325.632 | 264 | 71/132 | 512/128/0 |
| `factored_inner_right_none` | 0.187255859375 | 0.0136876869947 | 1,758,808 | 265.632 | 56 | 14/26 | 0/640/512 |
| `factored_inner_right_tmp_only` | 0.187255859375 | 0.0136876869947 | 1,758,808 | 275.808 | 152 | 44/74 | 384/256/128 |
| `factored_inner_right_final_only` | 0.187316894531 | 0.0136876832694 | 1,758,798 | 254.816 | 0 | 0/0 | 128/512/384 |
| `factored_inner_right_both` | 0.187316894531 | 0.0136876832694 | 1,758,798 | 320.064 | 256 | 72/128 | 512/128/0 |
| `factored_right_none` | 0.187255859375 | 0.0136876869947 | 1,758,799 | 265.088 | 40 | 10/20 | 0/640/512 |
| `factored_right_tmp_only` | 0.187255859375 | 0.0136876869947 | 1,758,799 | 273.376 | 80 | 25/40 | 384/256/128 |
| `factored_right_final_only` | 0.187316894531 | 0.0136876795441 | 1,758,795 | 254.432 | 0 | 0/0 | 128/512/384 |
| `factored_right_both` | 0.187316894531 | 0.0136876795441 | 1,758,795 | 326.592 | 264 | 71/132 | 512/128/0 |
| `factored_balanced_none` | 0.187255859375 | 0.0136876888573 | 1,758,808 | 275.168 | 24 | 5/10 | 0/640/512 |
| `factored_balanced_tmp_only` | 0.187255859375 | 0.0136876888573 | 1,758,808 | 264.672 | 24 | 6/12 | 256/384/256 |
| `factored_balanced_final_only` | 0.187316894531 | 0.0136876795441 | 1,758,812 | 253.760 | 0 | 0/0 | 128/512/384 |
| `factored_balanced_both` | 0.187316894531 | 0.0136876795441 | 1,758,812 | 293.888 | 128 | 36/64 | 384/256/128 |

Every controlled arm has 256 IMMA and 32 LDSM. Controlled SASS proof passed 24/24. The observed error envelope is:

```text
max_abs:     0.187255859375 .. 0.18731689453125
mean_abs:    0.0136876795441 .. 0.0136876944453
failures:    1,758,795 .. 1,758,882
```

That envelope is tiny relative to the residual and contains no trend toward the tolerance threshold.

## Baseline performance observation

The 26-arm rotating R31 schedule gives:

| Metric | Direct baseline | Factored baseline | Paired direct minus factored |
|---|---:|---:|---:|
| Median | `284.320 us` | `349.056 us` | `-63.648 us` |
| Factored wins | - | 0/31 | - |

These timings are diagnostic only because neither arm is numerically qualified. Several explicitly rounded variants remove all spills and run near 254-265 us, which is useful compiler evidence but not a candidate until the lower hop is repaired.

## Localization proof

The route can be separated into:

```text
host Q6/Q8 packing
  -> global address/tile mapping
  -> shared Q6/Q8 publication and decode
  -> IMMA integer dot/correction construction
  -> FP32 scale multiplication, p-tree, and contraction
  -> ordered epoch accumulation
  -> output store
  -> Stream-K subtotal/fixup (absent in Gate 1)
```

Gate 1 proves:

1. The residual is not created by partial-slot addressing, cross-CTA subtotal association, or the fixup; those hops are absent.
2. The residual is not repaired by either scale multiplication tree, any order-preserving four-term p tree, or any relevant FMA contraction regime; the emitted SASS was verified.
3. Therefore the first semantic mismatch is upstream of the swept post-dot FP32 tree, or in the fixed epoch/output mapping that supplies it. The leading candidates are real packed-address routing, Q6/Q8 publisher decode, integer dot/correction terms, scale-index selection, or K64/K128-to-K256 epoch mapping.

This does not prove which lower hop is wrong. It does bound the fault below the Stream-K and selectable FP32-association layers.

## Next gate: real-scale K256 prefix and hop ledger

### Phase A: prefix localization

Create compact real-model slices and compare the same direct recurrence at depths:

```text
1, 2, 4, 8, 16, 24, 32, 40, 48 K256 epochs
```

Each depth must build matching packed weight row strides and Q8 records rather than pointing a shorter-K kernel at full-K row-major storage. Use one CTA per tile, no fixup, and an independently generated wide-direct reference for the identical K prefix.

Decision:

- If depth 1 fails, enter Phase B immediately.
- If depth 1 passes, binary-search between the last passing and first failing depth. This isolates ordered epoch accumulation or an epoch address transition.
- If every prefix below 48 passes and 48 fails, inspect the last-epoch boundary and row stride.

### Phase B: first-failing-epoch hop dump

For one fixed tile and a small deterministic set of output lanes, save and compare:

```text
raw 105-u16 Q6 block and scale bytes
raw two-panel Q8 record and scale bits
published shared Q6 words
published shared Q8 words
decoded q6 values/scales/mins
IMMA accumulator words before correction
integer corrected dot[p,r]
FP32 wd and yscale bit patterns
per-p scaled contribution bits
accumulator bits before and after the epoch
final row/column destination index
```

Acceptance is bit identity at each integer/bit-pattern hop and the declared FP32 rounding at the first floating hop. Stop at the first mismatch. Do not optimize scheduling or reduction while this ledger is unresolved.
