# NV Q6_K direct-dA full-route Gate 0 decision

Date: 2026-08-31  
Shape: `M=512, N=4096, K=12288`  
Evidence: `docs/task_workflow/evidence/nv-q6-oracle-full-streamk-factor-da-gate0-20260831/result.json`

## Verdict

**Gate 0 does not admit the direct-`dA` candidate.**

Direct `dA` is a large and repeatable performance/resource improvement, but both arithmetic arms fail the trusted compiler-wide direct reference. Direct placement slightly worsens maximum absolute error, slightly improves mean absolute error, and removes only six tolerance failures out of roughly 1.76 million. This disproves `dA` placement as the numerical repair.

The next diagnostic is the 128-CTA tile-aligned direct route. Do not begin the one-physical-body compiler experiment until the numerical contract is qualified.

## Command

Executed from `/home/ubuntu/tinygrad-arkey`:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-full-streamk-factor-da-gate0-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-full-streamk-factor-da-gate0-20260831/artifacts
```

The process completed both compilations, correctness checks, artifact analyses, and 31 alternating timing rounds. It returned exit code 1 because the qualification predicate was false.

## Isolation contract

The two arms differ only in `factor_dA`:

| Invariant | Both arms |
|---|---|
| Main launch | `(170, 1, 1)` CTAs, `(256, 1, 1)` threads |
| Ownership | floor partition of 6,144 K256 tile-epochs across 170 owners |
| Owner lengths | 146 of 36, 24 of 37 |
| Segment bodies | static segment 0 plus static segment 1 |
| Active slots | 294, plane-major `segment*170+owner` |
| Publisher | oracle Q6 publisher, Q8 second-panel prefetch enabled |
| Barriers | 11 whole-kernel / five per selected body |
| Fixup | one shared deterministic ascending-map-order CUDA kernel |
| Reference | same compiler-wide direct output |
| Timing | same process, alternating AB/BA order, 31 rounds |

## Trusted reference

The compiler-wide direct reference qualified independently:

| Metric | Value |
|---|---:|
| Finite | true |
| Reference allclose | true |
| Maximum absolute error | `0.001220703125` |
| Mean absolute error | `0.00003766872760024853` |

## Correctness

Tolerance is `rtol=2e-5, atol=2e-3` over 2,097,152 outputs.

| Metric | Direct dA | Factored dA | Direct minus factored |
|---|---:|---:|---:|
| Finite | true | true | - |
| GPU fixup equals CPU slot recurrence | true | true | - |
| GPU/CPU recurrence max abs | `0.0` | `0.0` | `0.0` |
| Trusted-reference max abs | `0.18719482421875` | `0.1871337890625` | `+0.00006103515625` |
| Trusted-reference mean abs | `0.013687681406736374` | `0.01368770468980074` | `-2.3283064366e-8` |
| Failing elements | 1,758,835 | 1,758,841 | `-6` |
| Trusted-reference allclose | false | false | - |

Interpretation:

- The deterministic fixup implementation is not malfunctioning; both GPU outputs exactly match their declared CPU slot recurrences.
- Mathematically equivalent direct/factored placement changes FP32 results, but the change is negligible relative to the existing route error.
- Direct placement cannot be promoted on correctness even though it is the preferred diagnostic arithmetic because it is faster and much less spill-heavy.

## R31 same-process timing

| Metric | Direct dA | Factored dA | Paired direct minus factored |
|---|---:|---:|---:|
| Main median | `225.344 us` | `285.056 us` | `-59.904 us` |
| Fixup median | `25.472 us` | `26.720 us` | not used for causal main verdict |
| Main+fixup median | `251.040 us` | `312.192 us` | `-61.280 us` |
| Direct wins | 31/31 main | - | 31/31 pair |

Direct improves the paired main median by about 21.0% and the paired total by about 19.6%. It remains `24.128 us` above llama's `201.216 us` main and `41.184 us` above llama's `209.856 us` total. It also misses the final 5% thresholds (`211.2768 us` main and `220.3488 us` total).

## Normalized full-kernel SASS/resources

| Metric | Direct dA | Factored dA | Pinned llama main |
|---|---:|---:|---:|
| Static instructions | 8,176 | 8,328 | 8,648 |
| Registers | 255 | 255 | 255 |
| Stack | 16 B | 288 B | 72 B |
| LDL / STL | 7 / 14 | 251 / 377 | 31 / 29 |
| IMMA / LDSM | 512 / 64 | 512 / 64 | 512 / 64 |
| LDG / STS | 210 / 142 | 210 / 142 | 210 / 143 |
| BAR | 11 | 11 | 9 |
| FFMA / FMUL | 1,024 / 1,024 | 1,280 / 0 | 1,280 / not selected |

Direct combined cubin SHA-256:

```text
55a830e7698e702475634304db879fe24b3120d2d621ba9a8e90d483a79a7c8a
```

This confirms the depth result on the full route: direct `dA` removes almost all whole-kernel spill traffic. It also challenges the assumption that physical-body deduplication must precede other scheduling work; the two-body direct kernel already has fewer stack bytes and local operations than llama.

## Required Gate 0A association evidence

### Step 1: tile-aligned discriminator

Build a direct-`dA` diagnostic with exactly 128 CTAs. Each CTA owns one output tile and executes all 48 K256 epochs in ascending order. It writes the destination directly and has no scratch buffer or fixup.

Hold constant:

- Q6/Q8 records and publishers.
- Direct `dA` per-contribution recurrence.
- Output tile coordinate mapping and transpose.
- K256 epoch order.
- Trusted compiler-wide direct reference and tolerance.

Decision:

- If 128-CTA direct fails, the error exists inside the body contract. Compare one real-scale K256 update, then grow depth `1, 2, 4, 8, 16, 32, 48` to find the first divergence. Do not sweep partial associations.
- If 128-CTA direct passes, the body is qualified and the remaining error is introduced by segment subtotal formation/reduction.

### Step 2: association sweep only after a tile-aligned pass

Save the 294 active direct-`dA` partial tiles and the exact slot-to-tile/owner map. For every three-contributor tile, evaluate in FP32:

```text
(p0 + p1) + destination
p0 + (p1 + destination)
(p0 + destination) + p1
```

For every two-contributor tile, retain the single nontrivial association. Report max/mean/failing count for each complete output. The current forward order and llama backward-partials-then-destination order both reduce to the first association for the three-contributor case, so merely reversing `p0` and `p1` is not a distinct FP32 tree.

If no association passes, the mismatch is segment subtotal rounding rather than fixup ordering. The remaining choices are a higher-fidelity partial representation, a compensated two-component subtotal, or tile-aligned ownership.

## Admission state

```text
trusted_reference_qualified = true
direct_cpu_fixup_exact = true
factored_cpu_fixup_exact = true
direct_reference_qualified = false
factored_reference_qualified = false
direct_performance_gate = true
gate0_admitted = false
next_gate = gate0a_tile_aligned_direct
```
