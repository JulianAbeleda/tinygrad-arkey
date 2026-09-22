# NV llama shadow split result 20260821

## Verdict

Llama's per-kernel wait-exit spin is **data dependency wait, not launch
shadow**. In the earliest-block bracket 95.3% of the spin is the consumer
waiting for its producer's output to become visible; only 4.7% (63.5 us per
token) is true launch/scheduling shadow. This resolves the split the H1 and
reconciliation results left `unmeasured`, and confirms H1's note that the MMQ
anchors hold most of the dependency wait.

A preliminary working note had leaned toward launch shadow by using the
producer's `launch_dependents` trigger as the dependency boundary. That was
wrong: `griddepcontrol.wait` unblocks at producer completion, not at the
trigger. This supersedes the `unmeasured` label in the useful-body
reconciliation result.

## What the ring proves

The retained ring records every `cudaTriggerProgrammaticLaunchCompletion`
(kind=0) and `cudaGridDependencySynchronize` exit (kind=1). The linear
programmatic chain `0 -> 1 -> ... -> 761` fixes each consumer's producer as
the previous kernel. Across the steady replays:

- At least 99.5% of wait-exits land within 500 ns of the producer's CUPTI
  **end** (median offset -0.26 to -0.27 us).
- Producers that fire a trigger do so a median **2.3-2.5 us before** their
  end, yet the consumer's wait-exit still tracks that end, not the trigger.

`launch_dependents` therefore only enables early scheduling (co-residency); it
provides no memory visibility, so the consumer genuinely blocks until the
producer's writes are visible. The spin is the anchor sitting resident while
its quant/norm provider finishes, which is real serialization, not idle
overlap that can be harvested.

## Spin split

Merged per-token mean over the subsampled and full-sampling captures.

| bracket | spin us | dependency wait us | launch shadow us | launch share |
| --- | ---: | ---: | ---: | ---: |
| earliest block (`we_lo`) | 1362.5 | 1299.1 | 63.5 | 4.7% |
| latest block (`we_hi`) | 3770.1 | 1338.7 | 2431.4 | 64.5% |

`we_lo`/`we_hi` are the earliest/latest block wait-exits per kernel. The
`we_hi` row's "launch shadow" is the block scheduling tail of the large
4096-block MMQ grids (the last block starts long after the first), not host
launch latency. The `we_lo` row is the true per-block spin: nearly all of it
is dependency wait.

## Per-family, earliest-block bracket

Launch shadow is the component that overlap/PDL could plausibly recover. It
totals 63.5 us and is concentrated in the small kernels that launch after a
no-trigger MMQ producer completes.

| family | spin us | dependency us | launch us | launch share |
| --- | ---: | ---: | ---: | ---: |
| gemv (Q/O/gate/up/down) | 459.4 | 459.3 | 0.13 | 0.03% |
| quant_provider | 456.1 | 438.3 | 17.8 | 3.9% |
| rmsnorm | 47.2 | 2.0 | 45.2 | 95.8% |
| flash_combine | 164.7 | 164.7 | 0.01 | 0.01% |
| gemv_kv | 84.9 | 84.8 | 0.12 | 0.1% |
| rope | 70.1 | 70.1 | 0.07 | 0.1% |
| kv | 57.0 | 56.9 | 0.07 | 0.1% |
| flash_score | 18.5 | 18.4 | 0.05 | 0.2% |
| vocab | 3.2 | 3.2 | 0.00 | 0.0% |
| residual | 1.4 | 1.4 | 0.00 | 0.0% |

The 144 MMQ anchors are essentially 100% dependency wait (0.13 us launch):
each anchor pre-launches during its quant provider and blocks until that
provider completes. The `rmsnorm` family is the mirror case: its producer is a
no-trigger MMQ, so it launches only after the MMQ completes and its ~0.31 us
spin is pure launch latency (~45 us across the token).

## Relationship to the useful-body result

The useful-body reconciliation already counted this spin as shadow (not useful
work) and subtracted it from llama's node mass. This split does not change
that aggregate: llama's useful body stays ~3942.8-3982.5 us versus tinygrad's
4742.5 us, so tinygrad still does ~760-800 us more useful device work.

What changes is the interpretation of the lever. Because the spin is
dependency wait (anchors waiting for their quant providers), llama's overlap
is not a large hidden-parallelism win to copy; the launch-latency component is
only ~63 us per token. The forward lever remains reducing tinygrad's redundant
compute (reduce, residual, MMQ anchors, flash score), not adding overlap.

## Labels

- `observed`: ring wait-exit and trigger timestamps, CUPTI kernel intervals,
  the linear programmatic edge chain, spin totals.
- `inferred`: dependency wait and launch shadow, obtained by comparing each
  wait-exit to its producer's completion; producer-trigger lead.
- `unmeasured`: none for the headline split. Two kernels are excluded from the
  per-kernel table: local 0 has no producer inside a single replay (its
  producer is the previous graph), and local 751 (`get_rows_a`) has its
  wait-exit assigned to the overlapping `get_rows_b` by the shared H1
  assignment rule, a ~0.3 us tail artifact that does not move the totals.

The trigger timestamps are block-sampled, so the trigger lead is a lower
bound; this only strengthens the conclusion that the trigger fires well before
producer completion.

No production change or performance promotion follows from this record.

## Evidence

- Split JSON: `evidence/nv-llama-shadow-split-20260821/shadow-split.json`
  (SHA-256 `57d8657778f94871540d7b01c04386d206f01be66d7c249f548b9cc354c902e5`).
- Tool: `extra/llm_research/decode/nv_llama_shadow_split.py`.
- Inputs: both retained H1 captures (final subsampled and full sampling),
  whose SHA-256 hashes are embedded in the split JSON.
