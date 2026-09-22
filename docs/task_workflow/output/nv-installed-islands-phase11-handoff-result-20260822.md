# NV installed-island Phase 11 implementation handoff scopes

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

## Tier-0 disposition (2026-08-23, corrected)

Tier-0 is `PARTIAL_PASS`, and the verdict remains `240_UNMEASURED`. The prior
`240_BUILDABLE` upgrade is retracted. The campaign identified a specific
candidate schedule (K/V is serialized behind the Q branch, with ~14 us K/V and
~2.75 us flash predecessor gaps), but those gaps are outside the `P - C`
interval and do not partition `R`. The decisive cache-eviction arm of the
cold/hot probe was invalidated by a broken flush (only block 0 written), the
hot/cold rule was never reverse-bracketed, and no wait-exit or kernel-entry
observable was collected. See
`output/nv-r-residual-pdl-adjudication-result-20260823.md`. The next gates are
the corrected reverse-bracket cache test, predecessor-conditioned replay,
`P - C` closure to zero residual, and a scheduler microgate with a token-SHA
reverse wall promotion gate.

## Selection

The conservative named ceilings (`404.1 us`) do not by themselves close the
`604.756 us` required recovery. The production-conditioned residual `R`
(`~404 us`) must be adjudicated before the verdict can move off
`240_UNMEASURED`. The handoff is therefore two tiers:

Tier 0 (measurement gate): adjudicate `R` against a PDL/concurrency counter
experiment. This is the single prerequisite to a `240_BUILDABLE` verdict.

Tier 1 (implementation, ranked by non-overlapping wall leverage):

1. FFN GEMV DRAM streaming: `163.5 us`.
2. Q/K head norm + completion launch elimination: `96.1 us`.
3. vocab tail reduction topology: `58.3 us`.
4. flash combine body topology: `46.1 us`.
5. attn norm body (provider-preserving): `40.1 us`, coupled.

Every ceiling below is attribution, not booked recovery. No production file
was changed by this campaign.

## Exact-live disposition (2026-08-23)

* **[MEASURED]** Q cooperative occurrence-0 is closed by the exact-live
  predecessor-conditioned campaign with zero timing-identity residual and
  matching output/token SHA.
* **[INFERRED]** Its excess is tied to the live predecessor prefix; this does
  not establish a generic scheduler, cache, or fusion mechanism.
* **[UNMEASURED]** K/V and O remain open. They must use the amended protocol
  in `input/nv-r-residual-pdl-concurrency-adjudication-scope-20260822.md`,
  including same-session `P`, C0/C2/C3/C4/C5/C6, reverse arm order, raw
  evidence, and SHA validation. No wall recovery is booked until closure.

## Scope documents

| rank | scope | ceiling |
| --- | --- | ---: |
| 0 | `input/nv-r-residual-pdl-concurrency-adjudication-scope-20260822.md` | gates verdict |
| 1 | `input/nv-ffn-dram-streaming-scope-20260822.md` | 163.5 us |
| 2 | `input/nv-qk-norm-completion-launch-elimination-scope-20260822.md` | 96.1 us |
| 3 | `input/nv-vocab-tail-reduction-topology-scope-20260822.md` | 58.3 us |
| 4 | `input/nv-flash-combine-body-topology-scope-20260822.md` | 46.1 us |
| 5 | `input/nv-attn-norm-body-provider-scope-20260822.md` | 40.1 us |

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control)
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
