# NV useful-body reconciliation result 20260821

## Verdict

Applying H1's measured shadow share to the authority overlap mass, llama's
**useful kernel-body time is 3942.8-3982.5 us** per token while tinygrad's is
its full node mass of **4742.5 us** (tinygrad overlap is ~0). Tinygrad
therefore does **760-800 us more useful device work per token** than llama.

The prior wall framing ("llama hides work in overlap") is too generous to
llama. 91.9-95.4% of llama's 1133 us overlap mass is dependency wait plus
launch shadow, not simultaneous useful traffic. The gap is kernel-body
efficiency, not overlap harvesting.

## Aggregate

| quantity | tinygrad | llama | delta (tinygrad - llama) |
| --- | ---: | ---: | ---: |
| node sum us | 4742.464 | 5023.823 | -281.359 |
| overlap mass us | 9.214 | 1133.255 | -1124.041 |
| overlap shadow share | ~0 | 0.9189-0.9540 | - |
| overlap shadow us | ~0 | 1041.4-1081.1 | - |
| useful body us | 4742.464 | 3942.8-3982.5 | +760.0 to +799.7 |

Useful body is `node_sum - shadow`; for tinygrad the near-serial node mass is
already useful. The delta bracket sums the family rows exactly (identity
closes to <0.02 us).

## Per-family useful-work delta

Sums are authority us per token. Llama useful body is `node_sum - shadow`;
shadow is attributed per family by the same time-sliced wait-exit sweep H1
uses, so the rows sum to the aggregate bracket.

| family | tinygrad useful | llama useful | delta (tinygrad - llama) |
| --- | ---: | ---: | ---: |
| gemv (Q/O/gate/up/down anchors) | 2949.44 | 2604.3-2629.7 | +319.7 to +345.2 |
| reduce | 234.30 | 0 | +234.3 |
| residual | 219.17 | 0.79-0.80 | +218.4 |
| flash_score | 267.78 | 156.0-158.8 | +109.0 to +111.8 |
| gemv_kv | 298.43 | 216.2-220.4 | +78.1 to +82.3 |
| flash_combine | 105.54 | 56.1-57.8 | +47.7 to +49.5 |
| rmsnorm | 323.39 | 308.0 | +15.4 |
| vocab | 313.54 | 301.0-301.2 | +12.3 to +12.5 |
| kv | 0 | 31.5-32.7 | -32.7 to -31.5 |
| rope | 0 | 71.1-72.0 | -72.0 to -71.1 |
| quant_provider | 30.88 | 197.7-201.1 | -170.2 to -166.9 |

The solid tinygrad-excess rows (convention-independent) are reduce +234 us and
residual +218 us, both families llama folds into epilogues. flash_score adds
~109-112 us and the MMQ anchors add ~320-345 us.

## Anchor correction

The authority ledger's "-50 us anchors" row is a wall view, not a body view.
Llama's 144 MMQ anchors are serial in interval space (zero anchor-anchor
overlap), but each anchor launches early and spins behind its quant provider,
which always launches first in llama's graph. That spin (~368-394 us across the
token) is hidden behind the quant/norm providers and therefore never shows on
the device union.

So llama's anchor wall of 2997.9 us decomposes into ~2604-2630 us of MMQ
compute plus ~368-394 us of hidden spin, while tinygrad's 2949.4 us of anchor
mass is all compute (no PDL spin). In compute terms tinygrad's MMQ anchors are
~320-345 us slower, about 12-13% per anchor (20.5 us vs 18.1-18.3 us). The
wall advantage the ledger attributed to tinygrad was an artifact of llama's
spin hiding.

## Grounding and forward direction

- The serialization counterfactual (llama would lose ~52-91 us useful bracket,
  not ~1127 us) is confirmed; overlap parity is a bounded target.
- The lever is reducing tinygrad's actual compute in the MMQ anchors, the
  output/attention reductions (`reduce`), the explicit residual adds
  (`residual`), and flash score, not adding overlap.
- Roofline: tinygrad runs ~1065 GB/s effective versus llama ~1167 GB/s against
  the 1700-1792 GB/s measured peak. Both leave headroom, but the useful-body
  view says tinygrad's deficit is extra real traffic/work, not idle overlap.

## Labels

- `observed`: ring wait-exit timestamps, CUPTI intervals, authority ledger mass,
  H1 shadow share, the family-row identity.
- `inferred`: per-family shadow/useful and the delta bracket, obtained by
  applying measured shares to the authority overlap mass.
- `unmeasured`: the split of llama shadow into dependency wait versus host
  launch; a tinygrad-side wait-exit census on the decode route.

The retained dump's raw edge table is a single linear programmatic chain
`0 -> 1 -> ... -> 761` (761 `type=1`, port `1 -> 0` edges), so the
producer-consumer mapping needed for the shadow split is known: each kernel's
`cudaGridDependencySynchronize` waits for the previous kernel's
`cudaTriggerProgrammaticLaunchCompletion`. Both trigger (`kind=0`) and wait-exit
(`kind=1`) ring records are retained. The remaining blocker is confirming the
instrumented trigger line semantics (launch-completion at kernel start versus
data-ready at kernel end) against the instrumented llama source, which is not
present in this workspace.

No production change or performance promotion follows from this record.

## Evidence

- Reconciliation JSON:
  `evidence/nv-useful-body-reconciliation-20260821/reconciliation.json`.
- Tool: `extra/llm_research/decode/nv_useful_body_reconciliation.py`.
- Inputs: the authority ledger and both retained H1 captures, with SHA-256
  hashes recorded inside the reconciliation JSON and the evidence manifest.
