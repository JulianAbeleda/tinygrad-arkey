# Flash entry-hop ledger

## Decision

The score kernel enters Flash with inherited cache state. Its immediate
Q/K/V-ready edge is not the source of the production tax.

The first causal hop is the **previous layer's FFN-down projection**. Gate/up
alone leaves the reheated score almost hot. Gate/up plus FFN-down crosses the
device's L2-capacity neighborhood, evicts useful Flash input lines, and adds
0.832 us/layer at the next score. The later Q projection adds 0.272 us/layer.
Provider, K/V projection, and completion work add no material tax; the
completion kernels slightly reheat current-token state.

The complete captured entry chain adds 1.200 us/layer versus hot, or 43.200
us/token across 36 layers. This is exposure, not booked recovery. No code or
installed endpoint changes as a result of this ledger.

## Exact entry chain

The captured common-layer path is:

```text
previous layer gate/up
  -> previous layer FFN-down + residual
  -> next layer shared RMS/Q8 provider
  -> Q projection
  -> paired K/V projection
  -> Q norm + RoPE completion
  -> K norm + RoPE + K/V cache completion
  -> Flash score
```

The replay retains the production cubins, launch dimensions, scalar values,
and buffer identity. Shared producer/output buffers therefore alias exactly as
they do in the graph. Every sample executes:

```text
untimed score reheat -> cumulative production prefix -> timed score
```

Only the final score is timestamped. Each native arm has 40 retained samples
after eight warmups.

## Hop-by-hop timing and residency

| cumulative checkpoint before score | score, native | score - hot | marginal tax | target DRAM reads | target L2 read hit |
|---|---:|---:|---:|---:|---:|
| hot score reheat only | **4.576 us** | reference | reference | 0.001 MB | 100.00% |
| previous gate/up | 4.768 us | +0.192 us | +0.192 us | 0.003 MB | 99.98% |
| previous gate/up + FFN-down | **5.600 us** | **+1.024 us** | **+0.832 us** | **3.692 MB** | **78.79%** |
| plus shared provider | 5.600 us | +1.024 us | +0.000 us | 3.702 MB | 78.80% |
| plus Q projection | 5.872 us | +1.296 us | +0.272 us | 4.232 MB | 75.32% |
| plus paired K/V projection | 5.920 us | +1.344 us | +0.048 us | 4.234 MB | 75.31% |
| plus Q completion | 5.824 us | +1.248 us | -0.096 us | 4.217 MB | 75.37% |
| plus K/V completion: full entry | **5.776 us** | **+1.200 us** | **-0.048 us** | **4.213 MB** | **75.43%** |

Native HCQ brackets are the timing authority. The counter columns come from a
separate steady-state NCU application replay: nine conditioning repetitions,
the first 17 matching score launches skipped, the final score captured, and
cache flushing disabled. NCU explicitly warns that caches are uncontrolled;
the counters are therefore used to locate the residency transition, not to
replace the native timings.

The score's executed instructions, L1 traffic, and L2 traffic remain
essentially fixed across all NCU arms: 1,097,088 instructions, 22.086 MB at
L1, and approximately 17.46 MB at L2. What changes is where the fixed read
demand is served. The prior FFN checkpoint turns roughly 3.7 MB of target
reads into DRAM service, and Q raises that to roughly 4.2 MB. This is why the
same score body becomes slower without changing its arithmetic or launch
geometry.

NCU's target-attributed DRAM writes are dirty-cache writebacks created by the
conditioning sequence and vary by arm. They are not interpreted as the
score's logical output size.

## Where the tax is created and paid

The causal and payment hops are different:

| lifecycle event | accounting |
|---|---|
| score reheat | establishes the hot K/V and partial state |
| gate/up weight stream | 54 MiB of packed weights; below the observed knee by itself |
| FFN-down weight stream | adds 39.375 MiB; gate/up + down reach 93.375 MiB before other live data and approach the 96-MiB L2 capacity |
| Q weight stream | adds another 9 MiB of displacement pressure |
| Q/K/V completions | touch the current Q and cache entries; they do not restore the historical K/V horizon and slightly reduce the measured penalty |
| score | first consumer that demands the displaced K/V lines, so its counters and duration pay the inherited miss tax |

This reconciles the previously confusing observations. An isolated immediate
Q/K/V prefix is neutral because the score was not first displaced by the
large prior-layer stream. A synthetic 96-MiB conditioner works because it
recreates that capacity state. The exact production chain now identifies the
specific stream that crosses the knee.

## Llama-relative accounting

The exact tinygrad entry chain costs 1.200 us/layer. The retained matched
llama S8 synthetic conditioner costs 0.608 us/layer. These are not identical
production chains, so their difference is a ceiling rather than a booking,
but it is the best current equal-geometry cold-sensitivity comparison:

| exposure | latency/token | endpoint ceiling | gain from 244.230 tok/s |
|---|---:|---:|---:|
| make the exact tinygrad entry fully hot | 43.200 us | 246.834 tok/s | +2.604 tok/s |
| match llama's measured S8 cold sensitivity | 21.312 us | 245.508 tok/s | +1.278 tok/s |
| booked by this test | **0.000 us** | **244.230 tok/s remains installed** | **0.000 tok/s** |

Even the full-hot score ceiling does not by itself reach the retained llama
endpoint of 248.711 tok/s. Flash combine and the rest of the lifecycle still
matter. Conversely, the 21.312-us llama-relative exposure is narrower and
more admissible than claiming the entire hot-to-production delta.

The matched llama production-order replay resolves an important ambiguity in
that ceiling. Llama also loses K/V residency after the FFN prefix: its Flash
target reads about 3.18 MB from DRAM after gate/up plus down. But that residency
loss costs only 0.128 us of target service. The later Q projection adds 0.832 us
without adding meaningful target DRAM traffic, and Q completion gives 0.192 us
back. A separate capacity sweep is flat through 54 MiB, bends at 90 MiB,
crosses a sharp knee between 90 and 92 MiB, and then plateaus through 108 MiB.
Llama therefore does **not** clear or pay the producer footprint one-to-one;
ordinary cache replacement produces a threshold-and-plateau response.

On the matched full prefix, tinygrad pays 1.200 us/layer and the corrected
production-flags llama replay pays 0.736 us/layer. The 0.464-us/layer excess is
16.704 us/token, a 245.230-tok/s ceiling
from the installed endpoint. It remains a zero booking because the retained
llama replay models the small completion hops, and because fewer Flash DRAM
bytes are not sufficient unless target service time also falls. The complete
cross-runtime accounting is in
`docs/task_workflow/output/nv-flash-entry-hop-vs-llama-result.md`.

This prefix comparison is not the complete kernel-to-production conversion.
Tinygrad falls 1.648889 us/layer from hot to production, while llama falls
0.654--0.718 us/layer. The excess conversion is 33.500--35.804 us/token; the
prefix explains only part of it. See
`docs/task_workflow/output/nv-flash-kernel-to-production-conversion-result.md`.

## What is now closed

- The provider-to-score handoff is not a large entry tax.
- The immediate paired K/V projection is not the first eviction event.
- Q/K/V completion kernels are not making Flash cold.
- The visible ready-to-score timestamp interval remains scheduling metadata,
  not a new recovery pool.
- A different score arithmetic body is not required to explain this penalty;
  the same target instruction and byte demand is served from a colder level.
- Blanket `evict-first` across all dense weights remains closed because it
  harms useful packed-weight reuse in gate/up and down.

## Next admissible construction

The next test should be scoped specifically to the stream that crosses the
knee, not to every weight consumer. It must track the residency ledger and the
service-time ledger separately:

1. Split the prior FFN-down loads by reuse class.
2. Preserve ordinary caching for packed headers/data while they still have
   intra-kernel reuse.
3. Apply an evict-first or last-use policy only after a line's final useful
   consumption, or protect the active K/V window without changing projection
   service.
4. Re-run this entry ledger first. A candidate must lower both the full-entry
   score's DRAM reads and native 1.200-us/layer penalty without slowing gate/up
   or down. A bytes-only change is a no-go.
5. Only then run exactness, graph census, and a reverse token-wall bracket.

This is a new policy granularity. It does not reopen the already-failed
whole-dense streaming switch.

## Evidence

- `docs/task_workflow/evidence/nv-flash-entry-hop-ledger/entry-native-r1.json`
- `docs/task_workflow/evidence/nv-flash-entry-hop-ledger/entry-hop-summary.json`
- `docs/task_workflow/evidence/nv-flash-entry-hop-ledger/ncu-steady-*.csv`
- `docs/task_workflow/output/nv-flash-entry-hop-vs-llama-result.md`
- `docs/task_workflow/output/nv-flash-kernel-to-production-conversion-result.md`
- `extra/llm_research/decode/nv_flash_entry_hop_ledger.py`
- `extra/llm_research/decode/nv_flash_entry_hop_cuda.py`
- `extra/llm_research/decode/nv_flash_entry_hop_counter_summary.py`
