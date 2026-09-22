# Flash kernel-to-production conversion versus llama

## Decision

The original intuition was directionally correct: tinygrad has a much larger
hot-kernel-to-production fall-off than llama. But llama's fall-off is not zero.

| runtime | hot score | production score | production penalty | relative fall-off |
|---|---:|---:|---:|---:|
| tinygrad | 4.536 us/layer | 6.184889 us/layer | **+1.648889 us/layer** | **+36.35%** |
| llama | 3.808--3.872 us/layer | 4.526333 us/layer | **+0.654--0.718 us/layer** | **+16.90--18.86%** |

Llama therefore falls roughly half as much in percentage terms and pays only
40--44% as much absolute conversion latency. It does not move from an isolated
kernel to production for free.

## Why the earlier accounting was confusing

Two boundaries had been mixed:

```text
tinygrad hot score:       about 4.54 us/layer
llama production score:   about 4.53 us/layer
```

Those numbers are nearly equal, but they are not a hot-to-hot comparison. The
correct production-flags llama hot body is 3.81--3.87 us/layer.

The first llama entry harness also omitted llama's production CUDA release
flags, most importantly `-use_fast_math`. It measured a 4.096-us hot body and
executed 894,720 dynamic instructions. Rebuilding the same source with the
library's actual flags lowers the hot body to 3.872 us and the dynamic count to
850,944 instructions. The corrected full-prefix target is 4.608 us.

The missing flags barely change the within-harness cold penalty (0.768 becomes
0.736 us/layer), so the cache-knee conclusion survives. They materially change
the absolute hot baseline and therefore the kernel-to-production story.

## Corrected llama entry curve

| cumulative checkpoint | corrected llama Flash | from hot |
|---|---:|---:|
| hot | **3.872 us** | reference |
| gate/up | 3.872 us | +0.000 us |
| plus FFN-down | 4.000 us | +0.128 us |
| plus attention input | 4.000 us | +0.128 us |
| plus Q | 4.832 us | +0.960 us |
| plus Q completion | 4.640 us | +0.768 us |
| plus K | 4.608 us | +0.736 us |
| plus V | 4.608 us | +0.736 us |
| full entry | **4.608 us** | **+0.736 us** |

Llama's PDL-off production trace averages 4.526 us/layer and has a 4.512-us
median. Thus the corrected full-prefix replay is within 0.082 us/layer of the
actual production mean. The producer replay is now a good production-state
model for llama.

## Direct production-state test

An NCU application-replay capture targeted one Flash launch in llama's actual
PDL-off decode graph after graph capture. Cache control was disabled.

| counter | actual llama graph | corrected full-prefix replay |
|---|---:|---:|
| Flash DRAM reads | 3.166 MB | about 3.16 MB |
| L2 read hit | 75.58% | 75.52% |
| L2 traffic | 13.208 MB | about 13.16 MB |
| dynamic instructions | 850,944 | 850,944 |
| long-scoreboard stalls | 61.03% | 59.42% |

Llama production is not preserving a hot K/V target. It reaches Flash in the
same cold residency and instruction state as the corrected prefix replay. Its
advantage is that the production-compiled body starts faster and converts the
cold state into less additional time.

## Where the production score gap comes from

The measured score-only production gap is 59.708 us/token. Aligning both
boundaries decomposes it as:

| component | llama-relative tinygrad debt | share of score gap |
|---|---:|---:|
| hot body | 23.904--26.208 us/token | 40--44% |
| kernel-to-production conversion | **33.500--35.804 us/token** | **56--60%** |
| total production score | **59.708 us/token** | 100% |

So the larger tinygrad conversion remains the majority of the score gap, but
the hot bodies are not actually tied. Llama also has a genuine 24--26-us/token
hot-body advantage after compile flags and boundaries are aligned.

Tinygrad's exact one-layer entry replay explains 1.200 of its 1.648889-us/layer
production conversion. The remaining 0.448889 us/layer, or 16.160 us/token,
does not appear in that isolated entry chain. By contrast, llama's corrected
0.736-us/layer prefix penalty accounts for essentially all of its measured
0.654--0.718-us/layer conversion.

That asymmetry is the next useful discriminator: tinygrad has an additional
full-graph service state beyond the already-mapped producer prefix.

## Token translation

Matching only llama's kernel-to-production conversion would expose
33.500--35.804 us/token. Applied mechanically to the installed 4.094502-ms
endpoint, that is a **246.245--246.384 tok/s ceiling**, or approximately
+2.01--2.15 tok/s. Zero is booked.

Eliminating the entire 59.708-us score gap, including the hot-body difference,
would be a 247.844-tok/s ceiling. That is not an independent booking and still
does not include the remaining combine debt.

## Full-history follow-up

The requested follow-up is complete. Exact serialized history reaches 5.712
us, while restoring the production Q versus K/V fork/join reaches 6.240 us,
within 0.096 us/layer of the retained production layer's 6.144-us median. A
target-working-set reheat after the fork returns the score to 4.608 us.

The formerly unexplained residual is therefore topology-conditioned cache
residency, not an intrinsic graph-launch tax or a need for still older history.
The full result and its token translation are in
`docs/task_workflow/output/nv-flash-full-history-topology-result.md`. Zero is
booked; a useful construction must improve the cold score path without giving
back Q/K/V overlap.

## Evidence

- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/summary.json`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/llama-production-layer18.ncu-rep`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/llama-entry-prodflags-*.sqlite`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/llama-entry-prodflags-full-counter-steady.ncu-rep`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/llama-prodflags-tc768-*.sqlite`
- `docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-llama-pdl0-dag.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/priority1-conditioning-r3.json`
- `extra/llm_research/decode/nv_flash_kernel_to_production_conversion.py`
- `docs/task_workflow/output/nv-flash-full-history-topology-result.md`
