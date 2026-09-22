# NV Q6 oracle combined-publication decision

## Question

Does correcting the broad rolling-prefetch CTA to llama's exact four-barrier
K256 lifecycle materially improve it?

The control fenced Q6 publication and Q8 panel 0 separately, then fenced the
end of half 0 and Q8 panel 1 publication. It had four barriers but no explicit
end-of-K256 lifecycle barrier. The candidate instead uses the audited oracle
order:

```text
publish Q6 + Q8 panel 0 -> barrier
consume half 0          -> barrier
publish Q8 panel 1      -> barrier
consume half 1          -> barrier
```

All other work is held fixed: canonical inputs, `128x128xK256` ownership,
expanded 76-word Q6 rows, rolling 64-value FP32 accumulator bank, Q8 panel-1
register prefetch, 256 IMMA, 32 LDSM, and 58,368 bytes of launch shared memory.

## R31 results

| population | control median | corrected median | delta |
|---|---:|---:|---:|
| one CTA | `13.024 us` | `13.152 us` | `+0.128 us` (`+0.98%`) |
| 170 CTAs | `18.848 us` | `19.168 us` | `+0.320 us` (`+1.70%`) |

Both arms are exact. At 170 CTAs this compares all 2,785,280 FP32 outputs.
Both arms use 255 registers, zero stack bytes, zero local static bytes, and
zero LDL/STL in every SASS region.

The corrected arm preserves 256 IMMA, 32 LDSM, and four BAR instructions. Its
static instruction body grows from 3,560 to 3,624 instructions, principally
because the explicit terminal dependency materializes 65 additional MOV
instructions while removing one PRMT. It projects `552.189 us` for the full
main, or `-233.389 us` recovery from the current `318.8 us` route. This misses
the required `+23.5 us` recovery by a decisive margin.

## Decision

`NO_GO_COMBINED_PUBLICATION`

Correct barrier placement is part of llama's correctness/lifecycle contract,
but it is not a performance lever in the generated broad CTA. Do not integrate
this candidate into the 170-owner Stream-K route. The next audit must compare
the instruction dependency schedule inside the Q6 producer and consumer body;
barrier count and placement are now causally closed.

## Evidence

- `docs/task_workflow/evidence/nv-q6-oracle-combined-publish-1-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-combined-publish-170-20260831/result.json`
- `extra/llm_research/prefill/nv_q6_oracle_broad_cta.py`
- `extra/llm_research/prefill/bench_nv_q6_oracle_broad_cta.py`
