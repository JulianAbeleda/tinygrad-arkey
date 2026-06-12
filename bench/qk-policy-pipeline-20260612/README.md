# QK Policy Pipeline Rerun

Date: 2026-06-12

This directory is the reproducible generated-policy gate for Qwen3 Q4_K_M on
native Ubuntu AMD/gfx1100. It generates a policy, checks parity against the
explicit Q4/Q6 primitive flags, runs repeated decode, runs greedy output A/B,
and profiles large accepted wins.

The decision uses the latest stable three-run window. If a sample has a late
collapse, the pipeline can add up to two extra samples and decide on the latest
stable window rather than accepting a noisy mean.

| model | status | decision window | explicit tok/s | generated tok/s | gain | llama.cpp reference | generated % llama.cpp | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-8B-Q4_K_M | accept | explicit2-4 vs generated3-5 | 49.61 | 52.65 | 6.14% | 101.2 | 52.0% | modest generated-policy win; no profile because gain is below the 20% profile gate |
| Qwen3-14B-Q4_K_M | accept | explicit1-3 vs generated3-5 | 22.53 | 39.99 | 77.48% | 65.8 | 60.8% | strong generated-policy win; profile reused after adaptive rerun |
| Qwen3-32B-Q4_K_M | blocked | n/a | n/a | n/a | n/a | 30.8 | n/a | policy generation/parity succeeded, but decode OOMs during primitive storage install |

## 32B Memory Verdict

The 32B generated-policy search completed and produced a runtime-supported
policy:

- explicit Q4/Q6 policy would install 320 wrappers;
- generated policy would install 448 wrappers;
- generated unsupported winners: 0.

Decode does not start. The explicit primitive load failed with:

```text
MemoryError: Allocation of 70.31 MB failed on AMD. Used: 23.80 GB
```

This means the next 32B task is not another policy search. It is reducing or
eliminating duplicate GPU storage for primitive-packed weights, or adding a
memory-aware policy cap. Until then, 32B cannot answer the scaling question.

## Artifacts

- `8b/README.md`
- `14b/README.md`
- `32b/README.md`
- `8b/decision.json`
- `14b/decision.json`
- `32b/decision.json`
