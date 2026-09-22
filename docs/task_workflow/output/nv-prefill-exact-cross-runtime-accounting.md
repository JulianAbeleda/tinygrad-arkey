# Exact NVIDIA pp512 cross-runtime lifecycle accounting

Status: **PASS for traced-mode accounting.** No profile percentage was scaled onto an unprofiled wall.

## Exact walls

| Boundary | tinygrad | llama.cpp | tinygrad - llama |
|---|---:|---:|---:|
| Unprofiled minimum | 69.378 ms | 34.680 ms | +34.698 ms |
| Unprofiled median | 69.492 ms | 35.019 ms | +34.473 ms |
| Profiled wall | 74.686 ms | 35.430 ms | +39.256 ms |
| Device interval union | 73.880 ms | 32.683 ms | +41.196 ms |
| Device idle in span | 0.307 ms | 0.201 ms | 0.106 ms |
| Host/boundary residual | 0.499 ms | 2.545 ms | -2.047 ms |

Each profiled wall closes as `GPU union + device idle + boundary residual`. The unprofiled minimum remains the performance authority; traced rows explain the profiled executions only.

## Device interval ledger

| Region | tiny active | llama active | active debt | tiny exclusive | llama exclusive |
|---|---:|---:|---:|---:|---:|
| input/embed/graph setup | 0.006 ms | 0.000 ms | 0.006 ms | 0.006 ms | 0.000 ms |
| norm/conversion | 1.400 ms | 2.146 ms | -0.745 ms | 1.400 ms | 2.109 ms |
| Q | 4.493 ms | 2.514 ms | 1.978 ms | 4.493 ms | 2.514 ms |
| K | 2.211 ms | 1.251 ms | 0.959 ms | 2.211 ms | 1.251 ms |
| V | 6.381 ms | 1.054 ms | 5.327 ms | 6.381 ms | 1.054 ms |
| Flash | 3.328 ms | 1.657 ms | 1.671 ms | 3.328 ms | 1.640 ms |
| O | 4.478 ms | 2.397 ms | 2.080 ms | 4.478 ms | 2.397 ms |
| gate | 12.762 ms | 6.087 ms | 6.675 ms | 12.762 ms | 6.083 ms |
| up | 13.006 ms | 6.111 ms | 6.895 ms | 13.006 ms | 6.111 ms |
| activation/multiply | 0.759 ms | 0.904 ms | -0.145 ms | 0.759 ms | 0.904 ms |
| down | 19.007 ms | 6.940 ms | 12.067 ms | 19.007 ms | 6.940 ms |
| residual/RoPE/KV/support | 3.119 ms | 1.379 ms | 1.739 ms | 3.119 ms | 1.320 ms |
| final-row gather/prune | 0.000 ms | 0.005 ms | -0.005 ms | 0.000 ms | 0.001 ms |
| vocabulary | 2.922 ms | 0.313 ms | 2.608 ms | 2.922 ms | 0.311 ms |
| output/token transfer | 0.009 ms | 0.000 ms | 0.009 ms | 0.009 ms | 0.000 ms |

All 1,449 tinygrad and 1,186 llama launches are classified; unknown count is zero on both sides.

The active columns may overlap and are not an additive wall charge. Exclusive columns omit shared intervals; the JSON retains the exact overlap sets and per-layer launch rows.

## Direct reading

The largest measured traced-mode debts are down, gate/up, V, Q/O, vocabulary, support, Flash, and K—in that order after combining paired roles. Tinygrad is faster in normalization/conversion and activation/multiply. Device idle is only about 0.1 ms worse, so the main loss is executed kernel service, not an idle GPU.

llama executes full M=512 FFNs for 35 layers, then gathers the final row and runs layer 35 FFN at M=1. Tinygrad runs the final FFN at M=512. That fact is included in the gate/up/down totals rather than double-booked as a separate recovery.

## Causality boundary

This trace proves where time was spent in the two instrumented executions. It does not claim that replacing one region recovers the full active-time difference in the unprofiled graph. Only fresh, correctness-qualified whole-model rollback brackets can assign that causal wall recovery.

## Evidence

- `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/tinygrad/tinygrad-safe-cut-accounting.json`
- `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/llama/llama-accounting.json`
- `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/cross-runtime-accounting.json`
- `docs/task_workflow/output/nv-prefill-exact-cross-runtime-trace-scope.md`
