# Prefill: graph-GEMM vs Tensile route — whole-prefill validation (2026-06-23)

## Verdict: `NO_PRODUCTION_TENSILE_GAP` + `RELOCATION_DOES_NOT_TRANSFER` — the isolated Tensile gap is a benchmarking artifact
A machine-search-style validation pass (full 8B, clock-pinned, 3 repeats) answers "can instruction scheduling close
the prefill→Tensile gap?" with **NO**, and overturns the premise: there is **no production Tensile gap to close**. The
graph-GEMM route already **beats** the Tensile route at whole-prefill by **~6% at every context**, even though Tensile's
*isolated* GEMM is ~10% faster (≈66 vs ≈60 TFLOPS). Tensile's raw-throughput edge does not survive in-model integration.

## Setup
RX 7900 XTX / gfx1100, clock-pinned (`rocm-smi --setperflevel high`). `DEV=AMD JIT=1 PREFILL_V2=1`, `PYTHONPATH=.`.
Model Qwen3-8B-Q4_K_M (no OOM; full 8B). Harness `extra/qk_prefill_whole_synced.py` (synced multi-chunk whole-prefill).

## Whole-prefill, all routes (tok/s)
| ctx | graph-GEMM (default) | Tensile route | graph advantage |
|---|---|---|---|
| @512 | 3720 | 3506 | +6.1% |
| @1024 | 3635 | 3429 | +6.0% |
| @2048 | 3399 | 3200 | +6.2% |
| @4096 | 2997 | 2793 | +7.3% |

`PREFILL_GRAPH_GEMM=0 PREFILL_TENSILE_GEMM=1` selects the Tensile route. It is uniformly slower in-model.

## Whole-prefill@4096, 3 repeats (median ± spread)
| config | median | min–max | Δ vs baseline |
|---|---|---|---|
| graph-GEMM baseline | 2994 | 2990–2994 | — |
| + `PREFILL_GEMM_RELOC=1 MAX_WGS=1` | 2984 | 2982–2993 | −0.33% |
| + `PREFILL_GEMM_RELOC=1 MAX_WGS=4` | 2988 | 2987–2999 | −0.20% |
| Tensile route | 2800 | 2795–2800 | −6.48% |

The relocation deltas (−0.2…−0.33%) are within per-config run-to-run noise (ranges overlap baseline) → **no meaningful
transfer**. The Tensile deficit (−6.5%) is consistent and tight.

## Isolated kv_halved occupancy sweep (M512×N1024×K4096, vary PAD → LDS → occupancy)
| LDS | WG/CU | base | reloc | speedup |
|---|---|---|---|---|
| 15360 | 4 | 17.14T | 17.52T | +2.24% |
| 30720 | 2 | 16.78T | 17.06T | +1.65% |
| 61440 | 1 | 15.22T | 15.84T | +4.12% |

Confirms the lever is **occupancy-driven** (benefit ∝ 1/occupancy; biggest at 1 WG/CU). All correct (rmse 2.08e-4),
spreads ±1 µs. At this realistic K=4096 shape it stays positive at every occupancy; the sign only flips negative on
small-K/few-iteration shapes where the per-iteration extra-waitcnt overhead dominates.

## What it means
- The asm-scheduler arc (Inc 0–3) and the prior audit targeted **Tensile parity**, measured on isolated GEMMs. That
  target is a benchmarking artifact: in-model, **routing prefill through Tensile makes the model ~6% slower** — the
  GEMM's raw-throughput edge is eaten by Tensile's integration cost (extra layout/transpose kernels, weaker fusion),
  while graph-GEMM fuses cleanly and carries the kv de-WG-starve fix.
- So there is **no production prefill-GEMM problem to solve.** Graph-GEMM is already the faster *route* (and at
  parity-to-ahead of llama.cpp — pp512 3720 vs llama ~3327). This **retires the older "graph-GEMM ~99.5% of Tensile,
  close the gap" framing** — at whole-prefill graph-GEMM is *ahead* of Tensile, not behind.
- "Isolated kernel wins don't transfer" ([[inference-perf-measured-map]]) holds in BOTH directions here: Tensile's
  isolated win doesn't transfer, and relocation's isolated +2–4% doesn't transfer (<0.3% in-model).
- **Practical upshot:** stop optimizing the prefill GEMM kernel for Tensile parity — solved. Remaining prefill headroom
  is in the NON-GEMM path (attention, which drives the @512→@4096 decay 3720→2997) or in-model integration, not the
  matmul.

## Final verdict
**Can instruction scheduling close the prefill→Tensile gap? NO** — confidence HIGH (tight spreads <0.5%, 3 consistent
rounds + isolated ±1 µs). And there is **no production Tensile gap** to close: graph-GEMM already beats the Tensile route
in-model by ~6%. Caveat: single @4096 endpoint + one kv shape for the isolated sweep; 8B/Q4_K_M only; chunked-512
whole-prefill methodology (not single-batch like llama-bench).

## Provenance
Isolated sweep + whole-prefill A/B/C/D (3 repeats) measured 2026-06-23; raw timing not committed (non-deterministic).
`PREFILL_GEMM_RELOC` stays default-off. No `tinygrad/` source, no production change.
