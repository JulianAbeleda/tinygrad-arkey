# Phase L0 — the loop is LIVE (make-or-break): PASS

Date: 2026-06-15. `extra/qk_loop_live.py`, fresh shape **(M=4096, K=14336, N=128)** — a real FFN GEMM
absent from the 26-shape N1 corpus (corpus has K=14336 at N=64 and N=256, not 128).

## What changed vs N2
N2 proved the guided loop works but **looked up** already-measured device times. L0 **times candidates
LIVE on device** (the exact `_time_program` path the dataset was built with), on a shape the model has
never seen, and measures the real wall-clock autotuning win. This converts the offline simulation into a
working tool.

## Result (PASS on all 4 pre-registered gate criteria)
| metric | guided (model) | random | oracle |
|---|---|---|---|
| frac of oracle @ top-1 | **0.911** | 0.597 | 1.0 |
| frac of oracle @ top-8 | **0.979** | 0.861 | 1.0 |
| live timings to reach 95% of oracle | **5** | ~49 (expected) | — |
| wall-clock to run the budget | **0.89 s** (top-8) | — | 36.62 s (all 277) |

- **41.2× wall-clock speedup**: the model reaches 97.9% of the best-of-277 by timing 8 configs (0.89 s)
  vs the exhaustive sweep (36.62 s).
- **The model's single top pick is already 91% of oracle** — the learned ranking transfers to a fresh
  shape on live silicon, not just in the offline lookup.
- 247/277 configs compiled on this shape; the 30 compile failures are dropped from the oracle/ranking
  (handled, not crashed).

## Honesty / caveats (pre-registered)
- **Live noise is real and visible**: a first run gave guided@8=1.0 / random-trials≈123; this run
  0.979 / 49.4. Both PASS the gate decisively; the offline 0.92/86× reproduces in spirit, with the
  expected device-timing wiggle. We report the live numbers, not the rosier first draw.
- This is an **autotuning search wall-clock win on a fresh shape** — NOT a llama.cpp decode win. The
  on-target quantized-decode spaces remain dead (final report); the loop's home is native-matmul
  autotuning, and L0 shows it works live there.
- Single shape — L1 generalizes to 4–6 fresh shapes before any broad claim.

## Reproduce
`DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_live.py` → `result.json`. Fixed seed (20260615) →
deterministic model ranking; live device times vary run-to-run (the caveat above).
