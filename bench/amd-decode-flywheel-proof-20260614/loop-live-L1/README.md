# Phase L1 — the live loop generalizes across fresh shapes: PASS

Date: 2026-06-15. `extra/qk_loop_live.py --l1`. Five GEMM shapes, each absent from the 26-shape N1
corpus, each trained leave-it-out and timed LIVE on device (277 configs/shape, 1385 total timings).

## Result (PASS, all 4 pre-registered gate criteria, aggregate over 5 fresh shapes)
- **mean guided@8 = 0.977** of live oracle vs **random@8 = 0.821**
- **median 3 live timings** to reach 95% of oracle vs **~82** expected for random
- **mean 42× wall-clock speedup** (guided top-8 vs exhaustive 277 sweep)

| fresh shape (M,K,N) | guided@1 | guided@8 | random@8 | k→95% | wall speedup | n_valid |
|---|---|---|---|---|---|---|
| 8192, 8192, 256 | 0.991 | **1.00** | 0.828 | 1 | 25.6× | 262 |
| 11008, 4096, 128 | 0.856 | 0.965 | 0.777 | 3 | 46.4× | 241 |
| 5120, 5120, 128 | 0.745 | **1.00** | 0.857 | 4 | 44.7× | 247 |
| 13824, 5120, 128 | 1.000 | **1.00** | 0.806 | 1 | 48.3× | 247 |
| 4096, 11008, 64 | 0.796 | 0.920 | 0.836 | 12 | 45.0× | 187 |

## Honest reading
- **Generalizes**: on unseen shapes, live, the model's top-8 reaches 92–100% of the best-of-277, beating
  random everywhere, and the search wall-clock drops ~40× — the N1/N2 offline result holds on real
  silicon for shapes the model never trained on.
- **The weak spot is small-N (N=64)**: `(4096,11008,64)` needs 12 timings to clear 95% (guided@8=0.92),
  and only 187/277 configs even compile there. This is the under-sampled small-N regime the postmortem
  flagged; it is the one case where the budget-8 tool lands at 0.92 rather than ≥0.95. The aggregate
  gate uses the MEDIAN k→95 (3) and MEAN guided@8 (0.977), both comfortably passing, but the small-N
  tail is real and not hidden: a budget-8 autotuner is excellent for N≥128 and merely good at N=64.
- Still an **autotuning search win on fresh shapes**, not a llama.cpp decode win (unchanged).

## Reproduce
`DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_live.py --l1` → `result.json`. Fixed seed
(20260615) → deterministic ranking; live device times vary run-to-run.
