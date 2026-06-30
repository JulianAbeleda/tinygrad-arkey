# Prefill P1/P2 BLOCKED — authoritative harness absent (not reproducible from model API)

**Verdict: PREFILL_P1_BLOCKED_NOISY_OR_STALE (sub-reason: authoritative harness missing) ; P2 not run.**

## What was tried (all 16–24× short of the ~3597 tok/s authority @ctx512)
| driver | tok/s @ctx512 | prefill ms | note |
|---|---|---|---|
| eager per-chunk m.forward (P0-turn tool) | ~217 | — | no JIT; Python/launch dominated |
| model.generate first-token, PREFILL_V2+CONCRETE_KV | 147 | 3487 | cold/recompiling |
| model.generate first-token, +PREFILL_SERVER_PROFILE, warmed 2× | 163 | 3140 | concrete-kv precompile ENGAGED ("precompile concrete jits, ON") yet still 22× short |
| **authority (aggressive-target-proof-20260624 artifact)** | **3597** | **~142** | produced by a harness NOT present in the repo |

## Root cause (precise)
1. The script that produced `bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_baseline.json` (the ~3597/3504/3248/2803 baseline) is **absent from the repo** — only the committed artifacts + a reader (`qk_prefill_theoretical_ceiling_audit.py`) remain. `git log` for those artifacts shows only `2478d7be4 [test] add ... decision snapshots` (snapshot commit, not the producer).
2. The model machinery exists (PREFILL_V2 + PREFILL_GRAPH_GEMM default-on + PREFILL_CONCRETE_KV precompile-at-load under PREFILL_SERVER_PROFILE, model.py:73–92) and **engages**, but end-to-end `model.generate` first-token wall is **22× slower** than the authority even fully warmed — so the authority number was measured by a **different methodology** (synced per-chunk GPU-time, excluding Python/launch/sampling/lm_head+gumbel overhead), encoded in the gone harness. `HARNESS_GUIDE.md` is decode-only; no prefill SOP documents the ~3597 methodology.

## What is needed to unblock (one of)
- Restore/locate the original whole-prefill harness script (synced GPU-time chunk measurement at ~3597 scale), OR
- A prefill HARNESS_GUIDE documenting the exact ~3597 methodology: synced vs wall, which overheads are excluded, warm/precompile sequence, chunk schedule, tok/s definition.

## Unaffected
P0 analytical ceiling (`PREFILL_P0_PASS_CEILING_PINNED`) stands — pure FLOP math, independent of this driver. The existing `qk_prefill_authority_refresh.py` / `qk_prefill_whole_role_attribution.py` remain the (defective) standalone-loop tools; NOT fixed, because the correct methodology is not recoverable from the repo. No speculative driver committed.
