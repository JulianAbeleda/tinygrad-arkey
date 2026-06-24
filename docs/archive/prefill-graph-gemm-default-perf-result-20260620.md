# Prefill Graph GEMM Repeated Performance Result (Gate 1) - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_REPEATED_PERF`

Run:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_default_perf.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

3 paired subprocess sessions (alternating order), each measured with the **synced arbiter** (K=8 forwards
back-to-back, one `dev.synchronize()`, total/K = true GPU throughput) — the trustworthy method, not the nosync
`realize()` loop that produced the prior numbers.

| session | baseline ms/512 | graph ms/512 | synced speedup | nosync speedup |
|---:|---:|---:|---:|---:|
| 0 | 415.2 | 256.7 | 1.617 | 0.97 |
| 1 | 414.2 | 257.8 | 1.607 | 1.026 |
| 2 | 414.8 | 258.3 | 1.606 | 1.00 |

| metric | value | threshold | pass |
|---|---:|---:|---|
| paired sessions | 3 | ≥3 | ✓ |
| median synced speedup | **1.61×** | ≥1.5× | ✓ |
| worst paired synced speedup | 1.606× | ≥1.25× | ✓ |
| graph p50 ms/512 | 257.8 | < baseline p50 (414.8) | ✓ |
| run failures | 0 | 0 | ✓ |

## The sync question is answered (and the prior number corrected)

Gate 1's goal was to prove the speedup is "not a sync/clock/session artifact." It is **not** an artifact — but
the honest magnitude is **1.61×, not 1.89×**:

- The **synced** speedup is a stable **1.61×** across 3 sessions (415 → 258 ms/512). This is real GPU
  throughput: the graph route genuinely runs the prefill forward faster.
- The **nosync** speedup (v2-only `realize()` loop) is **1.0×** — it *hides* the win, because a nosync loop
  measures host dispatch, not GPU time, and both routes dispatch at the same host rate.
- The prior promotion's **1.89×** came from `qk_prefill_v2_measure` (baseline-first nosync) — a different
  measurement pattern. The synced arbiter is the correct gate, and it confirms a material, stable win.

## Why it's a big win (recovers the in-model matmul penalty)

Baseline `PREFILL_V2` runs the prefill forward at **~415 ms / ~1236 tok/s** (synced) — matching the
independently-measured true prefill (the in-model tinygrad WMMA gate/up runs at ~22 TFLOPS, ~2.7× below its
isolated speed). The graph route (`build_gemm_lds2`) at **~258 ms / ~1983 tok/s** recovers most of that
in-model penalty: our dependency-free kernel, wired as a single fused `custom_kernel`, does not suffer the same
in-graph slowdown. Prefill moves from **~40% → ~66% of llama** (pp512 3020). The 1.61× exceeds the
Amdahl-from-isolated-ratio estimate precisely because the lever is the *in-model penalty*, not the isolated
kernel ratio.

Gate 1 is satisfied for default-on readiness.
