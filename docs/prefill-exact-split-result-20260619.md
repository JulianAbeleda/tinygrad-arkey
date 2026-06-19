# RESULT — the EXACT prefill split (tuned __call__ capture): two real ~1.2× levers; "near-ceiling" was WRONG

Got the tuned per-kernel split by profiling `model.__call__`'s prefill_v2_jit **capture run** (first call —
kernels execute individually WITH the warmstart try/finally → tuned; replay emits nothing). Cold clock, so
absolute ms are inflated, but the **% split and same-clock relative GFLOPS are valid**.

## The split (tuned, % of forward)
| component | % | GFLOPS | |
|---|---:|---:|---|
| **gate/up matmul** `r_16_192_32_2_2_2_2_4_32_2_8` | **41.6%** | **19647** | out>in (4096→12288); UPCAST(0,2) |
| **attention** (symbolic `(start_pos+512)` kernels, sum) | **~25%** | 106–4415 | **NON-TC** (symbolic KV blocks TC) |
| down matmul `r_8_64_..._96_2_8` | 11.6% | **32147** | in>out (12288→4096); UPCAST(0,4) |
| up-variant `r_16_64...` (+n1) | 11.0% | 23–26k | |
| lm_head + elementwise glue | ~4% | | |

## Why the TFLOPS don't convert — TWO measured causes (both ~1.2×, both bigger than the earlier "1.05×/near-ceiling")
1. **Attention ~25% of the forward, ALL non-TC.** symbolic `KV=start_pos+512` blocks the concrete-shape tensor-core
   lever (the same one firing for FFN matmuls). Kernels run at 106–4415 GFLOPS (vs 19–32k for the matmuls).
   **Fix = concrete-KV + TC attention** (probed 2.56× standalone). 25% × (1−1/2.56) ≈ **15% of forward → ~1.18× e2e.**
2. **gate/up matmul (42% of forward) at ~0.6× the down matmul's efficiency** (19647 vs 32147 GFLOPS, same clock).
   The out>in gate/up shape gets `UPCAST(0,2)`; down (in>out) gets `UPCAST(0,4)` and is 1.64× more efficient.
   **Fix = a better warmstart schedule for the gate/up shape** (schedule search, NOT walled). Lifting gate/up to
   down's efficiency on 42% of the forward → **~1.2× e2e.**

## Correction to the record
The earlier "prefill near-ceiling, attention only ~6%, lever ~1.05×" (`prefill-nonmatmul-missing-primitive-result`)
was WRONG — it used the EAGER profile, which produces different/untuned kernels and undercounted the symbolic-KV
attention. The tuned __call__ capture shows attention is ~25% and gate/up is inefficient. **Prefill has ~1.4–1.5×
of stacked headroom** (attention TC ×1.18 · gate/up schedule ×1.2), enough to match/pass llama (currently 82%).

## Levers (ranked, both un-walled)
- **L1 — concrete-KV TC attention (~1.18×):** capture prefill with concrete start_pos (KV concrete → TC fires) +
  re-wire the probed Option-B TC attention (`extra/qk_prefill_tc_wr_softmax_probe.py`, 2.56× standalone). Salvage
  path 1 from `amd-prefill-tc-attention-probe-20260617.md` ("blocked by jit arg plumbing" — the un-done work).
- **L2 — gate/up schedule (~1.2×):** search a better warmstart opt for the gate/up (4096→12288) shape so it hits
  down's ~32k GFLOPS instead of 19.6k. Extend `_prefill_v2_opts` / the warmstart gate (`extra/qk_prefill_gate.py`).

## Caveat
Cold-clock capture → absolute TFLOPS reconciliation vs the Tensile A/B (0.999×) is bounded (warmstart-application
+ clock ambiguity); the % split and relative GFLOPS are the trustworthy signals. JIT replay per-kernel still
uncapturable; the capture run is the workaround (tuned, individual kernels).

## Files
`/tmp/tuned_cap.txt` (tuned __call__ capture DEBUG=2). Prior (corrected): `prefill-nonmatmul-missing-primitive-result-20260619.md`,
`prefill-tensile-transpose-free-result-20260619.md`. Probe: `amd-prefill-tc-attention-probe-20260617.md`.
