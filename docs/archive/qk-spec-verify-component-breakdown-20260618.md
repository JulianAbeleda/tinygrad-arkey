# Arc A Phase 1.5 — spec-verify component breakdown: VERDICT → **distributed T-scaling; no single kernel wins; spec-verify route CLOSED**

Diagnostic accounting of where the T=K+1 verify GPU time goes, after Phase 1 showed the isolated Q4_K ffn GEMM is
already 2.58× one pass (not the bottleneck). `extra/qk_verify_component_breakdown.py`,
`bench/qk-spec-verify-component-breakdown/result.json`. ctx512, Qwen3-8B-Q4_K_M, decode_enabled=True, SDPA. **No
routes/defaults.** Per-kernel shares are EAGER (DEBUG=2 tm — the only per-kernel method; eager unbatches/inflates,
so shares are directional) anchored to the **real JIT W==D verify total** (authoritative).

## Real JIT verify totals (authoritative)

| T (=K+1) | JIT verify | × one T==1 pass |
|---|---:|---:|
| 1 | 12.67ms | 1.00 |
| 3 | 51.5ms | **4.07×** |
| 5 | 59.0ms | **4.66×** |
| 9 | 115.9ms | **9.15×** |

**The T=1→T=3 jump is 4× for just 2 more tokens.** That is the verify **falling off every T==1-only fast path** at
once: the shipped decode wins are T==1-gated — coop GEMV (Q4_K attn_q/o, Q6_K lm_head/ffn_down) and flash-decode
all revert to the slower **batched GEMM + SDPA** at T>1. So even T=3 is already 4× one pass before any per-token
scaling.

## Component scaling (eager ms by T — directional)

| component | T=1 | T=3 | T=5 | T=9 | scales with T? |
|---|---:|---:|---:|---:|:--:|
| q4k_gemm | 10.0 | 22.1 | 29.3 | 44.3 | yes (~linear) |
| attention + reduce_other¹ | 14.7 | 52.3 | 45.0 | 109.6 | yes (steep, T×KV) |
| q6k_gemm | 2.6 | 7.1 | 15.4 | 20.5 | yes |
| elementwise_norm | 1.4 | 2.6 | 2.9 | 2.8 | **no (flat)** |

¹ attention is split across the KV-named "attention" bucket and the per-layer big reduce mis-bucketed as
"reduce_other" (its KV dim is factored, e.g. `r_2_2_4_2_16_3_4_16_4_2_32`, 35 launches = one/layer). Combined they
are the steepest-scaling component. **Only `elementwise_norm` (norm/RoPE/residual/SwiGLU) is T-independent.**

## Amdahl ranking @T=5 (share × real 59ms; decision rule = a component ≥30% AND ≥2×-reducible)

| component | share | 2×→whole | T-indep→whole |
|---|---:|---:|---:|
| q4k_gemm | 31.6% | 1.19× | 1.26× |
| attention+reduces | ~48% | ~1.3× | ~1.5× |
| q6k_gemm | 16.6% | 1.09× | 1.16× |
| elementwise_norm | 3.2% | 1.02× | 1.02× |

**No single component clears "≥30% AND made T-independent → verify ≤1.5× one pass."** The biggest lever (attention
made fully T-independent) gets the whole verify to only ~1.5× *of its current self* → 59→~40ms = ~3.1× one pass,
still ≫ the 1.5×-one-pass spec gate. Q4_K-GEMM reuse alone (Phase 1's target) → 1.19–1.26× → ~2.6× one pass. The
components are co-dominant and all T-scaling; **no pair shares one primitive** (attention ≠ GEMM ≠ Q6_K).

## Verdict — which sentence is true

> **"Spec verify is distributed across attention + Q4_K GEMM + Q6_K GEMM, all T-scaling; it is not worth a single
> kernel."**

To take verify from **4.66× → ≤1.5× one pass** (a ~3× cut) requires making the **whole batched forward T-cheap** at
once: T>1 fast-path versions of (a) coop GEMV / weight-reuse GEMM for Q4_K **and** Q6_K, (b) a flash/decode-style
**batched attention for T queries over long KV** (the prefill-flash arc — refuted on perf / LDS-walled), and (c)
recovering the T==1 fast-path gating. That is the **prefill-class batched-forward problem across every component**,
not one primitive. **Spec-verify route CLOSED at the kernel level** — it does not bottleneck on a single buildable
primitive; it bottlenecks on the entire forward losing its T==1 specializations.

## Consequence for Arc A (the weight-reuse primitive)

- **Spec-decode justification for the Q4_K weight-reuse GEMM is withdrawn.** It addresses only ~31% of verify at a
  partial gain (Q4_K is already 2.58× via UPCAST) → ~1.2× whole-verify → spec still loses by a wide margin.
- **The primitive's surviving justification is PREFILL alone** (T≫K), where: the weight-reuse fraction is far larger
  (one weight read serves hundreds of columns, not 5), attention is handled by the separate flash-prefill arc, and
  PREFILL_V2 already owns the FFN. There the Boehm-style LDS/register tiling is the known lever (tinygrad matmul
  LDS=0 → ~27% peak). Pursue the primitive **as part of the prefill arc**, gated on prefill pp-throughput, not spec.
- **Spec decode stays banked proven-correct-but-not-fast** ([[qk-spec-decode-gate]]): correctness/on-device-accept
  proven; speed blocked not by one primitive but by the whole-forward T==1-specialization loss. Reopen only if a
  broad batched-forward (decode-speed-at-T>1) arc is funded — a much larger program than a single GEMM kernel.

## Files
`extra/qk_verify_component_breakdown.py` (profiler), `bench/qk-spec-verify-component-breakdown/result.json`
(gitignored; JIT totals + eager component us). Method caveats: eager per-kernel shares are unbatch-inflated
(directional); JIT W==D totals are authoritative; attention is split across two buckets (factored KV dim). No
kernel/model/default changes.
