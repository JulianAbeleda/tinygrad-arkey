# SCOPE+DO — transpose-free Tensile FFN (recover the prefill win the diagnostic localized)

## Why (measured)
`prefill-tensile-diag-result-20260619.md`: Tensile runs at ~63 TFLOPS in-model (810µs/gateup, fast) but the
per-linear `route_pf16` adds ~231µs/gateup of layout transposes — and the FFN chains 3 routed linears, so the
gate/up OUTPUT transposes + the down INPUT transpose (all ~172µs) materialize redundantly even though they cancel
mathematically. Measured FFN transpose tax ≈ 635µs/layer × 36 = ~23ms of the ~205ms forward. Removing it →
estimated e2e ~1.1–1.2× (Tensile's matmul win stops being cancelled).

## The fix — keep the FFN in [feature, T] (column) layout
The Tensile kernel computes C[out,T] = W[out,in] @ A[in,T]. The FFN cancels if everything stays column:
```
xT = xᵀ              [D,T]   (ONE entry transpose+contig)
g  = Tensile(Wg, xT) [H,T]   (no output transpose)
u  = Tensile(Wu, xT) [H,T]
h  = silu(g)·u       [H,T]   (elementwise, contig once = down's A)   <- gate/up out-transpose ELIMINATED
o  = Tensile(Wd, h)  [D,T]   (h is already [in,T] -> down in-transpose ELIMINATED)
return oᵀ            [T,D]   (ONE exit transpose)
```
Transposes: 635µs/layer → ~116µs/layer (entry xT 31 + h-contig 28 + exit oᵀ 57). Math: C = W@xᵀ = (x@Wᵀ)ᵀ = linearᵀ,
so column chain ≡ the row FFN exactly (verify rel_err + dNLL).

## DO
1. `extra/qk_tensile_inmodel.py`: add `route_pf16_col(lin, x_col)` — takes A=[in,T] contiguous, returns C=[out,T]
   (no transposes; reuse `trivial_fxn`/TensileRunner). Eligibility = same ELIGIBLE map + `_installed`.
2. `tinygrad/llm/model.py` `_feed_forward` (prefill_v2 dense branch): when `PREFILL_TENSILE_GEMM` and gate+up+down
   all eligible+installed → the column chain above; else fall through to the existing per-linear `.contiguous()`
   path (silent fallback, decode untouched, MoE/fused untouched).
3. Measure: clean clock-controlled interleaved A/B (`extra/qk_tensile_ab_measure.py`) OFF vs ON.
4. Quality: dNLL ≤ 0.01 (`extra/qk_prefill_v2_nll_eval.py`) with the column FFN.

## Gates
- correctness: column-FFN rel_err ≤ 2e-2 vs row FFN (the transpose identity); dNLL ≤ 0.01.
- speed: clean interleaved A/B ≥ 1.10× (the diagnostic's floor); report honest number.
- fallback: flag-off byte-identical; ineligible (k/v, MoE, attn) untouched; decode W==D untouched.
- only attn-q/o stays per-linear (its qo route still has transposes; FFN is the 3-chain win — do FFN first).

## Risk / honest ceiling
Only the FFN gate/up/down chain (3 of the ~74% routed matmuls) gets transpose-free; attn q/o keeps its transposes.
FFN is ~55% of prefill GPU time, so the FFN transpose removal is the bulk of the available win. If the clean A/B
still shows <1.10×, the transpose tax wasn't the whole story (re-open the diagnostic). Stays research-only
(`PREFILL_TENSILE_GEMM=0`); the deps/artifact (TPE-0) policy is still the separate landing gate.
