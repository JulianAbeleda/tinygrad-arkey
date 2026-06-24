# DIAGNOSTIC — why the isolated Tensile 66 doesn't transfer in-model: the route's LAYOUT-TRANSPOSE tax cancels it

Localized the prefill Tensile 0.999× (`prefill-tensile-land-result-20260619.md`) to its cause via clean
per-kernel GPU times (warmed eager DEBUG=2 `tm`, + isolated transpose timing). `extra/qk_tensile_diag.py`.

## Measured [M]
- **`tensile_gateup` runs at 810–850µs/ea in-model** = ~63 TFLOPS on the 51.5-GFLOP 512×12288←4096 matmul ≈ its
  isolated 66 TFLOPS. **→ the Tensile kernel is NOT slow in-model** (candidate "in-model regime kills Tensile" REFUTED).
- **The `route_pf16` layout overhead per routed gateup = ~231µs** (isolated, DEBUG=2):
  - output transpose `out^T` 512×12288 fp16 = **172µs** (dominant), input transpose `x^T` 4096×512 = 31µs,
    `zeros(12288,512)` = 28µs.
- So per routed gateup, ON = Tensile **810** + transpose **231** = **~1041µs**.
- (Caveat: eager-OFF gateup measured 2578µs/20 TFLOPS, but that's the UNTUNED path — warmstart-TC does NOT apply in
  eager (eager-OFF total 367ms ≈ 2× the JIT-OFF ~205ms). JIT-OFF per-kernel time is uncapturable (replay emits no
  ProfileRangeEvent/PMC). Since JIT e2e is 0.999×, the warmstart-TC fp16-WMMA gateup ≈ 1041µs (~50 TFLOPS) — i.e.,
  tinygrad's in-model warmstart-WMMA is BETTER than the banked 41-TFLOPS figure suggested.)

## Conclusion — the lever is transpose-free integration (NOT dead, NOT the kernel)
The Tensile matmul saves ~230µs/gateup vs warmstart-WMMA, but the route's **layout transposes add ~231µs/gateup**
(the 512×12288 output transpose alone is 172µs) → **net ≈ zero (0.999×)**. The Tensile kernel is genuinely fast
in-model; the integration layout-conversion eats the entire win. This is a textbook instance of the campaign's
meta-pattern: **the kernel wins in isolation, in-model integration loses it.**

**The fix (a real build, not a kernel swap):** eliminate the layout transposes around the Tensile call —
- the dominant cost is the OUTPUT transpose (`C[out,T]` → `[T,out]`, 172µs). Avoid it by having the consumer read
  `C` in `[out,T]` layout directly (propagate the transpose into the next op / RMSNorm·matmul fusion), or extract
  a Tensile kernel whose output layout matches tinygrad's `[T,out]` (a different Tensile solution / transpose-fused);
- the input `x^T` (31µs) can be fused into the activation's producing kernel.
- If the ~231µs/gateup tax is removed: ON ≈ 810µs vs WMMA ~1041µs → ~1.28× on the routed matmuls → e2e ~1.15–1.2×
  (Amdahl, ~74% routed). That would make Tensile a real prefill win.

## Status / next
- Tensile route stays `PREFILL_TENSILE_GEMM=0` research-only; **as-built no win, but the cause is now precise and
  the fix is scoped** (transpose-free integration). This is the prefill instance of the universal in-model-integration
  lever (`inference-perf-measured-map-20260619.md`).
- Measurement caveat banked: tinygrad HCQ per-kernel time is only capturable EAGER (replay emits nothing), and eager
  bypasses warmstart-TC — so JIT-tuned per-kernel times are not directly measurable; use isolated-kernel `tm` +
  the clock-controlled e2e A/B together.

## Files
`extra/qk_tensile_diag.py`, `extra/qk_tensile_ab_measure.py`, `extra/qk_tensile_inmodel.py` (`route_pf16`).
Verdict: `prefill-tensile-land-result-20260619.md`. Map: `inference-perf-measured-map-20260619.md`.
