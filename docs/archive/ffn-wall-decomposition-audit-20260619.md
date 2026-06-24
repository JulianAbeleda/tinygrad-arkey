# AUDIT — the "matmul ~24% of wall" decomposition is NOT cleanly measurable; the QUALITATIVE claim IS reliable

Audited the D1 claim ("matmul ~24% of the FFN wall, ~76% glue+host-dispatch"). The specific split does not survive;
the qualitative conclusion does.

## What does NOT hold (RETRACT the specific numbers)
- My "matmul = 24%" was an UNSUPPORTED ASSUMPTION (assumed the in-jit matmul runs at 42 TF). Internal check: the
  Tensile A/B Amdahl (1.56x matmul -> 1.005x e2e) implies ~1.4%, not 24% -- a 17x self-contradiction I glossed over.
- A clean per-component decomposition is BLOCKED on this stack: timing sub-pieces OVERSTATES them (one-matmul jit =
  2.72ms for a ~1.2ms-GPU matmul -> a tiny jit carries ~1.5ms fixed overhead that doesn't exist amortized in the
  full FFN); `.sum()`/elementwise wrappers break the warmstart TC match (matmul falls to 0.3 TF non-TC path -> 172ms
  garbage); graph timestamps are unreliable. So matmul/glue/dispatch %s are NOT cleanly separable. Retract 24%/76%.

## What IS reliable (the qualitative claim holds)
1. **Tensile (1.56x kernel, shape-matrix-verified) -> ~1.00x e2e**, reproduced: full forward 0.997x, FFN-only
   0.993x. A real 1.56x matmul moving the wall <1% => **the matmul is NOT the e2e wall bottleneck.** SOLID.
2. **Per-kernel/dispatch overhead is large**: one-matmul jit = 2.72ms vs ~1.2ms GPU-matmul -> ~1.5ms overhead/kernel
   (consistent with the campaign's ~1.3ms/kernel). Tensile speeds the GPU-compute (1.2->0.77ms) but NOT the
   overhead -> effective per-matmul ~1.19x, diluted to ~1.0x across the FFN+glue. So the wall is dominated by
   per-kernel overhead (host-dispatch + glue), which a faster matmul cannot touch.
3. Full FFN jit ~6.6-7.4ms (clock-varying), ~10-20 TF effective for 154 GFLOP -- far below the matmul's own ~42 TF,
   confirming the wall is NOT matmul-GPU-bound (whatever the exact split).

## Net (honest)
- TRUE + reliably measured: a faster matmul (even Tensile 66) gives ~1.0x e2e -> the matmul is not the wall
  bottleneck; the e2e lever is per-kernel overhead / kernel count (= concrete-KV 1.24x). This is what the user
  needs and it is SOLID (the Tensile A/B is the trustworthy instrument).
- NOT measurable: the exact matmul% vs glue% vs dispatch% of the wall (blocked by non-decomposable per-jit/kernel
  overhead + codegen-breaking probes + unreliable graph timestamps). My specific "24%/76%" is RETRACTED.
- Honest pattern this session: the QUALITATIVE conclusions (verified by clean e2e A/Bs) held; every attempt to
  attach a PRECISE per-component number was either unsupported or refuted by the measurement-stack limits.

## Files
/tmp/audit_ffn.py, /tmp/audit_ffn2.py (decomposition harnesses -- both hit the non-decomposable-overhead wall).
The reliable instrument is the e2e Tensile A/B (qk_tensile_ab_measure.py). Refines why-tensile-doesnt-transfer-ANSWERED.
