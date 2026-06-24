# RESULT — Tensile prefill route: clean clock-controlled A/B = 0.999× (NO in-model speedup); prior 1.27× was a clock-confound

Executed P1 of `prefill-tensile-land-scope-20260619.md`: a clean, clock-controlled A/B of PREFILL_V2 fp16-WMMA
(Tensile OFF) vs +Tensile (ON, all 3 roles), measured INTERLEAVED in one process (`extra/qk_tensile_ab_measure.py`).

## Result [M] (Qwen3-8B-Q4_K_M, gfx1100, T=512, warm, interleaved, reproduced 2×)
| | OFF (fp16-WMMA) | ON (+Tensile qo+gateup+down) |
|---|---:|---:|
| median pp512 | **2501 tok/s** | **2497 tok/s** |
| best | 2514 | 2511 |
| route | `{}` (pure fp16-WMMA) | `{qo:72, gateup:72, down:36}` |
| **speedup** | — | **0.999× (run1) / 0.998× (run2)** |

**Routing is correct and active (qo+gateup+down all route to Tensile), quality still accepts (dNLL prior), but
there is NO in-model speedup.** The strong gate (1.35×) and even the research gate (1.25×) are NOT met under clock
control.

## The prior 1.27× was a clock-confound
`prefill-tensile-inmodel-measurement-result-20260619.md` reported 1.27× (OFF 2709 → ON 3433), measured in
SEPARATE non-interleaved warm runs. Sustained load ramps the GPU clock, so the later (ON) run ran at a higher
clock — the classic documented trap (`amd-decode-measurement-confounds`). Interleaving OFF/ON round-robin
eliminates it → 0.999×. (Single-process variance was huge: the same ON config measured 3433→7321 tok/s across
processes purely from clock.) **The interleaved 0.999× is the trustworthy number.**

## WHY (not fully pinned, but the key fact) — the isolated 66 TFLOPS does NOT transfer in-model
Tensile is genuinely fast ISOLATED (shape-matrix: gateup 61, down 71, qo 77 TFLOPS) — far above tinygrad's
in-model fp16-WMMA (~41). Naive Amdahl (routed matmuls ~74% of prefill, 1.6× on them) predicts ~1.37× e2e. We get
1.00×. So the isolated Tensile win evaporates in-model. Candidate causes (next diagnostic; DEBUG=2 per-kernel was
polluted by warmstart-search noise): (a) the route's layout transposes/zeros (`route_pf16`: x→xᵀ.contiguous,
zeros(out,T), outᵀ.reshape) — but these are ~30µs vs ~780µs/matmul, too small alone; (b) the forced Tensile grid
/ transposed-contiguous inputs make the in-model Tensile kernel run far below its isolated 66; (c) the OFF
fp16-WMMA is already faster in-model than the 41 POWN figure. Most likely (b)+(c): Tensile-in-model ≈ WMMA-in-model.

## The meta-pattern — isolated wins don't transfer in-model (decode AND prefill)
This MIRRORS the decode finding exactly:
| regime | isolated kernel | in-model | transfer |
|---|---|---|---|
| decode | tinygrad GEMV **76%** peak (beats llama 57%) | **44%** | loses 32 pts |
| prefill | Tensile **66 TFLOPS** (beats llama 49, tinygrad 41) | **≈41 (1.00×)** | win evaporates |

**Both regimes: tinygrad has competitive/winning kernels in isolation, but the advantage is lost in-model.** The
universal bottleneck is **in-model integration/execution**, not the kernels. For decode it's fused-mmvq integration
(activation-quant amortization + sustained occupancy); for prefill it's whatever makes the routed Tensile kernel
run at WMMA-speed in-model (layout/grid/transpose-fusion). Same lesson, both halves.

## Verdict
- **Tensile prefill route as-built = NO win (0.999×, clock-controlled).** Not eligible to land; the 1.27× claim is
  retracted (clock-confound). Stays `PREFILL_TENSILE_GEMM=0` research-only.
- **Frontier #2 reframed:** the lever is NOT "route to Tensile" (the kernel's isolated speed doesn't transfer) —
  it's the SAME in-model integration problem as decode. Next diagnostic (bounded): clean per-kernel time of
  `tensile_gateup` vs the fp16-WMMA gateup it replaces (via ProfileRangeEvent on a warm replay, not DEBUG=2 during
  warmstart search) to localize (b) vs (c), and test a transpose-free route variant.
- Prefill rest state: PREFILL_V2 fp16-WMMA (~41 TFLOPS, 82% of llama) — unchanged, shipped, decode untouched.

## Files
`extra/qk_tensile_ab_measure.py` (clean interleaved A/B), `extra/qk_tensile_inmodel.py` (`route_pf16`). Scope:
`prefill-tensile-land-scope-20260619.md`. Prior (now-corrected) claim:
`prefill-tensile-inmodel-measurement-result-20260619.md`. Learning atlas:
`decode-bandwidth-bound-pmu-learning-20260619.md`.
