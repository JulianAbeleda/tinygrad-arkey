# Prefill roofline from first principles (2026-07-24)

Derived on 8B; the decay/floor arithmetic applies to both models and 14B rows were added after it went live.

Written after seven theories were tested in one session (six refuted, one worth +7.3%). Its purpose is
to stop the next arc from re-deriving the frame, and to record the *measurement* errors that made most of
those theories look attractive in the first place.

## 1. The achievable peak is ~105 TFLOPS, not 122.8 and not 61.4

Measured, not assumed: `extra/llm_research/microbench/wmma_peak.cpp` — pure back-to-back WMMA, 8 independent
accumulators, zero `global_load`, zero `ds_` in the loop (verified in the `.s`).

    waves=16384 iters=20000 nacc=8  ->  105.5 / 104.6 TFLOPS

86% of the 122.8 spec figure; **171% of 61.4**. So WMMA does reach dual-issue-class rates. Use **105 TF**
as the denominator for every efficiency claim on this device. Quoting 122.8 understates efficiency by 17%;
quoting 61.4 flatters it by 1.7x and produced a false "we are at 94% of peak, nothing left" reading.

## 2. Where the model actually sits

Per-chunk GEMM FLOPs must come from the model config, NOT from `2*P*T` and NOT from BoltBeam's role
shapes:

| source | per 512-tok chunk | note |
|---|---|---|
| `2*P*T` | 8.19 TFLOP | WRONG, +15%: counts embed + lm_head, which are not per-token matmuls |
| `QWEN3_8B_ROLE_SHAPES` | 4.48 TFLOP | WRONG, −37%: the 4 promoted roles cover only 63% of in-layer params |
| **config-derived** | **7.11 TFLOP** | q,k,v,o + gate,up,down over 36 layers = 6.95B in-layer params |

(Check: 6.95B in-layer + 0.62B embed + 0.62B lm_head = 8.19B, matches the model.)

| | achieved | of 105 TF | headroom | share of kernel time |
|---|---|---|---|---|
| **GEMM** | 50.7 TF | **48.3%** | **2.07x** | **79.6%** |
| attention | 7.7 TF | 7.3% | 13.7x | 6.1% (pp512) → 23% (pp4096) |

## 3. The efficiency chain, and what each factor means

    achieved/peak = (WMMA share of SQ-busy cycles) x (SQ-busy share of wall)
                            "mix"                          "duty"

`occupancy_pct` in the PMC output **is** the duty cycle: mean SQ busy / `GRBM_GUI_ACTIVE` = 466,583 /
1,281,775 = 36.4%, exactly the reported value.

For attention (single dispatch, so the counters are attributable): `0.375 x 0.364 = 13.6%`, and the
measured 12.5%-of-61.4 reading closes the chain. After the THEORY-6 fix mix rises to 46.5% and duty is
untouched, giving ~16.9%.

**For GEMM the chain does NOT close and that is a finding.** At duty 0.40, `0.483/0.40 = 1.21 > 1` is
impossible, so GEMM's real duty is >= 0.48. Cause: the GEMM rows MERGE 72 launches while
`GRBM_GUI_ACTIVE` is per-dispatch, so that ratio was never a duty cycle for a merged row — the same
aggregation trap as the `counters_by_chunk` bug fixed in `654c9b2ce`.
**MEASURED 2026-07-25 -- and it does not rescue the chain.** Per-dispatch counters now exist
(`bench/prefill-pmc-per-dispatch/latest.json`, `--hw-trace --context 512 --chunk 512` under sudo;
`raw_GRBM_GUI_ACTIVE_count == 1` proves the row carries exactly ONE dispatch). Duty =
mean(`SQ_BUSY_CYCLES`)/`GRBM_GUI_ACTIVE`, which reproduces `occupancy_pct` exactly:

| kernel | calls | share of kernel wall | **duty** | VALU% | L2 hit% |
|---|---:|---:|---:|---:|---:|
| `E_4_96_32_...` | 72 | 39.3% | **42.4%** | 2.03 | 31.7 |
| `E_4_32_32_...` | 36 | 22.1% | **41.5%** | 1.75 | 54.4 |
| `E_4_32_32_...` | 72 | 14.9% | **36.3%** | 1.60 | 40.9 |
| `r_1187_32_...` | 1 | 8.7% | **46.1%** | 0.55 | 11.6 |
| `E_4_8_32_...` | 72 | 4.9% | **28.8%** | 0.92 | 77.4 |
| `amd_gfx1100_q16_grid_hd128...` (attention) | 36 | 4.1% | **33.8%** | 6.41 | 63.3 |

Wall-weighted mean duty = **38.6%**. Rows with VALU% < 5 -- the WMMA-class kernels -- are **95.9%** of kernel
wall. (Graph-GEMM kernels are named `E_*`, so `_classify_kernel` files them as elementwise; identify them by
low VALU% + high `SQC_LDS_IDX_ACTIVE`, not by the `kind` field.)

**The chain still does not close, and that is the finding.** `achieved/peak = mix x duty` with the claimed
GEMM `achieved/peak = 0.483` and measured duty 0.424 requires `mix = 1.14 > 1`. Since mix cannot exceed 1,
**the 48.3%-of-peak GEMM figure in the table above is overstated** -- and the "+18.4% / +46.1%" prizes in
Section 6 are derived from it. Treat them as unproven until the achieved figure is re-derived; the duty
number is the direct measurement and the 48.3% was inferred from config FLOPs over an assumed 79.6% runtime
share.

Caveat, stated so nobody over-reads it: counters require `PROFILE=1`, and the profiled run measures
1536 tok/s against 3700 unprofiled, so per-dispatch `GRBM_GUI_ACTIVE` may include profiling-induced idle.
That makes 38.6% a lower bound on true duty. It does not restore the 48.3% claim -- a *higher* true duty
implies a *lower* mix, moving the deficit further into mix rather than resolving it.

### CORRECTED same day: subtract the instrumentation floor first

Every dispatch in the trace carries a **fixed 232,182 idle cycles regardless of size** -- the minimum idle
observed across near-empty dispatches. At 2.3 GHz that is **~101 us**, where real AMD dispatch launch is
2-5 us. It is the profiler, not the hardware. It swamps small dispatches (they read 0-15% duty, a meaningless
number) and accounts for ~6% of the large ones.

Correcting duty as `busy / (GRBM_GUI_ACTIVE - 232182)`:

| dispatch | cycles | raw duty | **corrected** | scaling idle |
|---|---:|---:|---:|---:|
| `r_1187_32_4_16_2_2_2_4` | 52.6M | 46.1% | **46.3%** | 53.7% |
| `E_4_32_32_...` | 3.9M | 41.5% | **44.1%** | 55.9% |
| `E_4_96_32_...` | 3.5M | 42.4% | **45.4%** | 54.6% |
| `E_4_32_32_...` | 1.5M | 36.3% | **43.0%** | 57.0% |
| attention `amd_gfx1100_q16_grid_hd128` | 0.9M | 33.8% | **45.2%** | 54.8% |

**Corrected duty over the four large dispatches = 46.1%, implying mix = 0.483/0.461 = 1.05.** That is within
measurement error of the 1.00 ceiling, so **the retraction above is too strong** -- the 48.3% achieved figure
is borderline, not refuted. Retained as "unconfirmed", not "wrong".

**The finding that survives, and it is sharper than the original:** corrected duty converges on **43-46%
across a 57x range of dispatch sizes**, for GEMM and attention alike. Size-independence is the evidence that
this is a real in-kernel stall rather than launch/drain overhead (which would shrink with size) or
instrumentation (removed above). And **mix ~= 1.0 means the instruction composition is already near-optimal**:
there is no "issue better work" lever left. The whole remaining deficit is the machine stalling ~55% of the
time inside the kernel. Every lever pursued in the 07-24 session targeted mix.

Next measurement, before any fix: re-run with `SQ_WAIT_ANY`, `SQ_WAIT_INST_LDS`, `SQ_BUSY_CU_CYCLES` to split
that 55% into memory latency / LDS / barrier / insufficient waves -- four different bugs with four opposite
fixes. Also worth one unprofiled cross-check, since all of the above comes from a single `PROFILE=1` run.

## 4. Instruction counts are the wrong currency

WMMA is **1.7% of instructions but 37.5% of cycles**: 16 cycles each, against **0.457** for the average
non-WMMA instruction (dual-issued VALU is sub-1). Every projection made in instruction-share overshot ~2x;
the one priced in cycles (THEORY 3) landed on its measurement.

| theory | instrs | cycles | predicted kernel | predicted whole-model |
|---|---:|---:|---:|---:|
| T6 reductions | 328 | 150 | 1.28x | +5.0% |
| T4 sync/sched | 213 | 97 | 1.17x | +3.3% |
| V-gather loads | 112 | 51 | 1.08x | +1.7% |
| T3 causal skip | 113 | 52 | 1.08x | +1.7% (**measured +1.66%**) |

## 5. Zero decay is impossible; the floor is −6.14%

    tok/s = S/T = rate / (2P + K*S)      K = 2*2*Hq*Hd*L/2

Tokens grow O(S); causal attention grows O(S^2). Attention goes from 0.9% of FLOPs at pp512 to 7.0% at
pp4096, which alone costs **−6.14%** even if attention were exactly as efficient as GEMM.

| | decay pp512→pp4096 | excess over floor | |
|---|---|---|---|
| ours 8B, before T6 | −16.76% | −10.6pp | 2.7x ideal |
| **ours 8B, after T6** | **−12.48%** | −6.3pp | 2.0x ideal |
| llama 8B (SAME-SESSION) | **−5.62%** | +0.5pp | ~0.9x ideal |
| **ours 14B** | **−8.24%** | −2.1pp | 1.3x ideal |
| llama 14B (same-session) | −10.99% | −4.9pp | 1.8x ideal |

**CORRECTED 2026-07-24 late.** The row previously read "llama −11.51%, gap closed to 0.57pp, context-scaling
parity". That llama figure was CROSS-SESSION. Re-measured same-session, llama 8B decays only −5.62% (its
pp512 is 3347 tonight, not 3571) so **we are at ~2.2x llama's 8B decay, NOT at parity**. The reverse holds on
14B, where our −8.24% genuinely beats llama's −10.99% and sits closest to the floor of anything measured.
Drift is SHORT-context: llama 8B pp4096 reproduced to −0.1% across sessions, pp512 drifted −6.3%.

**Decay is still the more robust comparison** (a within-run ratio), but it is NOT drift-immune the way this
doc originally claimed: drift is concentrated at short context, and decay is computed FROM the pp512
endpoint, so a drifting pp512 moves the decay figure directly. That is exactly how the "0.57pp / parity"
error arose.

Same-session llama has now been measured, so the absolute margins ARE valid: **8B +3.3% @pp4096,
14B +5.6% @pp512 and +8.8% @pp4096** (8B @pp512 is +11.4% but llama's own stdev there is 7%, so treat it as
soft). See `docs/prefill-current-state.md` for the authoritative table.

**All remaining attention headroom is +6.7%**, capped by this floor. Going below it needs sub-quadratic
attention (windowing/sparsity) = a model change, not a kernel change.

## 6. Prize ranking

| lever | whole-model | status |
|---|---|---|
| **the ~55% in-kernel stall (duty 46%)** | unquantified | **the live lever** — measured 2026-07-25, uniform across a 57x size range (§3) |
| GEMM 48% → 60% of peak | +18.4% | **UNCONFIRMED** — mix comes out at 1.05, borderline; not refuted (§3) |
| GEMM 48% → 80% of peak | +46.1% | **UNCONFIRMED**, same reason |
| better instruction mix, any form | ~0 | **CLOSED** — mix ≈ 1.0, nothing left to win here |
| all remaining attention | +6.7% | capped by the decay floor |
| T3 causal skip | +1.66% | measured, default-OFF, gated on 14B evidence |
| LDS bank conflicts | +0.13% | shipped (`114277f36`), conflict now bit-exact 0% |

GEMM is 79.6% of runtime at 48% of peak and has never been attacked. Attention was pursued all session
because the vs-llama deficit lived there; that deficit is now closed and the deficit was never where the
throughput was.

## 7. The measurement errors that made bad theories look good

Recorded because each one cost a full investigation:

1. **Treating one measured point as a slope.** The stride-96 LDS arm (12.5%→50% conflict, −5.5%) implied
   0.15%/pp and a +1.8% prize; eliminating the conflict entirely delivered **+0.13%**, a ~14x overshoot.
   Sensitivity was a *threshold*, not a slope: 896→1024 LDS cycles costs 0.009%/%, 1024→1792 costs
   0.073%/% — 8x steeper. The baseline was below the cliff and the 12.5% was fully hidden.
2. **Instruction-count deltas are not cycle deltas** (§4), and cycles/instruction differs by class:
   `global_load_d16` 3.02 cyc vs `b128` 6.61 cyc.
3. **Pricing the work removed but not the machinery that removes it.** V-transpose: a `[kv][hd]→[hd][kv]`
   copy is a strided scatter, ~7% of whole-model, not the ~3% a bytes/HBM-peak estimate gives.
4. **Aggregated counter rows.** Merged rows carry one dispatch's counters; a `--context 4096` run returned
   byte-identical counters to `--context 512` until keyed per chunk.
5. **Comparing against a recorded baseline.** This box drifts ~5% across a session vs 0.59% back-to-back.
   Paired, interleaved, same-session A/B repeated >=2x is the only valid protocol.
6. **Reasoning about the wrong compiler.** `DEV=AMD` renders via `HIPRenderer`+LLVM; `amd.py`'s
   isel/waitcnt/encoding stack is `DEV=AMD:ISA`-only research tooling. `s_waitcnt`/`s_delay_alu`/`s_clause`
   are LLVM output and no tinygrad edit changes them. But `amd_attention_abi.py` IS in the shipped path
   (`cstyle.py:135/148/457/458`), verified at runtime.
7. **Assuming the algorithm when the renderer was at fault.** The "third softmax butterfly" was
   `Ops.CUSTOMI` inlined regardless of `child_count`, growing the emitted HIP as 2^n (272 textual
   `ds_bpermute` for 64 distinct). The cheap detector: `grep -c ds_bpermute` on the emitted `.cpp` versus
   the count you expect.
