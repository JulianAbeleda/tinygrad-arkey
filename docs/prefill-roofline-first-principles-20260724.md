# 8B prefill: the roofline from first principles (2026-07-24)

Written after seven theories were tested in one session (six refuted, one worth +7.3%). Its purpose is
to stop the next arc from re-deriving the frame, and to record the *measurement* errors that made most of
those theories look attractive in the first place.

## 1. The achievable peak is ~105 TFLOPS, not 122.8 and not 61.4

Measured, not assumed: `extra/qk/microbench/wmma_peak.cpp` — pure back-to-back WMMA, 8 independent
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
**Open measurement: per-dispatch counters for ONE GEMM launch, so mix and duty separate.** Until that
lands, nobody should project a GEMM number.

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
| ours, before T6 | −16.76% | −10.6pp | 2.7x ideal |
| **ours, after T6** | **−12.08%** | **−5.9pp** | **2.0x ideal** |
| llama.cpp | −11.51% | −5.4pp | 1.9x ideal |

**Decay is the drift-robust comparison** (a within-run ratio, so the ~5% session drift cancels).
The T6 fix closed the decay gap to llama from **5.25pp to 0.57pp** — context-scaling parity. Absolute
throughput margins vs llama (+1.6% to +3.4%) are SMALLER than the session drift and rest on cross-session
llama numbers, so they are not yet a valid claim. Same-session interleaved llama re-measurement is
required before "we beat llama" goes in any doc.

**All remaining attention headroom is +6.7%**, capped by this floor. Going below it needs sub-quadratic
attention (windowing/sparsity) = a model change, not a kernel change.

## 6. Prize ranking

| lever | whole-model | status |
|---|---|---|
| GEMM 48% → 60% of peak | **+18.4%** | UNEXPLORED — needs the per-dispatch duty measurement first |
| GEMM 48% → 80% of peak | **+46.1%** | UNEXPLORED |
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
