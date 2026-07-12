# ctx512 device-time attribution — where the residual actually is

Date: 2026-07-12

Measured device-time share per kernel from a graph-profile capture of the real
four-role authority (Qwen3-8B-Q4_K_M, ctx512, `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1`
+ multirole-buffer2 candidate set), ranked by tick share. Roles verified by
FLOP/dispatch against `candidate-set.json` + census, not by guessed name tags.

## Ranking

| kernel | role | device-time | achieved | note |
|---|---|---:|---:|---|
| `E_4_96...` | ffn_gate_up (2/layer) | 33.1% | ~36.7 TFLOPS | promoted, most efficient GEMM |
| `E_4_32...383` | ffn_down (1/layer) | 22.9% | ~28.0 TFLOPS | promoted |
| `E_4_32...127` | attn_qo (2/layer) | 15.4% | ~14.4 TFLOPS | promoted, ~12% of peak |
| `E_4_8...` | attn_kv (2/layer) | 7.6% | ~13.8 TFLOPS | promoted, ~12% of peak |
| `prefill_gen_sched_gemm_512_151936_4096` | lm_head | 7.0% | resident fp16 | refuted for packing (see lane-a doc) |
| `r_*` + small `E_*` | non-GEMM (softmax/norm/rope/reductions) | ~14% | — | spread thin |

Four promoted GEMMs = ~79% of device time. Peak fp16 on 7900 XTX ~120 TFLOPS.

## Findings

1. **There is no fat unpromoted lane.** The single biggest kernel (`E_4_96`, 33%)
   is ffn_gate_up — already on the candidate route. It read as "unknown" only
   because `graph_profile_attribution.py::PROVEN_NAMES` was stale (omitted `E_4_96`,
   mislabeled ffn_down as ffn_gate_up). Fixed in this commit series.

2. **Semantic metadata tagging does not reach this export.** Every
   `ProfileGraphEntry.metadata` is `None` in real captures (original and fresh),
   so the `role_metadata` / `semantic_op` path (criterion 1) is inert here — graph
   batching drops per-op metadata before the entry is built. Attribution is
   therefore name/shape-based, not metadata-based.

3. **The residual is roofline headroom, not a missing lane.** The promoted GEMMs
   run at ~12-33% of fp16 peak. The attention GEMMs (attn_qo, attn_kv) are the
   least efficient (~12% of peak) but individually small (15.4% + 7.6%). ffn_down
   (28 TFLOPS, 22.9%) is the largest with clear headroom vs ffn_gate_up's 36.7.

4. **LM head = 7%**, near its resident-fp16 floor; packing refuted (+87 ms).

## Implication for the 30.7 ms residual (147.05 -> 116.36 ms target)

Closing ~21% of wall time when ~79% is already-promoted GEMMs means the levers are,
in order of potential:

- **GEMM efficiency on the candidate route** (biggest): lift ffn_down / attn_qo /
  attn_kv toward ffn_gate_up's 36.7 TFLOPS via tile/wave/occupancy tuning of the
  existing generated kernels. No new lane or route needed.
- **Non-GEMM ~14%**: many small softmax/norm/rope reductions; fusion/launch
  reduction. Individually tiny, aggregate meaningful.
- **LM head**: closed (resident fp16 is near floor).

## Next

Pick the highest-headroom promoted GEMM (ffn_down or the attention GEMMs) and run a
tile/wave search on its existing candidate kernel, measuring kernel-only pinned
delta and whole-prefill delta per the four-gate rule. This is a candidate-kernel
tuning problem, not a new-lane problem.
