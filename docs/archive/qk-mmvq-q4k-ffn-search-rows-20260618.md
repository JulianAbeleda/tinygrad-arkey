# Q4_K ffn_gate/up full-MMVQ — search rows (2026-06-18)

Derived from the Phase-1 audit (`qk-mmvq-q4k-ffn-full-arc-20260618.md`): the role is **fp-dequant-ALU-bound**
(coop 48%, READRAW 70%). So the search is **not** dataflow — it is making the dequant cheap.

## Rows

### A. `q4k_ffn_mmvq_q8` — EARNED (primary, first prototype)
q8_1 int8 activations (quantized once/linear) + **dp4a int-dot** inside the **coalesced coop lane structure** +
int32 accumulation + fp affine epilogue. Directly attacks the dequant ALU; READRAW proves the ~70% roofline.
- **knobs:** q8 block size, q8 scale layout, rows/workgroup, lanes/row (lane4), vector width, int-accum layout,
  affine-epilogue placement.
- **legality:** fp/int-reassoc tol vs current output (must keep greedy byte-identical in-model); **q8 quant cost
  counted** (whole-linear, not dot-only); no dense fallback; no default until in-model gate.
- **isolated kill gate:** whole-linear (incl q8 quant) <1.3× over base (365 GB/s), or >HBM peak, or less-work, or
  cold-input win vanishes → stop.
- **in-model ship gate:** W==D, greedy byte-identical (or accepted tol), ctx512 ≥+5% AND ctx1024 ≥+5%, no
  ctx4096/prefill regression.
- **expected e2e:** +5-12% (44% of weight traffic, 41%→~70%). **risk: high** (dp4a-as-toggle was +1% — A must be
  the full co-design, not a toggle).
- **files:** `extra/q4_k_gemv_primitive.py` (new `q4k_coop_dp4a_partial_kernel` + q8 pack), `tinygrad/llm/model.py`
  (`Q4K_FFN_FULL_MMVQ` opt-in route).

### B. `q4k_ffn_unpack_dataflow` — REFUTED by audit
coop already 48%; READRAW (no dequant) = 70% proves the schedule is not the limiter. Better dataflow cannot close
the 48→70 gap (it is dequant ALU). **Do not prototype.**

### C. `q4k_ffn_epilogue_reduce` — low-EV deferred
stage-2 `.sum` + 2 kernels/linear is a small fraction; fold into A's epilogue only if A ships.

## Decision
Skip B (refuted by audit). First prototype = **A** (q8_1+dp4a coop). It is a new-kernel-family build (high-risk,
uncertain — the dp4a-toggle precedent was +1%), so it warrants an explicit go before the build. `search_rows.json`
mirrors this.
