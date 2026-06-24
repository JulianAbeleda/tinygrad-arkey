# SCOPE — Measure the 4 prefill primitives via PMC (tinygrad WMMA vs Tensile, gfx1100, 2026-06-19)

Follow-up to `prefill-occupancy-lever-result-20260619.md` (Tensile prefill is a real byte-identical 1.84× win) and
the fresh disasm (the gap is dataflow/scheduling around the WMMA, not the matrix op). Static instruction counts show
the STRUCTURE; this scope MEASURES the runtime effect of each of the 4 primitives with hardware counters, to confirm
which actually dominate.

The 4 primitives and their counters (all present in `tinygrad/runtime/autogen/am/pmc.py`, driven by
`extra/qk_pmc_capture.py` + `PMC_COUNTERS=`):
1. **LDS operand staging / reuse** → `SQ_INSTS_LDS_LOAD`/`_STORE`, `SQC_LDS_IDX_ACTIVE`, `SQC_LDS_BANK_CONFLICT` +
   `GL2C_MC_RDREQ` (DRAM reads), `GL2C_HIT`/`MISS`.
2. **Occupancy / latency hiding** → `SQ_LEVEL_WAVES`, `SQ_WAVES`, `SQ_WAVES_LT_16/32/48/64`.
3. **Software-pipelined K-loop / stalls** → `SQ_WAIT_INST_ANY` (stall cycles), `SQ_INST_LEVEL_VMEM` (loads in flight =
   prefetch depth), `SQ_BUSY_CYCLES`÷`GRBM_GUI_ACTIVE` (issue util).
4. **Per-op VALU overhead** → `SQ_INSTS_VALU`, `SQ_INST_CYCLES_VALU` (+ static disasm mix).

Measure-only. No kernel/route/default change.

## Phase plan
- **M0 — Settle caveat 2 (WMMA counter).** Add `SQ_INSTS_MFMA, SQ_VALU_MFMA_BUSY_CYCLES, SQ_INSTS_VALU_MFMA_F16` on a
  known `v_wmma` kernel; if any increments, RDNA3 WMMA is counted (→ direct WMMA-busy metric); else WMMA throughput
  stays TFLOPS-only.
- **M1 — Settle caveat 1 (Tensile capture).** Run the isolated `tensile_gateup` (TensileRunner via `route_pf16`)
  under PMC; confirm the counters are non-zero / sane (i.e. PMC brackets the patched-runtime dispatch). If zero/garbage
  → fall back to standalone-HCQ measurement, or accept Tensile is PMC-opaque.
- **M2 — The A/B matrix.** Isolated `gateup` GEMM (out=12288, in=4096, T=512), tinygrad WMMA vs Tensile, captured
  across 2–3 counter passes (≤~8 counters/pass due to per-block register limits). Pick the dominant matmul kernel per
  side (max `GRBM_GUI_ACTIVE`). Report per-primitive deltas.
- **M3 — Attribute.** Map each primitive's measured delta to the throughput gap; identify which primitive(s) dominate
  the 42→66 TFLOPS / 1.84× difference. Cross-check vs the static disasm (LDS 0 vs 24.5KB, 1 wave vs 32, etc.).

## Gates / hazards (from `[[amd-decode-measurement-confounds]]`)
- PMC perturbs per-kernel TIMING → trust counter ratios, NOT PMC-run wall time.
- ≤~8 counters per block per pass (`out of perfcounter registers`) → split into passes.
- Counters sum across 32×GL2C / SQ instances (decode handles it); aggregate trustworthy, per-instance not.
- Isolated-kernel numbers must be sanity-checked vs the in-model role (isolated can mislead — the recurring lesson).

## Deliverables
- `docs/prefill-primitive-pmc-result-20260619.md`; `extra/qk_prefill_primitive_pmc.py`;
  `bench/qk-prefill-boost/primitive_pmc.json`; README pointer. No default/route change.

## Done
A per-primitive measured table (LDS traffic, occupancy waves, stall/in-flight, VALU) for tinygrad-WMMA vs Tensile
gateup, with the two caveats resolved (WMMA-counter yes/no; Tensile-PMC-capture yes/no), and a verdict on which
primitive(s) dominate the gap — OR an explicit statement of which primitive remains unmeasurable and why.
