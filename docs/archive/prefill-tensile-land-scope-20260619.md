# SCOPE — land the Tensile fp16 prefill route (the measured #2 lever) from PASS_RESEARCH → strong gate / ship

## Why (measured, this session)
Prefill is compute/WMMA-bound (atlas: L2 hit 54–87%). Throughput hierarchy (measured):
tinygrad fp16-WMMA **~41 TFLOPS (34% peak)** < llama int8-MMQ **~49** < **Tensile fp16 66 (54%)**. So tinygrad
loses prefill to llama today (82%), and **Tensile fp16 (66) beats both** → extracting/using the
confirmed-present rocBLAS Tensile `.co` is the correct prefill lever (no int8 GEMM needed).
`docs/decode-bandwidth-bound-pmu-learning-20260619.md`.

## State (already built — PASS_RESEARCH)
The extracted rocBLAS Tensile fp16 kernel is **routed into the real PREFILL_V2 forward** behind
`PREFILL_TENSILE_GEMM=1` (default off, research-only), JIT/HCQGraph-capturable, NO core `ops_amd.py` edits:
- Mechanism: install-once `dev.runtime` patch → `tensile_<role>` kernels run a role `TensileRunner`
  (`extra/qk_tensile_inmodel.py` `route_pf16`/`install`); `model.py:_pf16` flag-gated branch; silent fallback.
- **All 3 eligible roles route** (verified this session): `ROUTE_COUNT {qo:144, gateup:144, down:72}` — qo (attn
  q+o, 4096×4096, extracted @76.7 TFLOPS), gateup (4096×12288 @60.96), down (12288×4096 @70.9). k/v (4096×1024)
  not eligible (no extracted kernel); attn-core stays tinygrad.
- **Quality: dNLL ACCEPT** (−0.00078, eps 0.01, 1022 tok). Correctness rel_err ≤ 3.7e-4 per routed linear.
- **Speed (prior careful measure, ffn-only): warm pp512 1.27×** (2709→3433 tok/s). Below the 1.35× strong gate.
- Docs: `prefill-tensile-inmodel-measurement-result`, `…-tpe7cd-injection-result`, `…-a5-strong-gate-scope`.

## Blockers to landing (only two)
1. **Strong-gate speed (≥1.35× pp512, validate pp1024).** Prior 1.27× was ffn-only; qo now also routes → the
   real all-3-role number is unmeasured cleanly. Must measure **clean, clock-controlled, back-to-back** OFF vs ON.
2. **Deps/artifact policy (TPE-0).** This bundles a **vendored rocBLAS Tensile `.co`** (gfx1100-specific binary
   blob) — NOT pure dependency-free. Lighter than a runtime rocBLAS dep (HSACO launched via HCQ, no librocblas),
   but still a vendored artifact. **User decision required before shipping as a default** (the work below keeps it
   flag-gated/research until that call).

## Plan (do)
- **P1 — clean A/B measurement (all 3 roles).** One process, model loaded once: build PREFILL_V2 prefill JIT with
  `PREFILL_TENSILE_GEMM` OFF → warm (sustained clock) → median pp512; toggle the module flag, rebuild JIT, ON →
  warm → median; interleave/repeat for clock fairness (the clock-ramp confound is severe — `amd-decode-measurement-confounds`).
  Report clean speedup + ROUTE_COUNT. **This is the number that decides the strong gate.**
- **P2 — pp1024.** Warmstart is 512-shape-specific; the Tensile kernels are T=512-tile. Validate the route at
  ubatch 1024 (two 512-tiles or extract a 1024 kernel); confirm speedup holds and dNLL accepts.
- **P3 — re-confirm dNLL** with all 3 roles routed (prior was ffn-emphasis); gate ≤ 0.01.
- **P4 — ceiling.** If pp512 all-roles ≥ 1.35× and clean: strong gate met → only the TPE-0 policy blocks shipping.
  If < 1.35×: report the honest ceiling (routed matmuls are ~74% of prefill; Amdahl caps e2e even at Tensile 66).
- **P5 — landing (gated on user TPE-0 call):** keep `PREFILL_TENSILE_GEMM` flag; if policy allows the vendored
  `.co`, document the artifact + provenance and the supported shape/arch matrix; decode untouched (prefill-only).

## Gates
correctness rel_err ≤ 2e-2/linear ✓ · dNLL ≤ 0.01 ✓ · **strong pp512 ≥ 1.35× (+pp1024)** = P1/P2 target ·
fallback flag-off byte-identical ✓ · decode W==D untouched ✓ · graph: warm JIT-replayed number (no host-sync wall) ✓.

## Honest ceiling
Routed matmuls (ffn gate/up/down + attn q/o) are ~74% of prefill GPU time; attention-core, k/v, norms, rope stay
tinygrad. Even with Tensile at 66 vs tinygrad 41 on the routed part (~1.6×), Amdahl caps e2e: ~1/(0.26 + 0.74/1.6)
≈ **1.37× theoretical max** e2e — so the strong gate (1.35×) is near the ceiling, achievable only if the routed
kernels hit their isolated TFLOPS in-model and transposes/overhead are cheap. P1 measures where we actually land.
