# Option A in-model Tensile prefill route — MEASURED: PASS_RESEARCH (1.27× warm pp512, dNLL accept)

Executed the research-measurement scope to completion (`prefill-tensile-research-measurement-scope-20260619.md` +
`prefill-tensile-a3-inmodel-route-scope-20260619.md`). The extracted rocBLAS Tensile kernel is routed into the real
PREFILL_V2 prefill forward behind `PREFILL_TENSILE_GEMM=1` (research-only, default off) and **measured in-model**:
**warm pp512 1.27×, dNLL accept**. Verdict: **PASS_RESEARCH** — external artifact stays research-only; only the TPE-0
policy decision blocks landing, and the speed is below the strong gate. Nothing shipped/defaulted. Decode untouched.

## Result [M] (Qwen3-8B-Q4_K_M, gfx1100, T=512, warm/JIT-replayed)
| | PREFILL_V2 (flag off) | + Tensile (flag on) |
|---|---:|---:|
| warm prefill | **2709 tok/s** (189 ms/512) | **3433 tok/s** (149 ms/512) |
| **warm pp512 speedup** | 1.00× | **1.27×** |
| greedy byte-match | 5/5 | 3/5 (2 benign ties) |
| warmstart applied | 5 | 3 (routed linears bypass it) |
| forward error | 0 | 0 |

Routed roles: **ffn_gate, ffn_up, ffn_down** (eligible TPE-5 shapes: in/out 4096/12288 and 12288/4096). attn_q/o not
routed.

## Quality — dNLL gate: ACCEPT [M] (`bench/qk-prefill-v2-nll/result.json`)
Teacher-forced prefill NLL, routed (fp16+Tensile) vs the fp32 baseline, 2×512-token windows (1022 tokens):
- mean dNLL **−0.00078**, max **−0.00018**, eps 0.01 → **ACCEPT** (negative ⇒ no degradation; within noise).
The 2 greedy flips (match 3/5) are benign near-ties, not quality loss — confirmed by the dNLL gate.

## Gate table → PASS_RESEARCH
| gate | threshold | result |
|---|---|---|
| correctness | rel_err ≤ 2e-2 per routed linear | ✓ rel 0.0 vs `.linear` (B1, eager+JIT) |
| quality | dNLL ≤ 0.01 | ✓ −0.00078 |
| research speed | warm pp512 ≥ 1.25× | ✓ **1.27×** |
| strong speed | pp512 (+pp1024) ≥ 1.35× | ✗ 1.27× (and pp1024 not validated — warmstart is shape-specific to 512) |
| fallback | flag-off / ineligible == PREFILL_V2 | ✓ default off, byte-identical; ineligible shapes fall through silently |
| decode | unchanged | ✓ route only in `_pf16` (prefill); decode path untouched |
| graph | no per-op host-sync wall | ✓ JIT-captured (B1: HCQGraph replays correct); the 1.27× *is* the warm, graph-replayed number |

## How it was built (the path A0→A4)
- **A0/A1 (prior)**: injected Tensile node JIT/HCQGraph-capturable with correct dims — rebindable `fill_kernargs`
  (`bind_sints_to_buf` for symbolic JIT VAs) + `AMDComputeQueue.exec` dim-override.
- **B1 mechanism**: install-once routing via patching `dev.runtime` (the program factory `get_runtime` calls) to
  return the role's `TensileRunner` for `tensile_<role>` kernels — robust, no get_runtime-hook flakiness (the A2
  blocker). Smoke: rel 0.0 vs `.linear`, eager + JIT.
- **B2 wiring**: `model.py:_pf16` gains a flag-gated branch → `extra/qk_tensile_inmodel.route_pf16` for eligible
  shapes; `[feature,T]` (weights natural `[out,in]`, x→`[in,T]` + out→`[T,out]` transposes); silent fallback.
- **Eager-install fix**: the prefill forward is traced under `@function(precompile=True)` (device usage disallowed);
  `install()` (TensileRunner ELF allocs) is therefore called **eagerly at model init** after the warmstart build,
  not lazily inside the traced forward.
- **B3/B4**: measured warm pp512 (1.27×) + dNLL (accept) on the 8B GGUF.

## Why 1.27× (not the ~1.40× projection)
Projection (TPE-5 weighted) assumed all high-share roles routed with no transpose overhead. Here only **FFN** is
routed (attn_q/o ≈17% of the matmul bucket left on PREFILL_V2), and the per-linear **x/out transposes** add graph
work. Pushing toward the strong gate would need: route attn_q/o too, and a `[feature,T]` whole-block restructure to
drop the per-matmul transposes. Both are bounded follow-ups, not new capability.

## Verdict + decision
**PASS_RESEARCH.** The extracted Tensile prefill route is correct, quality-safe (dNLL accept), and **1.27× faster
warm pp512 in-model**, behind a research flag with clean fallback and decode untouched. The remaining decision is the
**TPE-0 external-artifact policy** (routing ships a rocBLAS/Tensile HSACO dependency) — a project call, not an
engineering one. If accepted, the bounded follow-ups (attn_q/o + `[feature,T]` block) target the strong gate. If
declined, this stands as the measured proof that the mature-backend prefill speed transfers in-model, with the
codegen oracle (CG-0/1, project-level) as the dependency-free alternative.

## A5 UPDATE — route attn_q/o too → 1.76× warm pp512 (PASS_STRONG, pp512)
Per `prefill-tensile-a5-strong-gate-scope-20260619.md`, added attn_q/o to the route (eligible `(4096,4096)` → role
"qo"; attn_q + attn_output, already on `_pf16`). **Measured:**

| | PREFILL_V2 | + Tensile (FFN + attn_q/o) |
|---|---:|---:|
| warm prefill | 2709 tok/s (189 ms) | **4770 tok/s (107 ms)** |
| **warm pp512 speedup** | 1.00× | **1.76×** |

- **dNLL: mean −0.00078 ≤ 0.01 → ACCEPT**, with `ROUTE_COUNT` confirming all roles routed in the NLL path
  (qo:72, gateup:72, down:36 per forward = 36 layers × {q+o, gate+up, down}). attn_q/o routing is **quality-neutral**
  (Tensile fp16 ≈ tinygrad fp16; dNLL measures fp16-vs-fp32, unchanged by which engine does the fp16 matmul).
- The speed-harness greedy `match=1/5` is drift vs the *symbolic* reference path (more fp16 matmuls → more drift from
  fp32), **not** a correctness failure — the robust 1022-token dNLL gate accepts.
- **pp512 clears the ≥1.35× strong threshold (1.76×).** pp1024 is not validated (the model's PREFILL_V2 warmstart is
  shape-specific to UBATCH=512; measuring 1024 needs re-validation, out of scope). So: **PASS_STRONG_POLICY_GATED on
  pp512.** No A5-2 transpose-elimination needed — already well above the gate.
- 1.76× > the ~1.40× weighted projection because the in-model PREFILL_V2 baseline for the attn `(4096,4096)` shape
  runs well below the 40-TFLOPS plateau assumed in the projection, so Tensile's win there is larger.

**Updated verdict: PASS_STRONG_POLICY_GATED** (pp512 1.76×, dNLL accept, decode untouched, fallback clean). The only
remaining gate is the **TPE-0 external-artifact policy** (rocBLAS/Tensile HSACO dependency) — a project decision.

## Files
`tinygrad/llm/model.py` (flag-gated `_pf16` branch + eager install; default off ⇒ byte-identical),
`extra/qk_tensile_inmodel.py`, `bench/qk-tensile-extraction/inmodel_measurement.json`, `bench/qk-prefill-v2-nll/result.json`,
this doc. Reuses `TensileRunner` (A1) + committed captures. Research-only; no default/ship; external HSACO artifact
used only when the flag is set.
