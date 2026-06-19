# TPE-5 RESULT — extracted Tensile primitive generalizes across prefill roles (PASS, ~1.40× weighted pp)

Executed Phase TPE-5 of `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`: repeat the TPE-1→TPE-4
extract/launch/verify/time sequence for the other high-share prefill GEMM roles (`ffn_down`, `attn_q/o`), build the
shape matrix, and compute the weighted full-prefill model. **Verdict: PASS.** All three high-share roles launch
correctly and stably through tinygrad HCQ with **no HIP runtime, no copies, no workspace, no aux kernels, no layout
copies**, and the weighted model predicts **~1.40× full warm pp512** if all three are routed — above the 1.25× gate.
Research/probe only; no model route, no defaults, decode untouched. Probes:
`extra/qk_tensile_{kernarg_capture.cpp,shape_matrix.py,hcq_launch.py}`; artifacts:
`bench/qk-tensile-extraction/{shape_matrix.json,kernarg_all.jsonl}`.

## Shape matrix [M] (RX 7900 XTX / gfx1100, fp16→fp32, rocBLAS col-major C[m,n]=A[m,k]·B[k,n])

| role | M,N,K | kernel | ksz | median ms | median TFLOPS | % of ref | × tinygrad(42) | rel_err | workspace | gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| ffn_gate/up | 512,12288,4096 | MT128x128 AMAS0 (GSU1, SU0) | 128 | 0.77 | **66.8** | 109% (60.96) | 1.59× | 3.5e-4 | none | ≥62 ✓ |
| ffn_down | 512,4096,12288 | MT128x128 AMAS3 (**StreamK SU32**) | 132 | 0.75 | **68.9** | 97% (70.9) | 1.64× | 2.8e-4 | none | ≥62 ✓ |
| attn_q/o | 512,4096,4096 | MT128x128 (GSU1, SU0) | 128 | 0.29 | **58.9** | 77% (76.7) | 1.40× | 4.2e-4 | none | <62 |

All three: **correct** (rel_err < 5e-4 vs tinygrad fp16 oracle) and **stable** (≤1e-4 spread over 4 re-zeroed runs).

### Key per-role findings
- **ffn_down is StreamK** (`SU32_SUS256`, 20-arg kernarg with an extra `OrigStaggerUIter` field, captured size 132B) —
  yet it launches correctly through HCQ with **no external workspace and no PostGSU/fixup kernel**: there is no
  workspace pointer in the kernarg, so the StreamK-named kernel writes C directly for this shape. Keeping the captured
  kernarg verbatim (substituting only the 4 Address VAs at the fixed offsets 16/24/32/40) was sufficient — no
  special-casing. This retires the scope's main ffn_down risk (StreamK/GSU workspace orchestration).
- **attn_q/o** (small K=4096) lands at **58.9 TFLOPS — below the 62 individual gate** (and 77% of its 76.7 ref) but
  still **1.40× the tinygrad plateau**, so it contributes net-positively to the weighted model. It is not a blocker;
  it is the weakest role.
- All roles use the **same `Ailk_Bljk` Tensile code object** and the **same pointer offsets** — one extraction path,
  not three opaque contracts. Maintainability is good.

## Weighted full-prefill model (anchored on measured PREFILL_V2)
PREFILL_V2: forward 245 ms, matmul bucket ~181 ms (74%), non-matmul ~64 ms. Each role's tinygrad time = matmul bucket
× FLOP share; replacing a role swaps in its measured Tensile-HCQ time × per-layer-count × 36 layers (so the
all-tinygrad case == the measured 245 ms). FLOP shares/layer: ffn_gate+up 52%, ffn_down 26%, attn_q+o 17%, attn_k/v 4%.

| replace | forward ms | pp speedup vs PREFILL_V2 |
|---|---:|---:|
| ffn_gate/up only | 206.1 | **1.19×** |
| + ffn_down | 185.9 | **1.32×** |
| + attn_q/o | 175.4 | **1.40×** |

**~1.40× full warm pp** ⇒ PREFILL_V2 ~2090 tok/s → **~2920 tok/s ≈ 95% of llama (~3069)**. (attn_k/v ~4% left at
tinygrad speed — low EV per the scope.) This is consistent with the earlier `qk_prefill_blas_sequence` whole-matmul
rocBLAS projection (~1.34× when also holding non-matmul fixed); the per-role HCQ measurement is slightly higher
because the HCQ launch path has lower overhead than rocBLAS's own dispatch.

## Gate table → PASS
| TPE-5 gate | result |
|---|---|
| all roles correct vs oracle | ✓ (rel < 5e-4) |
| all roles stable across repeated runs | ✓ |
| no layout copies / no separate HIP preprocessing | ✓ |
| no workspace / opaque per-role contract | ✓ (one code object, same offsets; StreamK needs none) |
| weighted full pp ≥ 1.25× | ✓ **1.40×** |
| KILL conditions (one-shape-only / <1.15× / near-plateau / per-role opacity) | none triggered |

## Verdict + next step
**TPE-5 PASS → proceed to TPE-6 (one-block transfer / minimal runtime-helper design).** The extracted Tensile
primitive generalizes: three high-share prefill roles run correct, stable, and fast (1.40×–1.65× tinygrad) through
HCQ from one code object with one pointer-offset convention and no workspace. The weighted upside (~1.40× full pp,
~95% of llama) clears the gate. TPE-6 should route a single prefill block/layer behind a research flag using
PREFILL_V2 fp16 weights, with the named-descriptor loader promoted from probe-local to a tightly-scoped runtime helper
only as needed — still no model default, decode untouched, external-artifact policy still pending separate review.

Caveats (not blockers): attn_q/o is below the 62 individual gate (1.40× tinygrad regardless); the kernarg is captured
once per shape from a separate HIP-only run (the contract is shape-specific); the weighted model is an Amdahl estimate
anchored on PREFILL_V2's measured buckets (real routing overhead measured in TPE-6).

## Files
`extra/qk_tensile_kernarg_capture.cpp` (multi-role + symbol via hipModuleGetFunction), `extra/qk_tensile_shape_matrix.py`,
`bench/qk-tensile-extraction/{shape_matrix.json,kernarg_all.jsonl}`, this doc. Reuses `qk_tensile_hcq_launch.py`
(NamedAMDProgram), `qk_prefill_blas_ceiling.cpp`. Provenance: `prefill-tensile-tpe4-perf-result-20260619.md`. No
kernel/model/default changes; no runtime files modified.
