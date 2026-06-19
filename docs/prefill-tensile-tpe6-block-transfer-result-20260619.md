# TPE-6 RESULT — FFN-block transfer: correctness + GPU speedup transfer; end-to-end needs graph integration (REDIRECT)

Executed Phase TPE-6 of `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`: route a whole prefill
**FFN block** (gate, up, silu·up, down) through the extracted rocBLAS Tensile kernels via tinygrad HCQ and compare to
the pure-tinygrad fp16 block. **Verdict: REDIRECT.** The kernels transfer with **exact correctness and a real ~1.53×
GPU-matmul speedup** over the PREFILL_V2 plateau, **with no per-matmul transpose and no weight copy** — but naive
per-op routing in a JIT-less probe adds large host sync overhead that swamps the GPU win end-to-end. Realizing the
gain in-model requires a **single-dispatch graph (HCQGraph/TinyJit) runtime helper** — which is TPE-6's own stated
deliverable. Not a KILL (the win is real), not a clean end-to-end PASS. Probe:
`extra/qk_tensile_block_transfer.py`; artifact: `bench/qk-tensile-extraction/block_transfer.json`. No model.py route,
no defaults, decode untouched.

## The layout result that makes routing copy-free [M]
`model.py:46` computes `out[T,out] = x[T,in] @ W[out,in].T`, with `W` realized as natural `[out,in]` fp16 (PREFILL_V2
`_pf16_w`). The TPE-5 algebra shows the captured kernel wants **B = W in exactly that `[out,in]` layout (no
transpose/copy)**, A = activation as `[in,T]`, C = output as `[out,T]`. So running the block in **`[feature,T]`
space** needs **zero per-matmul transposes**: gate/up outputs `[FF,T]` feed `silu·up` `[FF,T]` which feeds `ffn_down`
as its A directly. Only one entry transpose (`x[T,in]→[in,T]`) and one exit transpose are charged to routing. The
shared intermediate confirms the kernels compose, not just run in isolation.

## Result [M] (T=512, hidden=4096, ffn=12288, 154.6 GFLOP, fp16)

| measurement | value |
|---|---:|
| correctness (rel_err vs tinygrad fp16 oracle) | **4.8e-4** (exact) |
| routed GPU matmul time (3 kernels: gate 0.82 + up 0.81 + down 0.90 ms) | **2.53 ms** |
| routed GPU throughput | **61.0 TFLOPS** |
| PREFILL_V2 plateau (~40 TFLOPS) for same FLOP | 3.87 ms |
| **block matmul GPU speedup vs PREFILL_V2** | **1.53×** |
| routed naive per-op wall (host-overhead-dominated) | 8.69 ms |
| host overhead (naive wall − GPU matmul) | **6.16 ms** |
| default tinygrad wall (no warmstart, context only) | 44.6 ms |

## Interpretation — why REDIRECT, not PASS or KILL
- **The kernels transfer.** Correctness is exact and the block matmuls run at 61 TFLOPS = 1.53× the PREFILL_V2
  tinygrad plateau — the TPE-4/TPE-5 isolated speed survives a real multi-matmul block with shared intermediates.
- **The copy-free routing holds.** Weights stay in their natural `[out,in]` layout (no transpose), the FFN
  intermediate feeds `ffn_down` directly, and there is no weight copy — clearing the scope's "model layout forces
  transposes/copies" kill.
- **But naive per-op routing fails the "≥1.20× after all routing overhead" gate.** The 6.16 ms host overhead (each
  `.realize()` / `wait=True` is a separate tinygrad schedule + GPU sync; ~6 dispatch cycles) dwarfs the ~1.3 ms GPU
  saving, so the end-to-end wall (8.69 ms) is worse than a PREFILL_V2 block. **This overhead is a JIT-less probe
  artifact**: in the real model the forward is TinyJit-captured into one graph with no per-op host sync. The win is
  reachable only if the raw Tensile launches are captured as nodes in that forward graph.
- **That graph integration is the TPE-6 deliverable itself** ("minimal runtime-helper design"), and it is the scope's
  open kill-gate question ("TinyJit/HCQGraph cannot represent the call without material overhead"). This probe does
  not resolve it either way — it proves the GPU-time ceiling is worth chasing and isolates graph integration as the
  single remaining blocker. Hence **REDIRECT to the runtime-helper design**, not a premature PASS.

## Verdict + next step
**TPE-6 REDIRECT → design + build the minimal single-dispatch runtime helper** that lets the extracted Tensile kernel
be enqueued as a node in tinygrad's forward graph (HCQGraph/TinyJit-capturable), so the 1.53× GPU-matmul speedup
survives end-to-end without per-op host sync. Concretely: wrap `NamedAMDProgram` launch as a graph-capturable op (or
an HCQGraph segment) operating on tinygrad buffers, behind a research flag, then re-run the block gate (≥1.20× after
overhead) and proceed to TPE-7 (full in-model warm pp512 + dNLL gate) only if it clears. Still no model default;
external-artifact policy remains a separate decision. If graph capture proves to carry irreducible material overhead,
that is the TPE-6 KILL and the route rests at PREFILL_V2 (with the extracted kernels retained as a codegen-transfer
oracle).

## Files
`extra/qk_tensile_block_transfer.py`, `bench/qk-tensile-extraction/block_transfer.json`, this doc. Reuses
`qk_tensile_hcq_launch.py` (NamedAMDProgram) + `kernarg_all.jsonl`. Provenance:
`prefill-tensile-tpe5-shape-matrix-result-20260619.md`. No kernel/model/default changes; no runtime files modified.
