# TPE-4 RESULT - extracted rocBLAS Tensile ffn_gate/up keeps mature-backend speed through HCQ (PASS)

Executed Phase TPE-4 of `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`: measure the selected
rocBLAS Tensile ffn_gate/up kernel after TPE-3 proved it can launch from tinygrad HCQ on tinygrad-owned buffers.
**Verdict: PASS.** The extracted mature-backend primitive is not merely runnable; it keeps the expected speed with no
HIP runtime, no copies, and correct output. Probe: `extra/qk_tensile_hcq_perf.py`; artifact:
`bench/qk-tensile-extraction/hcq_perf.json`.

## Result [M]

Shape: M=512, N=12288, K=4096, fp16 GEMM, 51.54 GFLOP. 40 timed HCQ launches after 8 warmups.

| measurement | result |
|---|---:|
| median device time | **0.7703 ms** |
| median throughput | **66.91 TFLOPS** |
| min / max timed throughput | 54.82 / 72.38 TFLOPS |
| rocprof trace reference | 62.9 TFLOPS / 0.819 ms |
| PXB-1 rocBLAS reference | 60.96 TFLOPS |
| PXB-1 tinygrad plateau | 42.0 TFLOPS |
| vs trace rocBLAS | **1.064x** |
| vs PXB-1 rocBLAS | **1.098x** |
| vs PXB-1 tinygrad | **1.593x** |
| correctness | rel_err 0.000348, max_abs 0.1287 |
| HIP runtime loaded in tinygrad process | no |

The first few warmups were slower (1.34 -> 0.98 ms), then the timed set stabilized around 0.71-0.94 ms. The measured
median exceeds the TPE-4 minimum gate (>=62 TFLOPS) and the >=90% trace-parity gate (>=56.61 TFLOPS).

## Gate table

| TPE-4 gate | result |
|---|---|
| correct output vs tinygrad fp16 oracle | PASS, rel_err 0.000348 |
| no HIP runtime / rocBLAS / hipBLASLt in-process | PASS |
| >=90% of trace rocBLAS throughput | PASS, 66.91 / 62.9 = 1.064x |
| >=62 TFLOPS minimum | PASS, 66.91 TFLOPS |

## Interpretation

Lane B is now proven past the main risk boundary. The performance ceiling measured in a separate HIP-only process is
reachable from tinygrad's HCQ path by loading the Tensile code object, resolving the named `.kd` descriptor, and filling
the recovered 128-byte kernarg directly with tinygrad buffer VAs.

This is not a tinygrad machine-search win yet. It is an executable target primitive: one mature schedule that tinygrad
can now call without HIP. The next research question is whether this should remain an external-artifact primitive, or
whether the recovered schedule becomes the oracle for codegen transfer / machine-search rows.

## Consequence

**TPE-4 PASS -> proceed to TPE-5 shape matrix.** Repeat TPE-1 through TPE-4 for the other high-share prefill matmul
roles (`ffn_down`, `attn_q/o`) before any model route. The shape matrix decides whether the extracted primitive has
enough weighted full-prefill upside and whether each role needs a separate opaque contract.

Do not route this into the model yet:

- this is one fixed ffn_gate/up shape only;
- the named-descriptor loader is still probe-local, not runtime code;
- the shape-matrix and weighted pp512 model are not complete;
- external-artifact policy has not been accepted as a default.

## Files

`extra/qk_tensile_hcq_perf.py`, `bench/qk-tensile-extraction/hcq_perf.json`, this doc. Provenance:
`prefill-tensile-tpe3-hcq-launch-result-20260619.md`, `bench/qk-tensile-extraction/{selection.json,kernarg_capture.json}`.
No kernel/model/default changes; no runtime files modified.
