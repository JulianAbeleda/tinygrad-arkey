# NV decode parity P6-C — exact Q4_K fused gate/up oracle

Date: 2026-08-04
Status: **construction and isolated numerical correctness closed**

## Result

The original P6-B numerical failure was a checker defect, not a launch or ABI
defect.  The CUDA source implements `result * silu(gate_value)`.  The host
reference computed only `silu(gate_value)`, omitting the leading `result`.

After correcting that reference, the extracted llama `Q4_K, ncols=1,
has_fusion=true` cubin entry passes:

| Arm | shape | max absolute error | verdict |
| --- | --- | ---: | --- |
| ordinary Q4_K MMVQ | 1024 x 4096 | 4.53e-6 | PASS |
| fused, gate pointer aliases up | 1024 x 4096 | 3.43e-5 | PASS |
| fused two independent Q4_K weights | 12288 x 4096 | 4.01e-5 | PASS |

The checker uses `1e-3` absolute tolerance. Relative error is reported but
not used as a gate because a correct GLU output crosses zero.

Raw payloads are `/tmp/llama_q4k_plain_1024x4096.json`,
`/tmp/llama_q4k_same_gate_1024x4096.json`, and
`/tmp/llama_q4k_fused_exact_12288x4096.json`.

## Interpretation and stop gate

This makes the llama fused primitive a valid *diagnostic* construction. It
does not establish a parity-sized kernel opportunity. The independent role
ledger estimates tinygrad's two Q4 cores at about 1489.7 us/token and llama's
36 fused MMVQs plus packs at about 1473.4 us/token: roughly 16 us before
removing the exposed SiLU/multiply/cast work. Therefore the next admissible
test is one layer, exact token/logit-gated CUDA graph replacement, with a
separate epilogue/topology accounting row. Do not time or promote all 36
layers until that one-layer control shows a measurable wall improvement.
