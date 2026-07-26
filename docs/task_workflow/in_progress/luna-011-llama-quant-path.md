# LUNA-011: Q4_K/Q6_K linear-path source map

## Status

`PASS (static selection map; dispatch confirmation pending LUNA-021)`. Source revision and provenance limitation are stated in LUNA-010. ROCm is implemented by compiling `ggml/src/ggml-cuda` with HIP (`ggml/src/ggml-cuda/vendors/hip.h` and `ggml/src/ggml-hip/CMakeLists.txt`), so source symbols retain the `cuda` prefix.

## Graph op to HIP kernel family

| Stage | Graph/source owner | Selection / shape predicate | Expected ROCm/HIP family |
|---|---|---|---|
| Transformer projections and MLP | `src/llama-graph.cpp` builds `ggml_mul_mat` nodes for Q/K/V/O and FFN weights | Quantized model weight (Q4_K or Q6_K), activation tensor, output column count `M = n_tokens` for the graph node. | `mul_mat` backend dispatch in `ggml/src/ggml-cuda/ggml-cuda.cu`; quant matrix-matrix queue (`mmq`) when its type/device/shape gate accepts it. |
| Prompt | Same graph, `M > 1` (or `M = n_ubatch` per prompt microbatch) | MMQ has enough columns to amortize activation quantization and tiles. Exact threshold is backend/device-code dependent and must be retained from trace/source revision, not assumed universal. | Q4_K/Q6_K instantiations from `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu` and `mmq-instance-q6_k.cu`, template logic in `mmq.cuh`/`mmq.cu`. |
| First generated token | Same graph, `M = 1` | Decode has one output token; the generic `mul_mat` dispatcher can select vector-matrix (`mmv`/`mmvq`) rather than MMQ. | `mmvq.cu`/`mmvq.cuh`, quant vector-dot code in `vecdotq.cuh`, or a fallback `mul_mat` implementation. |
| Steady decode | Same as first-token, repeated with growing attention K dimension | `M = 1` remains; weight projections are GEMV-like. Context affects attention and any per-token auxiliary work, not the projection's output-column count. | Same `mmvq`/quant-vector-dot family unless a backend gate or tensor placement causes fallback. |
| Fallback | `ggml-cuda.cu` `mul_mat` operation dispatch and backend BLAS path | Unsupported quant type/layout, unsuitable dimensions, missing MMQ support, or tensor/backend placement can route to dequantization plus GEMM/BLAS or another generic path. | BLAS/dequantize or non-MMQ `mul_mat`; treat trace kernel names as authoritative. |

## Quant algebra and tile ownership

- `Q4_K` and `Q6_K` are K-quants with super-block scales/minima and packed low-bit values. `ggml/src/ggml-quants.h` defines their packed block layouts; CUDA/HIP quant decode and dot-product specializations are in `vecdotq.cuh` and `mmq.cuh`.
- The mathematical contract is `y[m,n] = sum_k dequant_qK(w[n,k]) * a_quant_or_fp(m,k)`, with per-block scale/min reconstruction. MMQ may quantize activations to Q8-like blocks before the dot product; the vector path uses its own quant-dot specialization. Do not label activation quantization as observed until LUNA-021 confirms the selected family.
- Tile ownership is template-defined: MMQ tiles output rows/columns and reduction-K blocks across a HIP grid; MMV/MMVQ assigns output rows (and reduction fragments) to blocks/waves. Grid/block dimensions are launch-time values in `ggml-cuda.cu`, not fixed properties of Q4_K or Q6_K.

## Source-to-dispatch join keys

1. ggml op: `GGML_OP_MUL_MAT`, input types (`Q4_K`/`Q6_K`, F16/BF16 activation), and tensor dimensions.
2. Backend path: MMQ vs MMVQ vs fallback selected in `ggml-cuda.cu`.
3. Compilation unit/template: `mmq-instance-q4_k.cu`, `mmq-instance-q6_k.cu`, `mmvq.cu`.
4. Trace: demangled HIP kernel name, grid/block, stream, and correlated host launch.

## Controls

- Prompt control: trace a prompt microbatch with `M > 1`; expect an MMQ candidate for each offloaded Q4_K/Q6_K linear.
- Decode control: trace exactly one `M = 1` decode; expect MMVQ/vector-dot or explicitly record MMQ/fallback if selected.
- No claim is made that a particular M threshold, tile, or kernel spelling applies to gfx1100 until the LUNA-021 trace confirms it.
