# llama.cpp MMVQ Notes

Date: 2026-06-12

Pinned upstream commit: `ba1df050f3dc7827fc64936b2e24fe499c9f74eb`

Source links:

- `mmvq.cu`: https://github.com/ggml-org/llama.cpp/blob/ba1df050f3dc7827fc64936b2e24fe499c9f74eb/ggml/src/ggml-cuda/mmvq.cu
- `vecdotq.cuh`: https://github.com/ggml-org/llama.cpp/blob/ba1df050f3dc7827fc64936b2e24fe499c9f74eb/ggml/src/ggml-cuda/vecdotq.cuh
- `common.cuh`: https://github.com/ggml-org/llama.cpp/blob/ba1df050f3dc7827fc64936b2e24fe499c9f74eb/ggml/src/ggml-cuda/common.cuh

## Findings

- `mmvq.cu:10-24` maps `GGML_TYPE_Q4_K` and `GGML_TYPE_Q6_K` decode to
  `vec_dot_q4_K_q8_1` and `vec_dot_q6_K_q8_1`.
- `mmvq.cu:38-52` maps the same types to `VDR_Q4_K_Q8_1_MMVQ` and
  `VDR_Q6_K_Q8_1_MMVQ`.
- `mmvq.cu:1185-1192` stages the activation side into `block_q8_1` storage
  before MMVQ runs. This means the packed dot is not an isolated instruction
  swap; llama.cpp changes the activation representation first.
- `vecdotq.cuh:501-527` implements Q4_K x q8_1 with `ggml_cuda_dp4a` for both
  the quant dot and the Q4_K min-correction sum.
- `vecdotq.cuh:620-644` implements Q6_K x q8_1 with `ggml_cuda_dp4a`.
- `common.cuh:694-732` defines `ggml_cuda_dp4a`. On HIP RDNA3/RDNA4 it lowers
  through `__builtin_amdgcn_sudot4(...)`; on CUDA it lowers through `__dp4a`.
- `mmvq.cu:404-423` has an RDNA3-specific single-column warp policy. Q6_K gets
  two warps, while Q4_K is not in that RDNA3 whitelist and falls back to the
  default. The llama.cpp schedule is type- and architecture-specific, not simply
  "more warps".

## Interpretation

llama.cpp does agree that packed 4-byte dot instructions are part of its fast
Q4_K/Q6_K decode path on RDNA3. It does not prove that isolated tinygrad
renderer emission of `v_dot4` is the next lever.

The upstream path combines:

- q8_1 activation staging;
- packed Q4/Q6 lane extraction;
- packed dot emission;
- scale/min correction around that packed dot;
- RDNA-specific MMVQ scheduling choices.

That matches the local negative results: the serial vdot candidate, the parallel
`Ops.CUSTOMI` vdot candidate, and the v1 roofline check all say instruction
emission alone is not enough. If compiler research continues, the target should
be a semantic packed-layout plus schedule/codegen package, not a standalone
`v_dot4` renderer peephole.
