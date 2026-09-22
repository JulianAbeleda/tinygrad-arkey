// Standalone research wrapper around llama.cpp's exact Q6_K MMVQ template.
#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cu"

extern "C" void q6k_m1_vocab_raw(const void * weights, const void * q8,
                                  float * out, uint32_t rows, cudaStream_t stream) {
  // The llama MMVQ template is compiled for one destination column and four
  // warps. One block owns one vocabulary row, matching grid=151936, block=32x4.
  ggml_cuda_mm_fusion_args_device fusion{};
  mul_mat_vec_q_switch_type(weights, GGML_TYPE_Q6_K, q8, nullptr, fusion, out,
    4096, rows, 1, 16, 128, rows, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);
}
