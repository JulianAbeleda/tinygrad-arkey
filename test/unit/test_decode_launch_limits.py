"""K8/K9: decode GEMV emitters refuse a block they cannot launch (threads per block, static shared memory)."""
import pytest
from tinygrad.llm.decode_kernels import (PORTABLE_STATIC_SHARED_BYTES, q4k_g3_lanemap_gemv_kernel,
  q4k_g3_lanemap_gemv_w1w3_kernel, validate_launch_limits)
from tinygrad.llm.q6k_v_mmvq import validate_four_warp


def test_four_warp_q6k_refuses_more_than_a_block_of_threads():
  validate_four_warp(1024, 4096, 4)
  validate_four_warp(64, 32 * 256 * 2, 32)  # 32 warps x 32 lanes = 1024 threads
  with pytest.raises(ValueError, match="threads per block"):
    validate_four_warp(64, 64 * 256 * 2, 64)  # 64 warps = 2048 threads: k divides, the launch would not


def test_quad_spellings_refuse_x_that_does_not_fit_shared_memory():
  fits = PORTABLE_STATIC_SHARED_BYTES // 2  # k fp16 halfs staged
  q4k_g3_lanemap_gemv_kernel(16, fits, load_style="quad")
  q4k_g3_lanemap_gemv_w1w3_kernel(16, 4096, load_style="quad")  # its lane map holds k at 4096
  too_big = fits + 1024  # e.g. a 17408-wide ffn_down
  with pytest.raises(ValueError, match="shared memory"): q4k_g3_lanemap_gemv_kernel(16, too_big, load_style="quad")
  q4k_g3_lanemap_gemv_kernel(16, too_big, load_style="quad", shared_max=too_big * 2)  # a device's own fact may admit more


def test_the_limit_helper():
  validate_launch_limits("k", 1024, PORTABLE_STATIC_SHARED_BYTES)
  with pytest.raises(ValueError): validate_launch_limits("k", 1025)
  with pytest.raises(ValueError): validate_launch_limits("k", 32, PORTABLE_STATIC_SHARED_BYTES + 1)
  validate_launch_limits("k", 32, 65536, shared_max=65536)
