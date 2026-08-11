"""Hermetic CPU tests for the residual-family fp16-store absorption (M2a).

The fused w1+w3 decode GEMV stores its result in fp32; the graph then renders
the ordinary E_128_32_3 ffn-activation cast (fp32 -> fp16) before the ffn_down
GEMV consumes it. store_fp16=True renders a distinct
q4k_g3_lanemap_gemv_w1w3fused16_* kernel that stores the same value already
cast to fp16, so the consumer's cast folds away. The in-kernel cast is the same
round-to-nearest-even fp32->fp16 conversion, so the stored bytes are
bitwise-identical. The legacy fp32 kernel name and route are unchanged when
the lease is absent.
"""
import hashlib
import re

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.decode_routes import q4k_gate_up_primitive_linear_call
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance,
  OutputSpec, execute_promoted_program)
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.decode.route_class_numerics import _make_q4k_words


ROWS, K = 32, 1024


def _digest(t: Tensor) -> str:
  return hashlib.sha256(np.ascontiguousarray(t.numpy()).view(np.uint8)).hexdigest()


def _words_tensor(words: np.ndarray) -> Tensor:
  return Tensor(words.copy(), dtype=dtypes.uint32, device="CPU").contiguous().realize()


def test_w1w3_fp16_store_kernel_name():
  scalar = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar")
  fp16 = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=True)
  out = UOp.placeholder((ROWS,), dtypes.float32, 0)
  gw = UOp.placeholder((ROWS * (K // 256) * 36,), dtypes.uint32, 1)
  uw = UOp.placeholder((ROWS * (K // 256) * 36,), dtypes.uint32, 2)
  x = UOp.placeholder((K,), dtypes.float16, 3)
  assert scalar(out, gw, uw, x).arg.name == f"q4k_g3_lanemap_gemv_w1w3fused_{ROWS}_{K}"
  assert fp16(out, gw, uw, x).arg.name == f"q4k_g3_lanemap_gemv_w1w3fused16_{ROWS}_{K}"


def test_w1w3_fp16_store_rejects_quad_style():
  with pytest.raises(ValueError, match="store_fp16"):
    q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="quad", store_fp16=True)


def test_w1w3_fp16_store_renders_through_cuda():
  for store_fp16 in (False, True):
    kernel = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=store_fp16)
    out = UOp.placeholder((ROWS,), dtypes.float16 if store_fp16 else dtypes.float32, 0)
    gw = UOp.placeholder((ROWS * (K // 256) * 36,), dtypes.uint32, 1)
    uw = UOp.placeholder((ROWS * (K // 256) * 36,), dtypes.uint32, 2)
    x = UOp.placeholder((K,), dtypes.float16, 3)
    ast = kernel(out, gw, uw, x)
    src = next(u.arg for u in to_program(ast, CUDARenderer(Target("NV", arch="sm_120"))).src
               if u.op is Ops.SOURCE)
    assert "__shfl_xor_sync" in src


def test_fp16_store_renders_the_same_value_with_a_half_cast():
  # The fp16 variant wraps the fp32 store expression in exactly one half cast:
  # the same round-to-nearest-even conversion the separate E_128_32_3 kernel
  # applies, so the stored bytes are bitwise-identical (verified on NV by the
  # AB harness logits gate; the CPU renderer cannot execute gidx0 kernels).
  def store_line(store_fp16: bool) -> str:
    kernel = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=store_fp16)
    out = UOp.placeholder((ROWS,), dtypes.float16 if store_fp16 else dtypes.float32, 0)
    gw = UOp.placeholder((ROWS * (K // 256) * 36,), dtypes.uint32, 1)
    uw = UOp.placeholder((ROWS * (K // 256) * 36,), dtypes.uint32, 2)
    x = UOp.placeholder((K,), dtypes.float16, 3)
    ast = kernel(out, gw, uw, x)
    src = next(u.arg for u in to_program(ast, CUDARenderer(Target("NV", arch="sm_120"))).src
               if u.op is Ops.SOURCE)
    return next(line for line in src.splitlines() if "data0" in line and "=" in line)

  fp32_line = store_line(False)
  fp16_line = store_line(True)
  assert "((half)" in fp16_line
  assert "((half)" not in fp32_line
  # The cast is a pure wrapper: strip it and the value expression is identical.
  fp16_expr = re.match(r".*\(\(half\)\(\((.*)\)\)\);$", fp16_line).group(1)
  fp32_expr = re.match(r".*= \((.*)\);$", fp32_line).group(1)
  assert fp16_expr == fp32_expr


def test_fp16_store_folds_the_consumer_cast_in_cpu_schedule():
  def schedule_names(store_fp16: bool) -> list[str]:
    words_t = _words_tensor(np.zeros((ROWS * (K // 256) * 36,), dtype=np.uint32))
    hidden = Tensor.empty((K,), dtype=dtypes.float16, device="CPU")
    producer = KernelProgram("cpu_topology", "w1w3", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=store_fp16),
      output_spec=OutputSpec((ROWS,), dtypes.float16 if store_fp16 else dtypes.float32))
    z = execute_promoted_program(None, words_t, words_t, hidden, program=producer)
    consumer = KernelProgram("cpu_topology", "down", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
      output_spec=OutputSpec((1,), dtypes.float32))
    return [call.src[0].arg.name for call in
            execute_promoted_program(None, words_t, words_t, z.cast(dtypes.float16).contiguous(), program=consumer).schedule_linear().src]

  control = schedule_names(False)
  candidate = schedule_names(True)
  # The fp32 arm materializes the consumer's fp32->fp16 cast as its own kernel
  # (the CPU renderer's generic materialization is named "test"); the fp16
  # store folds it away entirely.
  assert len(control) == 3 and control[0] == f"q4k_g3_lanemap_gemv_w1w3fused_{ROWS}_{K}"
  assert len(candidate) == 2 and candidate[0] == f"q4k_g3_lanemap_gemv_w1w3fused16_{ROWS}_{K}"


class _FakeQ4K:
  def __init__(self, words: np.ndarray):
    self.route_admission = QKPrimitiveRouteAdmission(
      QKPrimitiveCapability("NV", "sm_120", 32, True), True, q4k_w1w3_fusion_promoted=True)
    self.bias, self.decode_enabled = None, True
    self.out_features, self.in_features = ROWS, K
    self.q4k_storage = type("S", (), {"mode": "sidecar", "words": _words_tensor(words)})()


def _call_names(t: Tensor) -> list[str]:
  names = []
  for call in t.schedule_linear().src:
    arg = call.src[0].arg
    if arg is not None and getattr(arg, "name", None) is not None:
      names.append(arg.name)
  return names


def test_gate_up_call_default_route_keeps_fp32_output_and_name():
  words, _ = _make_q4k_words(ROWS, K, seed=20260812)
  gate, up = _FakeQ4K(words), _FakeQ4K(words)
  x = Tensor(np.random.default_rng(1).normal(0, 0.5, (1, 1, K)).astype(np.float16), device="CPU")
  z = q4k_gate_up_primitive_linear_call(gate, up, x, fallback=lambda: x)
  names = _call_names(z)
  assert z.dtype == dtypes.float32
  assert f"q4k_g3_lanemap_gemv_w1w3fused_{ROWS}_{K}" in names
  assert not any("fused16" in name for name in names)


def test_gate_up_call_fp16_store_emits_fused16_output():
  words, _ = _make_q4k_words(ROWS, K, seed=20260813)
  gate, up = _FakeQ4K(words), _FakeQ4K(words)
  x = Tensor(np.random.default_rng(2).normal(0, 0.5, (1, 1, K)).astype(np.float16), device="CPU")
  z = q4k_gate_up_primitive_linear_call(gate, up, x, fallback=lambda: x, store_fp16=True)
  names = _call_names(z)
  assert z.dtype == dtypes.float16
  assert f"q4k_g3_lanemap_gemv_w1w3fused16_{ROWS}_{K}" in names
  assert not any("w1w3fused_" in name and "16" not in name for name in names)
