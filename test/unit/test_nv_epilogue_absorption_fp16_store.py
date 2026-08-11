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
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance,
  OutputSpec, TypedLayout, TypedViewRequest, execute_promoted_program, _validated_typed_view)
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
    typed_output = (DeclaredTypedOutput(TypedLayout(dtypes.float16, (ROWS,), (1, 1, ROWS)),
                                        combine_fusion_admitted=False, epilogue_absorption_admitted=True)
                    if store_fp16 else None)
    producer = KernelProgram("cpu_topology", "w1w3", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=store_fp16),
      output_spec=OutputSpec((ROWS,), dtypes.float16 if store_fp16 else dtypes.float32,
                             typed_output=typed_output))
    z = execute_promoted_program(None, words_t, words_t, hidden, program=producer)
    consumer = KernelProgram("decode_q4k_cpu_topology", "down.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
      output_spec=OutputSpec((1,), dtypes.float32),
      typed_input_views=(TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(ROWS,),
                                          route_role="ffn_down", requires_combine_fusion=False,
                                          requires_epilogue_absorption=True),))
    return [call.src[0].arg.name for call in
            execute_promoted_program(None, words_t, words_t, z.cast(dtypes.float16).contiguous(), program=consumer).schedule_linear().src]

  control = schedule_names(False)
  candidate = schedule_names(True)
  # The fp32 arm materializes the consumer's fp32->fp16 cast as its own kernel
  # (the CPU renderer's generic materialization is named "test"); under the fp16
  # store lease the producer declares its typed layout and the consumer's typed
  # view request folds, so both the cast and the boundary copy disappear.
  assert len(control) == 3 and control[0] == f"q4k_g3_lanemap_gemv_w1w3fused_{ROWS}_{K}"
  assert len(candidate) == 2 and candidate[0] == f"q4k_g3_lanemap_gemv_w1w3fused16_{ROWS}_{K}"


def _typed_producer(declared: bool, admitted: bool = True) -> tuple[Tensor, KernelProgram]:
  words_t = _words_tensor(np.zeros((ROWS * (K // 256) * 36,), dtype=np.uint32))
  hidden = Tensor.empty((K,), dtype=dtypes.float16, device="CPU")
  typed_output = (DeclaredTypedOutput(TypedLayout(dtypes.float16, (ROWS,), (1, 1, ROWS)),
                                      combine_fusion_admitted=False,
                                      epilogue_absorption_admitted=admitted)
                  if declared else None)
  producer = KernelProgram("cpu_topology", "w1w3", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=True),
    output_spec=OutputSpec((ROWS,), dtypes.float16, typed_output=typed_output))
  return execute_promoted_program(None, words_t, words_t, hidden, program=producer), producer


def _ffn_down_request() -> TypedViewRequest:
  return TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(ROWS,), route_role="ffn_down",
                          requires_combine_fusion=False, requires_epilogue_absorption=True)


def test_epilogue_absorption_folds_only_against_a_declared_producer():
  z, producer = _typed_producer(declared=True)
  request = _ffn_down_request()
  consumer = KernelProgram("decode_q4k_cpu_topology", "down.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
    output_spec=OutputSpec((1,), dtypes.float32), typed_input_views=(request,))
  view, reason = _validated_typed_view(z.cast(dtypes.float16).contiguous().uop, request, consumer)
  assert view is not None and reason == "ok"


def test_epilogue_absorption_is_fail_closed_without_a_declaration():
  z, _ = _typed_producer(declared=False)
  request = _ffn_down_request()
  consumer = KernelProgram("decode_q4k_cpu_topology", "down.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
    output_spec=OutputSpec((1,), dtypes.float32), typed_input_views=(request,))
  view, reason = _validated_typed_view(z.cast(dtypes.float16).contiguous().uop, request, consumer)
  assert view is None and "producer declared no typed output layout" in reason


def test_epilogue_absorption_gate_closed_rejects():
  z, _ = _typed_producer(declared=True, admitted=False)
  request = _ffn_down_request()
  consumer = KernelProgram("decode_q4k_cpu_topology", "down.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
    output_spec=OutputSpec((1,), dtypes.float32), typed_input_views=(request,))
  view, reason = _validated_typed_view(z.cast(dtypes.float16).contiguous().uop, request, consumer)
  assert view is None and "epilogue-absorption gate is closed" in reason


def test_epilogue_absorption_wrong_consumer_role_or_program_rejects():
  z, _ = _typed_producer(declared=True)
  consumer = KernelProgram("decode_q4k_cpu_topology", "down.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
    output_spec=OutputSpec((1,), dtypes.float32),
    typed_input_views=(TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(ROWS,), route_role="attn_qo",
                                        requires_combine_fusion=False, requires_epilogue_absorption=True),))
  view, reason = _validated_typed_view(z.cast(dtypes.float16).contiguous().uop,
                                       consumer.typed_input_views[0], consumer)
  assert view is None and "wrong consumer route_role" in reason
  consumer2 = KernelProgram("decode_flash_other", "down", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
    output_spec=OutputSpec((1,), dtypes.float32), typed_input_views=(_ffn_down_request(),))
  view2, reason2 = _validated_typed_view(z.cast(dtypes.float16).contiguous().uop,
                                         consumer2.typed_input_views[0], consumer2)
  assert view2 is None and "program is not a q4k/q6k GEMV consumer" in reason2


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


class _FakeQ6K:
  def __init__(self, route_role: str | None):
    self.q6k_storage = type("S", (), {"halfs": Tensor.zeros(8, dtype=dtypes.uint16)})()
    self.decode_enabled = True
    self.bias = None
    self.in_features = K
    self.out_features = ROWS
    self.parts = 1
    self.opts = ()
    self.route_admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability("NV", "sm_120", 32, True), True)
    self.route_role = route_role


def test_q6k_ffn_down_gemv_carries_the_epilogue_absorption_request(monkeypatch):
  from tinygrad.llm import decode_routes
  linear = _FakeQ6K(route_role="ffn_down")
  x = Tensor.zeros((1, 1, K), dtype=dtypes.float16)
  binding = decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, x, True)
  assert binding is not None
  captured = []
  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  decode_routes.Q6K_DECODE_CANDIDATE.execute(linear, x, binding)
  assert len(captured) == 1
  views = captured[0].typed_input_views
  assert len(views) == 1
  req = views[0]
  assert req.slot == 1 and req.dtype is dtypes.float16 and req.flat_shape == (K,)
  assert req.route_role == "ffn_down"
  assert not req.requires_combine_fusion and req.requires_epilogue_absorption


def test_q6k_non_ffn_down_gemv_keeps_the_generic_flat_buffer_abi(monkeypatch):
  from tinygrad.llm import decode_routes
  linear = _FakeQ6K(route_role="attn_kv")
  x = Tensor.zeros((1, 1, K), dtype=dtypes.float16)
  binding = decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, x, True)
  assert binding is not None
  captured = []
  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  decode_routes.Q6K_DECODE_CANDIDATE.execute(linear, x, binding)
  assert len(captured) == 1
  assert captured[0].typed_input_views == ()
