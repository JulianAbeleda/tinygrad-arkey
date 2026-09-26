"""ADD-REDUCE(CAST(a*b)) with a lossless widening CAST forms the products in the accumulation dtype.

`x.dot(w, dtype=dtypes.float)` on bf16 operands used to round every product to bf16 before the fp32 sum on the
scalar (non-tensor-core) lowering (1.8e-3 relative vs fp64), while the tensor-core lowering of the same graph is
bf16 x bf16 -> fp32 exact. pm_widen_reduce_products (codegen/late/reduce_lowering.py) makes the scalar path agree;
it runs after apply_opts, so tensor-core kernels are untouched.
"""
from __future__ import annotations
from dataclasses import replace
import unittest.mock

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
import tinygrad.codegen as codegen
from tinygrad.codegen import to_program
from tinygrad.codegen.late.reduce_lowering import widen_reduce_products
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, PatternMatcher, UOp


def _operands(dtype, m=16, k=256, n=8, seed=0):
  rng = np.random.default_rng(seed)
  a = Tensor(rng.standard_normal((m, k)).astype(np.float32), device="CPU").cast(dtype).realize()
  b = Tensor(rng.standard_normal((k, n)).astype(np.float32), device="CPU").cast(dtype).realize()
  return a, b, a.float().numpy().astype(np.float64) @ b.float().numpy().astype(np.float64)


def _rel(out, ref): return float(np.abs(out.astype(np.float64) - ref).max() / np.abs(ref).max())


@pytest.mark.parametrize("dtype", [dtypes.bfloat16, dtypes.half])
def test_narrow_dot_with_float_acc_matches_fp64(dtype):
  a, b, ref = _operands(dtype)
  # rounded products were 1.8e-3 (bf16) / 2e-4 (half) relative; exact products leave only fp32 summation error
  assert _rel(a.matmul(b, dtype=dtypes.float).numpy(), ref) < 1e-6
  assert _rel(a.dot(b, dtype=dtypes.float).numpy(), ref) < 1e-6
  assert np.array_equal(a.matmul(b, dtype=dtypes.float).numpy(), (a.float() @ b.float()).numpy())


def test_int8_dot_with_int32_acc_is_exact():
  rng = np.random.default_rng(1)
  a_np, b_np = rng.integers(-128, 128, (8, 64), dtype=np.int8), rng.integers(-128, 128, (64, 4), dtype=np.int8)
  out = Tensor(a_np, device="CPU").matmul(Tensor(b_np, device="CPU"), dtype=dtypes.int32).numpy()
  np.testing.assert_array_equal(out, a_np.astype(np.int32) @ b_np.astype(np.int32))


@pytest.mark.parametrize("dtype,acc", [(dtypes.float, None), (dtypes.float, dtypes.float), (dtypes.int32, None)])
def test_same_dtype_matmul_unaffected(dtype, acc):
  ast = _gemm_ast(dtype, acc)
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))
  assert _source(ast, disable_rule=False) == _source(ast, disable_rule=True)


def test_rule_declines_non_widening_shapes():
  x = UOp.variable("x", 0, 10).cast(dtypes.bfloat16)
  rng = UOp.range(4, 0)
  mul = x * x
  red = lambda src, op=Ops.ADD: UOp(Ops.REDUCE, src.dtype, (src, rng), (op,))
  assert widen_reduce_products(red(mul.cast(dtypes.float)), mul.cast(dtypes.float), mul) is not None
  assert widen_reduce_products(red(mul.cast(dtypes.float), Ops.MAX), mul.cast(dtypes.float), mul) is None
  assert widen_reduce_products(red(mul.cast(dtypes.half)), mul.cast(dtypes.half), mul) is None  # not lossless
  fmul = x.cast(dtypes.float) * x.cast(dtypes.float)
  assert widen_reduce_products(red(fmul.cast(dtypes.bfloat16)), fmul.cast(dtypes.bfloat16), fmul) is None  # narrowing


def _gemm_ast(in_dtype, dtype):
  a = Tensor.empty(256, 256, dtype=in_dtype, device="NV")
  b = Tensor.empty(256, 256, dtype=in_dtype, device="NV")
  linear = a.matmul(b, dtype=dtype).schedule_linear()
  calls = [c for c in linear.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
  assert len(calls) == 1
  return calls[0].src[0]


def _source(ast, disable_rule:bool):
  ren = CUDARenderer(Target.parse("NV:CUDA:sm_120"))
  codegen.to_program_cache.clear()
  with unittest.mock.patch.object(type(ren.compiler), "compile", lambda self, src: b""):
    if disable_rule:
      with unittest.mock.patch.object(codegen, "pm_widen_reduce_products", PatternMatcher([])): prog = to_program(ast, ren)
    else: prog = to_program(ast, ren)
  return next(u.arg for u in prog.src if u.op is Ops.SOURCE and isinstance(u.arg, str))


@pytest.mark.parametrize("dtype", [dtypes.float, None])
def test_tensor_core_kernel_is_bit_identical(dtype):
  ast = _gemm_ast(dtypes.bfloat16, dtype)
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (-1, 2, 1)),)))
  src = _source(ast, disable_rule=False)
  assert "mma" in src.lower() or "wmma" in src.lower()
  assert src == _source(ast, disable_rule=True)


def test_scalar_kernel_widens_operands_not_product():
  ast = _gemm_ast(dtypes.bfloat16, dtypes.float)
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))
  widened, rounded = _source(ast, disable_rule=False), _source(ast, disable_rule=True)
  assert widened != rounded and "(float)(val0))*((float)(val1))" in widened
