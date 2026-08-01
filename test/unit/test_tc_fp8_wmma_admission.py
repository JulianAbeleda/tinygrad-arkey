"""Fail-closed admission for fp8 WMMA descriptors on renderers without native fp8.

`tinygrad/codegen/opt/postrange.py::_apply_generic_tensor_core_opt` now refuses a tensor-core
descriptor whose operand dtype the renderer cannot express natively, instead of letting the
dtype decomposer emulate it (fp8 -> half) after the TC opt. Emulated operands are incompatible
with WMMA render paths, and the two historical failure modes were:
  - K=128 fp8 on gfx942: the CDNA string pattern called `cstyle.py:fp8_index` on the emulated
    half operand and raised a bare `ValueError: tuple.index(x): x not in tuple`;
  - K=32 fp8 on gfx942 (`amd_cdna3` does bind those descriptors): rendered silently with half
    operands handed to an fp8 builtin -- the same mismatch, without the crash.

`HIPRenderer.supported_dtypes` admits fp8_ocp only on gfx950; gfx942 (MI300) emulates it. CUDA
sm_89 and gfx950 are fp8-native and must be unaffected. The guard raises `KernelOptError`
naming the capability at TC selection. This module FAILS on the pre-guard code (crash / silent
emission) and PASSES after.
"""
from __future__ import annotations
from dataclasses import replace

import pytest

import itertools

from tinygrad import Tensor, dtypes
from tinygrad.codegen import (apply_opts, full_rewrite_to_sink, pm_flatten_range, pm_load_collapse,
                              pm_mops, pm_simplify_ranges, pm_split_ranges, pm_store_ranges,
                              pm_syntactic_sugar, sym)
from tinygrad.codegen.opt import KernelOptError, Opt, OptOps, tc
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops, UOp, graph_rewrite


def _fp8_matmul_ast(dims: tuple[int, int, int], dtype_in, dtype_out, tc_select: int) -> UOp:
  m, n, k = dims[1], dims[0], dims[2]
  a = Tensor.empty(m, k, dtype=dtype_in)
  b = Tensor.empty(k, n, dtype=dtype_in)
  lin = a.dot(b, dtype=dtype_out).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  return ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (tc_select, 0, 1)),)))


def _tc_window(ast: UOp, ren) -> UOp:
  """Replay `_full_rewrite_to_sink` up to the TC opt, where WMMA vec carriers still exist (the
  devectorizer later scalarizes them). Same window the FC13 control asserts on."""
  sink = graph_rewrite(ast, pm_mops + pm_syntactic_sugar + pm_store_ranges, ctx=itertools.count(1000),
                       name="early movement ops", bottom_up=True)
  sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")
  sink = graph_rewrite(sink, pm_split_ranges + pm_flatten_range, ctx={}, name="split ranges")
  sink = graph_rewrite(sink, sym + pm_flatten_range, name="initial symbolic")
  sink = graph_rewrite(sink, pm_flatten_range + pm_simplify_ranges, ctx={}, name="simplify ranges")
  return apply_opts(sink, ren)


def _wmma(fs: UOp) -> UOp:
  wmmas = [u for u in fs.toposort() if u.op is Ops.WMMA]
  assert len(wmmas) == 1, f"expected exactly one WMMA, got {len(wmmas)}"
  return wmmas[0]


def test_gfx942_fp8_wmma_fails_closed_with_clear_message():
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx942"))
  ren.tensor_cores = tc.amd_cdna3
  ast = _fp8_matmul_ast(tc.amd_cdna_161632[0].dims, dtypes.fp8e5m2, dtypes.float, 0)
  with pytest.raises(KernelOptError, match="cannot be emitted"):
    full_rewrite_to_sink(ast, ren, optimize=True)


def test_gfx942_fp8_search_mode_fails_closed_not_no_tc():
  # Search mode must report the capability, not the generic "no tensor core available".
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx942"))
  ren.tensor_cores = tc.amd_cdna3
  ast = _fp8_matmul_ast(tc.amd_cdna_161632[0].dims, dtypes.fp8e5m2, dtypes.float, -1)
  with pytest.raises(KernelOptError, match="does not support"):
    full_rewrite_to_sink(ast, ren, optimize=True)


def test_gfx942_half_wmma_unaffected():
  # The guard must not over-block: the half descriptors amd_cdna3 binds on gfx942 still lower.
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx942"))
  ren.tensor_cores = tc.amd_cdna3
  tcx = tc.amd_cdna_161616[0]  # half:float 16x16x16, present in amd_cdna3
  ast = _fp8_matmul_ast(tcx.dims, dtypes.half, dtypes.float, 2)
  fs = _tc_window(ast, ren)
  w = _wmma(fs)
  assert w.dtype == dtypes.float.vec(tcx.elements_per_thread[2])
  assert w.src[0].dtype == dtypes.half.vec(tcx.elements_per_thread[0])


def test_gfx950_fp8_wmma_renders_native():
  # gfx950 has fp8_ocp natively: the K=128 descriptor must lower with fp8 carriers intact.
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx950"))
  ren.tensor_cores = tc.amd_cdna_1616128
  tcx = tc.amd_cdna_1616128[0]  # fp8e5m2:float 16x16x128
  ast = _fp8_matmul_ast(tcx.dims, dtypes.fp8e5m2, dtypes.float, 0)
  fs = _tc_window(ast, ren)
  w = _wmma(fs)
  assert w.dtype == dtypes.float.vec(tcx.elements_per_thread[2])
  assert w.src[0].dtype == dtypes.fp8e5m2.vec(tcx.elements_per_thread[0])
