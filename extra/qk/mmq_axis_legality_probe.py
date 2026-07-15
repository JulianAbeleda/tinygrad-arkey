"""Minimal generated-MMQ axis legality probe.

The important rule is that logical launch axes are ordinary integer-ID
``GLOBAL``/``LOCAL`` ranges.  The reduction range is a distinct ``REDUCE``
range and never participates in the output index.  This is intentionally a
compile-only probe: it has no owner gate, quant decode, or route behavior.
"""
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, ProgramInfo


def build_mmq_axis_probe() -> UOp:
  out = UOp.param(0, dtypes.float.ptr(8))
  # Integer IDs are required: postrange.py orders ranges by (AxisType, ID).
  global_m = UOp.range(2, 0, AxisType.GLOBAL)
  local_n = UOp.range(4, 1, AxisType.LOCAL)
  reduce_k = UOp.range(3, 2, AxisType.REDUCE)
  value = global_m.cast(dtypes.float) + local_n.cast(dtypes.float) + reduce_k.cast(dtypes.float)
  return out.index(global_m * 4 + local_n, ptr=True).store(value).sink(
    arg=KernelInfo(name="mmq_axis_legality_probe", opts_to_apply=()))


def compile_mmq_axis_probe() -> ProgramInfo:
  program = to_program(build_mmq_axis_probe(), PythonRenderer(Target("PYTHON")))
  return program.arg


__all__ = ["build_mmq_axis_probe", "compile_mmq_axis_probe"]
