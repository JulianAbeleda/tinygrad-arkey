"""Regression test for the generic tensor-core opt crashing on reduce axes that pm_split_ranges split.

tinygrad/codegen/opt/postrange.py::_apply_generic_tensor_core_opt assumed every entry of
`reduceop.src[1:]` (the reduce's own range list) is a bare Ops.RANGE, and read `.arg[0]` off each one
to sort them:

    red_ranges = sorted(reduceop.src[1:], key=lambda x: x.arg[0], reverse=True)

tinygrad/codegen/simplify.py::pm_split_ranges (lines 72-75) breaks that assumption. It pattern-matches
`RANGE % CONST` anywhere in the sink and, when it fires on a reduce-owning range `k`, substitutes EVERY
occurrence of `k` -- including inside the owning REDUCE's own `src[1:]` -- with:

    k.replace(src=(k.src[0]//v,), arg=k.arg[:-1]+(0,k.arg[-1])) * v + k.replace(src=(v,), arg=k.arg[:-1]+(1,k.arg[-1]))

i.e. `ADD(MUL(RANGE_lo, CONST_v), RANGE_hi)`. That pass runs unconditionally in full_rewrite_to_sink,
before apply_opts / the TC-opt search ever runs (codegen/__init__.py:130). Q4_K's 256-element block
addressing (`k % block_elems` inside PackedWeightTransform.dequant) triggers this on the K axis, so
`reduceop.src[1:]` becomes a mix of RANGE and ADD nodes by the time the generic TC opt inspects it, and
`x.arg[0]` on an ADD node (whose `.arg` is None) raises `TypeError: 'NoneType' object is not
subscriptable`. Verified to crash identically on METAL and AMD (scratchpad/t1_generic_tc_dequant_probe.py
rung 2) before this fix.

The fix in postrange.py:
  1. `_split_range_axis` -- a recognizer for the exact `(k//v)*v + (k%v)` shape pm_split_ranges emits,
     derived directly from that rewrite (not pattern-guessed from an observed graph). Returns the
     recovered `k%v` (block-local) RANGE, or None.
  2. A sibling branch inside `_apply_generic_tensor_core_opt`, beside (not replacing) the original
     all-RANGE `red_ranges = sorted(reduceop.src[1:], ...)` line: when some entries aren't bare RANGE,
     it tries `_split_range_axis` on each one and declines (skips this tensor core, `continue`) if any
     entry is neither a RANGE nor that exact split shape -- converting the crash into either a
     successfully-recovered axis list or a clean `KernelOptError("no tensor core available")`.

This module tests the recognizer's fail-closed guards directly (using pm_split_ranges itself to
produce the ground-truth split shape, not a hand-guessed one), that the owner (all-RANGE) path is
unmodified and still applies TC normally, and that the full Q4_K-dequant-into-generic-TC pipeline no
longer raises TypeError on either backend. This FAILS (TypeError) on the pre-fix code and PASSES after.
"""
from __future__ import annotations
from dataclasses import replace

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
from tinygrad.codegen.opt.postrange import _split_range_axis
from tinygrad.codegen.simplify import pm_flatten_range, pm_split_ranges
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer, MetalRenderer
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite

TC_OPT = Opt(OptOps.TC, 0, (-1, 2, 1))
TARGETS = {
  "METAL": ("METAL:METAL:Apple9", MetalRenderer),
  "AMD": ("AMD:HIP:gfx1100", HIPRenderer),
}


# **** recognizer: built from the real pm_split_ranges rewrite, not a hand-guessed shape ****

def _real_split_add(size:int, div:int, range_id:int=5, axis:AxisType=AxisType.REDUCE) -> tuple[UOp, UOp]:
  """Run the actual pm_split_ranges rewrite on `RANGE(size) % div` inside a REDUCE's own range list and
  return (the resulting ADD node that replaces the RANGE inside reduceop.src[1:], the original RANGE)."""
  k = UOp.range(size, range_id, axis)
  body = (k % div).cast(dtypes.float) * UOp.const(dtypes.float, 1.0)
  red = body.reduce(k, arg=Ops.ADD)
  sink = UOp.sink(red)
  out = graph_rewrite(sink, pm_split_ranges + pm_flatten_range, ctx={}, name="test split ranges")
  reduces = [u for u in out.backward_slice if u.op is Ops.REDUCE]
  assert len(reduces) == 1
  adds = [x for x in reduces[0].src[1:] if x.op is Ops.ADD]
  assert len(adds) == 1, f"expected exactly one split ADD in reduceop.src[1:], got {adds}"
  return adds[0], k


def test_recognizer_recovers_the_mod_range_from_the_real_split():
  add, k = _real_split_add(size=256, div=16, range_id=7, axis=AxisType.REDUCE)
  hi = _split_range_axis(add)
  assert hi is not None and hi.op is Ops.RANGE
  assert hi.vmax + 1 == 16, "recovered range must be the k%v (block-local) half, sized to the divisor"
  assert hi.arg[0] == k.arg[0] == 7, "recovered range must keep the original range's id for sort stability"
  assert hi.arg[-1] is AxisType.REDUCE


def test_recognizer_declines_a_plain_range():
  assert _split_range_axis(UOp.range(8, 0, AxisType.REDUCE)) is None


def test_recognizer_declines_a_plain_add():
  a, b = UOp.range(8, 0, AxisType.REDUCE), UOp.range(4, 1, AxisType.REDUCE)
  assert _split_range_axis(a + b) is None


def test_recognizer_declines_wrong_mul_shape():
  # ADD(MUL(RANGE, RANGE), RANGE) -- the multiplier must be a CONST, not another RANGE
  lo, mid, hi = UOp.range(8, 0, AxisType.REDUCE), UOp.range(2, 1, AxisType.REDUCE), UOp.range(2, 2, AxisType.REDUCE)
  assert _split_range_axis(lo * mid + hi) is None


def test_recognizer_declines_mismatched_divisor():
  # hi.src[0] is a DIFFERENT const object/value than the MUL's multiplier
  add, _ = _real_split_add(size=256, div=16, range_id=9)
  mul, hi = add.src
  bad_hi = hi.replace(src=(UOp.const(hi.dtype, 999),))
  assert _split_range_axis(mul + bad_hi) is None


def test_recognizer_declines_mismatched_ids():
  add, _ = _real_split_add(size=256, div=16, range_id=11)
  mul, hi = add.src
  lo = mul.src[0]
  other_lo = lo.replace(arg=(999,) + lo.arg[1:])
  assert _split_range_axis(other_lo * mul.src[1] + hi) is None


def test_recognizer_declines_mismatched_axis_type():
  add, _ = _real_split_add(size=256, div=16, range_id=13)
  mul, hi = add.src
  bad_hi = hi.replace(arg=hi.arg[:-1] + (AxisType.LOOP,))
  assert _split_range_axis(mul + bad_hi) is None


def test_recognizer_declines_swapped_tags():
  # lo tagged 1 and hi tagged 0 (backwards) must not be accepted as a valid split
  add, _ = _real_split_add(size=256, div=16, range_id=17)
  mul, hi = add.src
  lo = mul.src[0]
  swapped_lo = lo.replace(arg=lo.arg[:-2] + (1, lo.arg[-1]))
  swapped_hi = hi.replace(arg=hi.arg[:-2] + (0, hi.arg[-1]))
  assert _split_range_axis(swapped_lo * mul.src[1] + swapped_hi) is None


def test_recognizer_accepts_either_operand_order():
  add, k = _real_split_add(size=256, div=16, range_id=19)
  mul, hi = add.src
  assert _split_range_axis(mul + hi) is not None
  assert _split_range_axis(hi + mul) is not None  # ADD is unordered from the recognizer's point of view


# **** integration: the actual Q4_K-dequant-into-generic-TC pipeline ****

def _dense_gemm_ast(device:str, m:int, n:int, k:int) -> UOp:
  a = Tensor.empty(m, k, dtype=dtypes.half, device=device)
  b = Tensor.empty(n, k, dtype=dtypes.half, device=device)
  linear = (a @ b.transpose()).schedule_linear()
  calls = [c for c in linear.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
  assert len(calls) == 1
  return calls[0].src[0]


def _packed_dequant_ast(device:str, m:int, n:int, k:int) -> UOp:
  """Same shape as _dense_gemm_ast, but the B operand's load is replaced with a real Q4_K dequant
  expression built from the reduce's own K range -- exactly scratchpad/t1_generic_tc_dequant_probe.py's
  rung 2, which is what pm_split_ranges' `k % block_elems` match fires on."""
  ast = _dense_gemm_ast(device, m, n, k)
  reduces = [u for u in ast.backward_slice if u.op is Ops.REDUCE]
  assert len(reduces) == 1
  red = reduces[0]
  mul = red.src[0] if red.src[0].op is not Ops.CAST else red.src[0].src[0]
  assert mul.op is Ops.MUL
  in0, in1 = mul.src
  k_rng = red.src[1]
  desc = PackedWeightTransform("Q4_K", rows=n, k=k)
  packed_words = desc.packed_bytes // desc.storage_width
  existing_slots = {u.arg.slot for u in ast.toposort() if u.op is Ops.PARAM}
  packed_slot = max(existing_slots) + 1
  n_candidates = [r for r in in1.ranges if r not in in0.ranges and r.arg[-1].name == "LOOP"]
  assert len(n_candidates) == 1
  source = UOp.placeholder((packed_words,), desc.storage_dtype, packed_slot)
  dequant_value = desc.dequant(source, row=n_candidates[0], k=k_rng)
  assert dequant_value.dtype == dtypes.float16
  return ast.substitute({in1: dequant_value})


def _force_generic_tc(ast:UOp) -> UOp:
  assert ast.arg.candidate_context is None
  return ast.replace(arg=replace(ast.arg, opts_to_apply=(TC_OPT,)))


def _to_program_no_native_compile(ast:UOp, renderer, device:str):
  """AMD cross-compile (amd_comgr) is native and unstable off-device (crashes the process outright, not
  a catchable Python exception) -- unrelated to this bug, which lives entirely upstream of it in
  apply_opts. Stub only the native compile call, exactly like scratchpad/t1_generic_tc_dequant_probe.py,
  so this test still exercises the real apply_opts/TC-opt run and gets real rendered source on a machine
  with no AMD GPU, without ever touching the crashing native compile path."""
  if device != "AMD": return to_program(ast, renderer)
  import unittest.mock
  with unittest.mock.patch.object(type(renderer.compiler), "compile", lambda self, src: b""):
    return to_program(ast, renderer)


@pytest.mark.parametrize("device", ["METAL", "AMD"])
def test_owner_path_unmodified_dense_gemm_still_gets_wmma(device):
  """The all-RANGE (unsplit) path is the existing owner and must be untouched: a plain fp16 GEMM (no
  split ranges involved) must still lower to a real WMMA, exactly as before this fix."""
  target_str, renderer_cls = TARGETS[device]
  ast = _force_generic_tc(_dense_gemm_ast(device, 512, 512, 512))
  prog = _to_program_no_native_compile(ast, renderer_cls(Target.parse(target_str)), device)
  source = next((u.arg for u in prog.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
  assert source is not None and "__WMMA" in source


@pytest.mark.parametrize("device", ["METAL", "AMD"])
def test_q4k_dequant_generic_tc_no_longer_raises_typeerror(device):
  """The regression: this construction crashed with TypeError('NoneType' object is not subscriptable)
  on both backends before the fix. It must not crash anymore -- it may either apply the TC opt (and
  emit a WMMA) or decline cleanly with KernelOptError, but never TypeError."""
  target_str, renderer_cls = TARGETS[device]
  ast = _force_generic_tc(_packed_dequant_ast(device, 512, 12288, 4096))
  renderer = renderer_cls(Target.parse(target_str))
  from tinygrad.codegen.opt import KernelOptError
  try:
    _to_program_no_native_compile(ast, renderer, device)
  except KernelOptError:
    pass  # acceptable outcome: declines cleanly, not a crash
  except TypeError as exc:
    pytest.fail(f"generic TC opt crashed on a pm_split_ranges-produced axis (the bug): {exc}")
