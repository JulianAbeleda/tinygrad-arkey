"""Structural contract for the semantic attention primitive.

This is intentionally not a fusion-performance test. It verifies the useful
pre-fusion invariant: score/probability intermediates are bounded by KV blocks,
not full T x KV tensors. Kernel-count and WMMA promotion remain separate gates.
"""
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, UOp
from tinygrad.schedule.indexing import _resolve_composite_axis_owner
from tinygrad.schedule.rangeify import lower_attention_semantic
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
import pytest


def _primitive_buffers(t: int):
  q = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  k = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  v = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  attention = shared_prefill_attention(q, k, v)
  assert attention.uop.op is Ops.ATTENTION
  assert attention.uop.arg.kv_block == 64
  return [u.shape for u in attention.uop.src[1].toposort() if u.op is Ops.BUFFER and u._shape is not None]


def test_semantic_attention_primitive_has_no_full_score_or_probability_buffer():
  # 129 crosses two complete blocks and one tail, so a full T x KV temporary
  # would be visible in the Tensor graph if block ownership were lost.
  t = 129
  buffers = _primitive_buffers(t)
  assert not [shape for shape in buffers if len(shape) >= 4 and shape[-2:] == (t, t)]

def test_composite_axis_owner_preserves_collapsed_source_axes():
  owners = (UOp.range(2, 10, AxisType.LOOP), UOp.const(dtypes.weakint, 0),
            UOp.range(4, 11, AxisType.REDUCE), UOp.range(8, 12, AxisType.LOOP))
  assert _resolve_composite_axis_owner(owners, 0) is owners[0]
  assert _resolve_composite_axis_owner(owners, 1) is None
  assert _resolve_composite_axis_owner(owners, 3) is owners[3]
  assert _resolve_composite_axis_owner(owners, 4) is None
  assert _resolve_composite_axis_owner(owners, -1) is None

def test_native_gqa_prefill_keeps_original_kv_and_carries_typed_grid():
  q = Tensor.empty(1, 32, 512, 128, dtype=dtypes.float16)
  k = Tensor.empty(1, 8, 512, 128, dtype=dtypes.float16)
  v = Tensor.empty(1, 8, 512, 128, dtype=dtypes.float16)
  attention = shared_prefill_attention(q, k, v)
  assert attention.uop.op is Ops.ATTENTION
  grid = attention.uop.arg.attention_grid
  assert grid is not None and (grid.q_heads, grid.kv_heads, grid.group_ratio) == (32, 8, 4)
  assert attention.uop.src[2].shape == q.shape and attention.uop.src[3].shape == k.shape and attention.uop.src[4].shape == v.shape
  assert not any(u.op is Ops.BUFFER and u.shape == (1, 32, 512, 128) for u in attention.uop.src[3].toposort())
  lowered = lower_attention_semantic(attention.uop)
  reductions = [u for u in lowered.toposort() if u.op is Ops.REDUCE and hasattr(u.arg[0], "attention_grid")]
  assert len(reductions) == 1 and reductions[0].arg[0].attention_grid == grid

def test_native_gqa_prefill_accepts_both_model_geometries_without_score_buffer():
  for hq in (32, 40):
    q = Tensor.empty(1, hq, 512, 128, dtype=dtypes.float16)
    k = Tensor.empty(1, 8, 512, 128, dtype=dtypes.float16)
    v = Tensor.empty(1, 8, 512, 128, dtype=dtypes.float16)
    lowered = lower_attention_semantic(shared_prefill_attention(q, k, v).uop)
    assert sum(u.op is Ops.REDUCE and hasattr(u.arg[0], "attention_grid") for u in lowered.toposort()) == 1
    assert not any(u.op is Ops.BUFFER and u.shape == (1, hq, 512, 512) for u in lowered.toposort())

@pytest.mark.parametrize("hq", (32, 40))
def test_native_gqa_prefill_semantic_owner_reaches_one_grid_wmma_body(hq):
  """The shared semantic owner, not a hand-built test kernel, admits M10d."""
  from dataclasses import replace
  from tinygrad.codegen import full_rewrite_to_sink, to_program
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  q = Tensor.empty(1, hq, 512, 128, dtype=dtypes.float16, device="AMD")
  k = Tensor.empty(1, 8, 512, 128, dtype=dtypes.float16, device="AMD")
  v = Tensor.empty(1, 8, 512, 128, dtype=dtypes.float16, device="AMD")
  calls = shared_prefill_attention(q, k, v).schedule_linear().src
  assert len(calls) == 1
  ast = calls[0].src[0]
  params = sorted((u.arg.slot, u.ptrdtype.size) for u in ast.toposort() if u.op is Ops.PARAM)
  assert params == [(0, hq*512*128), (1, hq*512*128), (2, 8*512*128), (3, 8*512*128)]
  from tinygrad.uop.ops import KernelInfo
  assert isinstance(ast.arg, KernelInfo)
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (0, 0, 1)),)))
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  final = full_rewrite_to_sink(ast, ren, optimize=True)
  nodes = final.toposort()
  assert sum(u.op is Ops.WMMA for u in nodes) == 16
  assert sum(u.op is Ops.RANGE for u in nodes) == 1
  assert not any(u.op is Ops.EXPAND for u in nodes)
  program = to_program(ast, ren)
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  mnemonics = [str(u.arg).split("(", 1)[0] for u in linear.src if not isinstance(u.arg, tuple)]
  assert mnemonics.count("v_wmma_f32_16x16x16_f16") == 16
  assert mnemonics.count("s_barrier") == 1
