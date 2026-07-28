"""LR-062: the custom-kernel boundary -- a spec must be able to name the buffers its kernel binds.

Before this, FlashDecodeAttentionSpec could name its emitted KERNELS (`emitted_kernel_names`) but not the
buffers those kernels take. The binding order lived in a comment in extra/llm_research/flash_kernels.py and was
re-derived by hand at every construction site. These tests make the declaration load-bearing: if the spec and
the builder disagree about arity, shape, or order, something here fails.
"""
from __future__ import annotations

import pytest

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.flash_decode_attention_spec import describe_flash_decode_attention


def _spec(**kw):
  base = dict(Hq=32, Hd=128, Hkv=8, MAXC=4096, S=4)
  base.update(kw)
  return describe_flash_decode_attention(**base)


def _bind(tile):
  """Construct placeholders straight from the declared contract -- no hand-written shapes."""
  return [UOp.placeholder(r.shape, getattr(dtypes, r.dtype), i) for i, r in enumerate(tile.buffer_roles)]


def test_the_declared_contract_is_enough_to_build_the_kernel():
  """The whole point: a caller that knows only the spec can bind correctly. If this needed one hand-written
  shape, the contract would not actually be a contract."""
  spec = _spec()
  fn = spec.emit_tile(UOp.const(dtypes.int32, 1024))
  sink = fn(*_bind(spec.tile))
  assert sink is not None and sink.key


def test_buffer_roles_match_the_builder_arity():
  import inspect
  from extra.llm_research.flash_kernels import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel as builder
  inner = inspect.signature(builder).parameters
  assert {"quant", "rope"} <= set(inner), "builder no longer takes the flags buffer_roles keys off"
  # 3 fixed buffers, plus one per enabled optional flag
  assert len(_spec().tile.buffer_roles) == 3
  assert len(_spec(quant=True).tile.buffer_roles) == 4


def test_optional_buffers_appear_only_when_their_flag_is_set_and_in_builder_order():
  """flash_kernels.py unpacks `[kvscale] if quant` BEFORE `[freqs] if rope`. A contract that listed them the
  other way round would bind a rope table where the dequant scale is expected."""
  assert [r.name for r in _spec().tile.buffer_roles] == ["pout", "q", "cache"]
  assert [r.name for r in _spec(quant=True).tile.buffer_roles] == ["pout", "q", "cache", "kvscale"]
  quant_rope = _spec(quant=True, rope=True).tile.buffer_roles
  names = [r.name for r in quant_rope]
  assert names.index("kvscale") < names.index("freqs"), "kvscale must precede freqs, matching the builder"


def test_optional_roles_declare_what_enables_them():
  roles = {r.name: r for r in _spec(quant=True, rope=True).tile.buffer_roles}
  assert roles["kvscale"].optional_on == "quant"
  assert roles["freqs"].optional_on == "rope"
  assert roles["cache"].optional_on is None


def test_the_builder_still_rejects_a_missing_optional_buffer():
  """The contract describes the binding; it must not become the only thing enforcing it."""
  spec = _spec(quant=True)
  fn = spec.emit_tile(UOp.const(dtypes.int32, 1024))
  fixed = [UOp.placeholder(r.shape, getattr(dtypes, r.dtype), i)
           for i, r in enumerate(spec.tile.buffer_roles) if r.optional_on is None]
  with pytest.raises(ValueError, match="scale buffer"):
    fn(*fixed)
