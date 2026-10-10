"""Every Q6_K decode GEMV weight gets a primitive route entry, whatever its role; attn_v keeps its measured 4 parts."""
from tinygrad.llm.model_route_plan import primitive_route_entry_for_tensor


def _entry(name, rows, cols):
  return primitive_route_entry_for_tensor(name, 14, rows, cols)


def test_q6k_entries_are_role_agnostic():
  for name in ("blk.3.attn_output.weight", "blk.0.ssm_out.weight", "blk.1.ffn_up.weight", "blk.2.attn_q.weight",
               "output.weight", "blk.0.ffn_down.weight"):
    e = _entry(name, 5120, 6144)
    assert e is not None and (e.parts, e.opts) == (1, ("LOCAL:0:64",)), name
  assert _entry("blk.0.attn_v.weight", 1024, 4096).parts == 4


def test_q6k_entries_refuse_partial_blocks():
  assert _entry("blk.3.attn_output.weight", 5120, 6100) is None
