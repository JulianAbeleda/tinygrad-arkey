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


def test_a_partial_q6k_spec_lowers_through_the_route_family():
  """A partial binding (parts > 1) on a target whose coop tile is 2 builds a spec with its own row_tile; the
  route candidate must name that spec's tile, or the lowering refuses it as drift."""
  from tinygrad.llm.decode_kernels import q6k_spec_for_role
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  spec = q6k_spec_for_role(1024, 2560, parts=4, row_tile=2, use_coop=False, reduction="external_sum")
  cand = {"rows": 1024, "k": 2560, "row_tile": spec.row_tile, "reduction": "external_sum", "epilogue": "", "spec_binding": "q6_spec"}
  assert callable(build_registered_llm_emitter("q6_gemv_route.v1", cand, {"q6_spec": spec}))
