import inspect
from tinygrad.uop.ops import AxisType, Ops
from extra.llm_research.prefill.bench_nv_q6_oracle_streamk_single_body_packed import _ast,_ast_proof,_schedule
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import q6_oracle_broad_cta_kernel

def test_streamk_schedule_is_complete_plane_major_and_bounded():
  slots,records,invariants=_schedule()
  assert all(invariants.values())
  assert all(1 <= len(x) <= 3 for x in slots)
  assert all(x == sorted(x) for x in slots)
  assert all(r["plane"] in (0,1) and 0 <= r["k_begin"] < r["k_end"] <= 48 for r in records)

def test_single_body_ast_has_nested_lifecycle_and_five_barriers():
  ast=_ast(None);proof=_ast_proof(ast)
  assert all(proof.values())
  ranges={x.arg[0]:x for x in ast.toposort() if x.op is Ops.RANGE and x.arg[0] in (1498,1499)}
  assert ranges[1498] in ranges[1499].src[0].ranges
  assert ranges[1498].arg[1] is AxisType.LOOP
  assert ranges[1499].arg[1] is AxisType.LOOP

def test_streamk_segments_in_cta_is_default_off():
  signature=inspect.signature(q6_oracle_broad_cta_kernel)
  assert signature.parameters["streamk_segments_in_cta"].default is False
