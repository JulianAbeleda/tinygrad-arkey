"""Static topology contract for the boundary-free ordinary-UOp phase gate."""
from extra.llm_research.decode.nv_boundary_free_ordinary_uop_gate import run

def test_ordinary_rmsnorm_has_same_two_program_boundary_for_realized_and_lazy_input():
  got = run()
  assert got["verdict"] == "CONSTRUCTION_GAP"
  for row in got["baseline"].values():
    assert row["program_count"] == 2
    assert row["contains_custom_kernel"] is False
    assert row["contains_contiguous"] is False
