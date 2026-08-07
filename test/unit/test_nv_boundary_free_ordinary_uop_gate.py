"""Static topology contract for the boundary-free ordinary-UOp phase gate."""
from extra.llm_research.decode.nv_boundary_free_ordinary_uop_gate import run
from extra.llm_research.decode.nv_fusion_population_ledger import (
  POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK, POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB,
)

def test_ordinary_rmsnorm_has_same_two_program_boundary_for_realized_and_lazy_input():
  got = run()
  entry = got["populations"][POP_NORMS]
  assert entry["verdict"] == "CONSTRUCTION_GAP"
  for row in entry["baseline"].values():
    assert row["program_count"] == 2
    assert row["contains_custom_kernel"] is False
    assert row["contains_contiguous"] is False

def test_v2_envelope_covers_every_population_with_a_construction_or_no_construction():
  got = run()
  assert got["schema"] == "tinygrad.nv_boundary_free_ordinary_uop_gate.v2"
  assert set(got["populations"]) == {POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK,
                                     POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB}
  assert got["populations"][POP_OTHER]["verdict"] == "NO_CONSTRUCTION"
  assert got["populations"][POP_OTHER]["contract_shape"] == []
  for pop in (POP_NORMS, POP_FLASH, POP_RESIDUAL, POP_VOCAB, POP_ROPE_KV, POP_QUANT, POP_Q8PACK):
    entry = got["populations"][pop]
    assert entry["verdict"] in ("PASS", "CONSTRUCTION_GAP")
    assert entry["contract_shape"]
    assert entry["reason"]
