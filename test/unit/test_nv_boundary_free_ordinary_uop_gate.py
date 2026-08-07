"""Static topology contract for the boundary-free ordinary-UOp phase gate."""
from extra.llm_research.decode.nv_boundary_free_ordinary_uop_gate import OPAQUE_ARMS, run
from extra.llm_research.decode.nv_fusion_population_ledger import (
  POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK, POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB,
)

VERDICTS = ("ORDINARY_PASS", "CONSTRUCTION_GAP", "OPAQUE_PRODUCER_GAP")

def test_ordinary_rmsnorm_has_same_two_program_boundary_for_realized_and_lazy_input():
  got = run()
  entry = got["populations"][POP_NORMS]
  assert entry["verdict"] == "CONSTRUCTION_GAP"
  for row in entry["baseline"].values():
    assert row["program_count"] == 2
    assert row["contains_custom_kernel"] is False
    assert row["contains_contiguous"] is False

def test_v3_envelope_covers_every_population_with_a_construction_or_no_construction():
  got = run()
  assert got["schema"] == "tinygrad.nv_boundary_free_ordinary_uop_gate.v3"
  assert set(got["populations"]) == {POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK,
                                     POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB}
  assert got["populations"][POP_OTHER]["verdict"] == "NO_CONSTRUCTION"
  assert got["populations"][POP_OTHER]["contract_shape"] == []
  for pop in (POP_NORMS, POP_FLASH, POP_RESIDUAL, POP_VOCAB, POP_ROPE_KV, POP_QUANT, POP_Q8PACK):
    entry = got["populations"][pop]
    assert entry["verdict"] in VERDICTS
    assert entry["contract_shape"]
    assert entry["reason"]

def test_every_population_declares_the_scope_its_verdict_covers():
  """A verdict without a stated scope is how v2's ORDINARY_PASS got read as 'unblocked'."""
  for entry in run()["populations"].values():
    assert entry["scope"]

def test_ordinary_pass_is_never_reported_as_a_bare_pass():
  """v2 emitted `PASS`, which reads as an upper bound. It is only a lower bound."""
  for entry in run()["populations"].values():
    assert entry["verdict"] != "PASS"

def test_residual_ordinary_arm_passes_but_real_producer_form_does_not():
  """The v2 false-PASS regression test.

  The residual family's ordinary stand-in uses a plain realized buffer, which the M4 probe
  documents as the trivially-folding control case. The production producer is the previous
  block's output across an @function(precompile=True) boundary. v2 reported PASS off the
  control case alone, contradicting the S4 GPU gate; v3 must report the gap."""
  entry = run(POP_RESIDUAL)["populations"][POP_RESIDUAL]
  assert entry["ordinary_verdict"] == "ORDINARY_PASS"
  assert entry["opaque_producer_verdict"] == "CONSTRUCTION_GAP"
  assert entry["verdict"] == "OPAQUE_PRODUCER_GAP"
  assert entry["opaque_producer"]["program_count"] > 1

def test_split_cause_is_not_conflated_across_mechanisms():
  """A lowered reduce and a precompiled FUNCTION boundary both split the program count.
  Reporting both as 'reduction + dependent epilogue' would merge C1 with the M4 blocker."""
  got = run()
  assert "reduction + dependent epilogue" in got["populations"][POP_NORMS]["reason"]
  assert "reduction + dependent epilogue" in got["populations"][POP_FLASH]["reason"]
  residual = got["populations"][POP_RESIDUAL]
  assert "precompiled FUNCTION producer boundary" in residual["opaque_producer_reason"]
  assert "reduction + dependent epilogue" not in residual["opaque_producer_reason"]

def test_norms_and_flash_share_one_blocking_mechanism():
  """C1: both gaps are reduce-then-consume, so one primitive gates both populations."""
  got = run()
  for pop in (POP_NORMS, POP_FLASH):
    assert got["populations"][pop]["verdict"] == "CONSTRUCTION_GAP"
    assert "reduction + dependent epilogue" in got["populations"][pop]["reason"]

def test_populations_without_an_opaque_arm_say_so_rather_than_implying_plain():
  """Absence from OPAQUE_ARMS means 'not assessed', and the scope string must say that."""
  got = run()
  for pop, entry in got["populations"].items():
    if pop in OPAQUE_ARMS or entry["verdict"] == "NO_CONSTRUCTION": continue
    assert "not assessed" in entry["scope"]
    assert "opaque_producer" not in entry
