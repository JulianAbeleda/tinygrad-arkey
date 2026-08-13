"""Static topology contract for the boundary-free ordinary-UOp phase gate."""
from extra.llm_research.decode.nv_boundary_free_ordinary_uop_gate import OPAQUE_ARMS, run, run_v3
from extra.llm_research.decode.nv_fusion_population_ledger import (
  POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK, POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB,
)

VERDICTS = ("ORDINARY_PASS", "CONSTRUCTION_GAP", "OPAQUE_PRODUCER_GAP", "REDUCE_OUTPUT_PASS")


def test_v4_envelope_covers_every_population_with_a_construction_or_no_construction():
  got = run()
  assert got["schema"] == "tinygrad.nv_boundary_free_ordinary_uop_gate.v4"
  assert set(got["populations"]) == {POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK,
                                     POP_QUANT, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB}
  assert got["populations"][POP_OTHER]["verdict"] == "NO_CONSTRUCTION"
  assert got["populations"][POP_OTHER]["contract_shape"] == []
  for pop in (POP_NORMS, POP_FLASH, POP_RESIDUAL, POP_VOCAB, POP_ROPE_KV, POP_QUANT, POP_Q8PACK):
    entry = got["populations"][pop]
    assert entry["verdict"] in VERDICTS
    assert entry["contract_shape"]
    assert entry["reason"]


def test_norms_reduce_output_arm_passes_realized_identity_input_and_records_lazy_fallback():
  entry = run(POP_NORMS)["populations"][POP_NORMS]
  assert entry["verdict"] == "REDUCE_OUTPUT_PASS"
  assert entry["ordinary_verdict"] == "CONSTRUCTION_GAP"
  assert entry["reduce_output"]["verdict"] == "ORDINARY_PASS"
  realized = entry["reduce_output"]["realized"]
  assert realized["program_count"] == 1
  assert realized["contains_custom_kernel"] is False
  assert realized["contains_contiguous"] is False
  # Non-identity inputs are deliberately out of scope for REDUCE_OUTPUT; v4 must
  # record the fallback instead of silently claiming the lazy producer folds.
  lazy = entry["reduce_output"]["lazy_add"]
  assert lazy["program_count"] == 2


def test_every_population_declares_the_scope_its_verdict_covers():
  """A verdict without a stated scope is how v2's ORDINARY_PASS got read as 'unblocked'."""
  for entry in run()["populations"].values():
    assert entry["scope"]


def test_pass_verdicts_are_never_reported_as_a_bare_pass():
  """v2 emitted `PASS`, which reads as an upper bound. Every pass is only a lower bound."""
  for entry in run()["populations"].values():
    assert entry["verdict"] != "PASS"


def test_residual_ordinary_arm_passes_but_real_producer_form_does_not():
  """The v2 false-PASS regression test.

  The residual family's ordinary stand-in uses a plain realized buffer, which the M4 probe
  documents as the trivially-folding control case. The production producer is the previous
  block's output across an @function(precompile=True) boundary. v2 reported PASS off the
  control case alone, contradicting the S4 GPU gate; the gate must report the gap."""
  entry = run(POP_RESIDUAL)["populations"][POP_RESIDUAL]
  assert entry["ordinary_verdict"] == "ORDINARY_PASS"
  assert entry["opaque_producer_verdict"] == "CONSTRUCTION_GAP"
  assert entry["verdict"] == "OPAQUE_PRODUCER_GAP"
  assert entry["opaque_producer"]["program_count"] > 1


def test_split_cause_is_not_conflated_across_mechanisms():
  """A lowered reduce and a precompiled FUNCTION boundary both split the program count.
  Reporting both as 'reduction + dependent epilogue' would merge C1 with the M4 blocker."""
  got = run()
  assert "reduction + dependent epilogue" in got["populations"][POP_NORMS]["ordinary_reason"]
  assert "reduction + dependent epilogue" in got["populations"][POP_FLASH]["reason"]
  residual = got["populations"][POP_RESIDUAL]
  assert "precompiled FUNCTION producer boundary" in residual["opaque_producer_reason"]
  assert "reduction + dependent epilogue" not in residual["opaque_producer_reason"]


def test_norms_and_flash_still_share_the_ordinary_reduce_then_consume_mechanism():
  """REDUCE_OUTPUT resolves norms for identity inputs; the ordinary arms remain a shared gap."""
  got = run()
  assert got["populations"][POP_NORMS]["verdict"] == "REDUCE_OUTPUT_PASS"
  assert got["populations"][POP_FLASH]["verdict"] == "CONSTRUCTION_GAP"
  assert "reduction + dependent epilogue" in got["populations"][POP_NORMS]["ordinary_reason"]
  assert "reduction + dependent epilogue" in got["populations"][POP_FLASH]["reason"]


def test_populations_without_an_opaque_arm_say_so_rather_than_implying_plain():
  """Absence from OPAQUE_ARMS means 'not assessed', and the scope string must say that."""
  got = run()
  for pop, entry in got["populations"].items():
    if pop in OPAQUE_ARMS or entry["verdict"] == "NO_CONSTRUCTION": continue
    assert "not assessed" in entry["scope"]
    assert "opaque_producer" not in entry


def test_run_v3_preserves_the_closed_pre_reduce_output_gate():
  """The closed norms/residual A/B harnesses still use the v3 construction gate."""
  got = run_v3()
  assert got["schema"] == "tinygrad.nv_boundary_free_ordinary_uop_gate.v3"
  norms = got["populations"][POP_NORMS]
  assert norms["verdict"] == "CONSTRUCTION_GAP"
  assert norms["ordinary_verdict"] == "CONSTRUCTION_GAP"
  assert "reduce_output" not in norms
  for row in norms["baseline"].values():
    assert row["program_count"] == 2
