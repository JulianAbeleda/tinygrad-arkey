import numpy as np
import pytest

from extra.qk.mmq_regression import (validate_generated_mmq_abi, vector_pointer_bases, reject_vector_pointer_bases,
                                     validate_mmq_candidate_evidence_gate)
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec, enumerate_q4k_q8_mmq_candidates
from extra.qk.q4k_q8_mmq_emitter import MMQEmitterCandidate


def _spec(**kw):
  fields = dict(workload="test", profile="test", role="ffn_gate_up", quant_format="Q4_K",
                activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows",
                m=16, n=32, k=256, tile_m=16, tile_n=16, tile_k=256)
  fields.update(kw)
  return Q4KQ8MMQPrefillSpec(**fields)

def _candidate(spec):
  return MMQEmitterCandidate(spec, min(spec.m, 16), min(spec.n, 16), 16, "group", spec.output_layout,
                             spec.activation_layout, spec.tile_x_layout, spec.tile_y_layout,
                             spec.staging_strategy, spec.writeback_strategy)


def test_generated_mmq_abi_is_exact_q4_words_q8_values_scales_and_sums():
  # The emitter's q8 sum is an internal float32 reduction; the host ABI still
  # requires one float32 value per Q8 block, alongside the exact Q4/Q8 storage.
  validate_generated_mmq_abi(np.zeros(32 * 36, np.uint32), np.zeros((16, 256), np.int8),
                             np.ones((16, 8), np.float32), m=16, n=32, k=256)
  with pytest.raises(ValueError, match="Q4 ABI"):
    validate_generated_mmq_abi(np.zeros(32 * 36, np.uint16), np.zeros((16, 256), np.int8),
                               np.ones((16, 8), np.float32), m=16, n=32, k=256)


def test_candidate_axes_are_descriptor_search_owned_only():
  candidates = tuple(enumerate_q4k_q8_mmq_candidates(_spec(), tile_m=(16, 32), tile_n=(16,)))
  assert {(c.tile_m, c.tile_n, c.tile_k) for c in candidates} == {(16, 16, 256), (32, 16, 256)}
  with pytest.raises(ValueError, match="inert search axes"):
    tuple(enumerate_q4k_q8_mmq_candidates(_spec(), lds_bytes=(0, 1024)))


def test_smallest_emitter_graph_exposes_pre_late_vector_pointer_bases():
  from tinygrad import Tensor, dtypes
  from extra.qk.q4k_q8_mmq_emitter import emit_q4k_q8_mmq_prefill
  spec = _spec(m=8, n=16, tile_m=8, tile_n=16)
  out = emit_q4k_q8_mmq_prefill(Tensor.zeros(16 * 36, dtype=dtypes.uint32), Tensor.zeros((8, 256), dtype=dtypes.int8),
                                Tensor.ones((8, 8), dtype=dtypes.float32), _candidate(spec))
  assert out.shape == (8, 16)


def test_vector_pointer_base_probe_identifies_vector_index_bases():
  class DType: count = 4
  class Node:
    op = type("Op", (), {"name": "INDEX"})
    dtype = DType()
    src = (type("Base", (), {"dtype": DType()})(),)
  class Root:
    def toposort(self): return (Node(),)
  found = vector_pointer_bases(Root())
  assert len(found) == 1 and found[0].base is not None


def test_vector_pointer_bases_are_rejected_but_wmma_vector_carriers_are_valid():
  class DType:
    def __init__(self, count): self.count = count
  class Node:
    def __init__(self, name, dtype, src=()):
      self.op = type("Op", (), {"name": name})
      self.dtype, self.src = dtype, src
  vector_base = type("Base", (), {"dtype": DType(4)})()
  bad = type("Root", (), {"toposort": lambda self: (Node("INDEX", DType(1), (vector_base,)),)})()
  with pytest.raises(ValueError, match="vector-valued pointer base"):
    reject_vector_pointer_bases(bad)

  # WMMA fragments are intentionally vector-valued carriers, not pointer bases.
  wmma = type("Root", (), {"toposort": lambda self: (Node("WMMA", DType(4)),)})()
  reject_vector_pointer_bases(wmma)


def test_candidate_evidence_gate_is_fail_closed_for_timing_and_promotion():
  evidence = {name: {"passed": True, "status": "PASS"}
              for name in ("correctness", "guard", "gpu_health", "resources", "identity", "fallback")}
  assert validate_mmq_candidate_evidence_gate(evidence)["timing_allowed"] is True
  evidence.pop("identity")
  decision = validate_mmq_candidate_evidence_gate(evidence)
  assert decision["timing_allowed"] is False
  assert decision["promotion_eligible"] is False
  assert "missing or failed identity evidence" in decision["blockers"]


def test_candidate_evidence_gate_requires_explicit_fallback_evidence():
  evidence = {name: {"passed": True} for name in ("correctness", "guard", "gpu_health", "resources", "identity")}
  decision = validate_mmq_candidate_evidence_gate(evidence)
  assert decision["promotion_eligible"] is False
  assert "missing or failed fallback evidence" in decision["blockers"]
  evidence["no_fallback"] = True
  assert validate_mmq_candidate_evidence_gate(evidence)["promotion_eligible"] is True
