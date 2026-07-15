import inspect
import json
import numpy as np
import pytest
import extra.qk.q4k_q8_mmq_generated_harness as harness
from extra.qk.q4k_q8_mmq_generated_harness import _coverage, PROVENANCE, _validate_final_contract
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec

def _spec():
  return Q4KQ8MMQPrefillSpec(workload="test", profile="test", role="test", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows", m=16, n=16, k=256)

def test_generated_harness_declares_emitted_provenance_and_full_owner_coverage():
  assert PROVENANCE == "q4k_q8_mmq_descriptor_emitter_v1"
  coverage = _coverage(_spec())
  assert coverage["complete"] and coverage["covered_output_elements"] == 256

def test_harness_coverage_is_descriptor_driven():
  spec = _spec(); spec = spec.__class__(**{**spec.__dict__, "tile_m": 8, "tile_n": 8})
  assert _coverage(spec)["tile_count"] == 4

def test_generated_child_uses_descriptor_emitter_without_legacy_atom_import():
  source = inspect.getsource(harness)
  assert "emit_q4k_q8_mmq_prefill" in source
  assert "extra.qk.mmq_q4k_q8_atom" not in source
  assert "extra.qk.amdgpu_metadata" in source

def test_bootstrap_rejects_unlowered_launch_metadata(tmp_path):
  payload = {"spec": {**_spec().to_json(), "launch": {"workgroup_size": 64, "waves": 2}}}
  path = tmp_path / "bootstrap.json"
  path.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="launch metadata is unsupported"):
    harness.bootstrap_from_file(path)

def test_bootstrap_rejects_noncanonical_abi(tmp_path):
  payload = {"spec": {**_spec().to_json(), "abi": {"arguments": ["out"], "dtypes": ["float32"], "output_layout": "tokens_rows"}}}
  path = tmp_path / "bootstrap.json"
  path.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="ABI is unsupported"):
    harness.bootstrap_from_file(path)

def test_final_contract_never_infers_geometry_without_shared_candidate():
  with pytest.raises(ValueError, match="shared logical candidate"):
    _validate_final_contract(_spec(), {"abi": _spec().abi.to_json(),
      "geometry": {"global_size": [1, 1, 1], "local_size": [64, 1, 1]},
      "physical_contract": {}}, None)

def test_bootstrap_requires_matching_logical_candidate(tmp_path):
  payload = {"spec": _spec().to_json()}
  path = tmp_path / "bootstrap.json"; path.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="shared logical candidate"):
    harness.bootstrap_from_file(path)

def test_bootstrap_rejects_logical_candidate_identity_mismatch(tmp_path):
  spec = _spec(); logical = spec.logical_candidate()
  candidate = logical.to_dict()
  payload = {"spec": spec.to_json(), "logical_candidate": candidate,
             "candidate_identity": "wrong"}
  path = tmp_path / "bootstrap.json"; path.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="identity mismatch"):
    harness.bootstrap_from_file(path)

@pytest.mark.parametrize("evidence", [
  {"abi": _spec().abi.to_json(), "geometry": {"global_size": [1], "local_size": [1]},
   "physical_contract": {"local_size": [1], "consumed_local_dims": [0], "lane_map": {"lane": "lidx0"},
     "barriers": [], "owners": [], "expected_outputs": []}},
])
def test_final_contract_requires_artifact_identity(evidence):
  candidate = _spec().logical_candidate()
  with pytest.raises(ValueError, match="source/binary identity"):
    _validate_final_contract(_spec(), evidence, candidate)
