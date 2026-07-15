import inspect
import json
import numpy as np
import pytest
import extra.qk.q4k_q8_mmq_generated_harness as harness
from extra.qk.q4k_q8_mmq_generated_harness import _coverage, PROVENANCE
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
