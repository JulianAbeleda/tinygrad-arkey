from __future__ import annotations

import hashlib
import json

import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import UOp

from extra.qk.mmq_experiment import (
  BACKEND, BUNDLE_SCHEMA, CANDIDATE_SCHEMA, CANDIDATE_IDS, MMQCandidateSpec, canonical_candidate,
  produce_experiment_bundle,
)
from extra.qk.mmq_q4k_q8_atom import _q4k_q8_1_bounded_ds4_coop_tile_kernel


def _fake_report(status="PASS"):
  return {
    "status": status,
    "compile_evidence": {"binary_sha256": "b" * 64, "resources": {
      "vgpr": 32, "sgpr": 16, "lds_bytes": 256, "scratch_bytes": 0, "workgroup_threads": 32}},
    "correctness": {"max_abs": 0.0, "atol": 0.001, "tiles": 1},
    "timing": {"samples_ms": [2.0] * 10, "min_ms": 2.0, "median_ms": 2.0,
               "comparator_id": "direct_packed", "comparator_status": "measured",
               "direct_packed": {"status": "PASS", "samples_ms": [1.0] * 10,
                                 "min_ms": 1.0, "median_ms": 1.0}},
  }


@pytest.mark.parametrize("mode", tuple(CANDIDATE_IDS))
def test_canonical_candidate_round_trip(mode):
  spec = canonical_candidate(mode, seed=7)
  raw = spec.to_json()
  assert raw["schema"] == CANDIDATE_SCHEMA
  assert raw["candidate_id"] == CANDIDATE_IDS[mode]
  assert raw["backend"] == BACKEND
  assert raw["shape"] == {"M": 16, "N": 16, "K": 256}
  assert MMQCandidateSpec.from_json(json.loads(json.dumps(raw))) == spec


def test_candidate_rejects_identity_mode_and_sampling_mismatches():
  with pytest.raises(ValueError, match="candidate_id does not match"):
    MMQCandidateSpec(CANDIDATE_IDS["gated_matrix_v0"], "direct_owner_v0").validate()
  with pytest.raises(ValueError, match="warmups >= 3"):
    canonical_candidate("gated_matrix_v0", warmups=2).validate()
  raw = canonical_candidate("gated_matrix_v0").to_json()
  raw["backend"] = "other"
  with pytest.raises(ValueError, match="backend must be"):
    MMQCandidateSpec.from_json(raw)


def _kernel_repr(mode: str) -> str:
  kernel = _q4k_q8_1_bounded_ds4_coop_tile_kernel(16, 16, 256, "ffn_gate_up", mode)
  return repr(kernel(
    UOp.placeholder((16, 16), dtypes.float32, 0),
    UOp.placeholder((16 * 36,), dtypes.uint32, 1),
    UOp.placeholder((2 * 16 * 128,), dtypes.int8, 2),
    UOp.placeholder((2 * 16 * 4,), dtypes.float32, 3),
    UOp.placeholder((2 * 16 * 4,), dtypes.float32, 4),
  ))


def test_writeback_modes_preserve_body_marker_and_change_store_surface():
  gated, direct = _kernel_repr("gated_matrix_v0"), _kernel_repr("direct_owner_v0")
  assert "gated_matrix_v0" in gated and "direct_owner_v0" in direct
  assert gated != direct
  assert gated.count("STORE") > direct.count("STORE")
  for marker in ("BARRIER", "REDUCE", "q8_scales", "q8_sums"):
    assert gated.count(marker) == direct.count(marker)


def test_atomic_bundle_propagates_identity_and_hashes_files(tmp_path):
  out = tmp_path / "bundle"
  spec = canonical_candidate("direct_owner_v0")
  produce_experiment_bundle(spec, out, experiment_id="exp-1", system_snapshot_id="sys-1",
                            runner=lambda config: _fake_report())
  manifest = json.loads((out / "manifest.json").read_text())
  assert manifest["schema"] == BUNDLE_SCHEMA
  assert manifest["state"] == "EVIDENCE_COMPLETE"
  assert manifest["complete"] is True
  assert manifest["production_dispatch_changed"] is False
  assert not list(tmp_path.glob(".bundle.tmp-*"))
  for name, digest in manifest["files"].items():
    raw = (out / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    artifact = json.loads(raw)
    assert artifact["candidate_id"] == spec.candidate_id
    assert artifact["experiment_id"] == "exp-1"
    assert artifact["system_snapshot_id"] == "sys-1"
    assert artifact["source_sha256"] == manifest["source_sha256"]
    assert artifact["production_dispatch_changed"] is False


def test_failure_bundle_is_atomic_structured_and_not_complete(tmp_path):
  def fail(_config): raise RuntimeError("compile failed")
  out = tmp_path / "failed"
  produce_experiment_bundle(canonical_candidate("gated_matrix_v0"), out, experiment_id="exp-fail",
                            system_snapshot_id="sys-1", runner=fail)
  manifest = json.loads((out / "manifest.json").read_text())
  assert manifest["state"] == "PRODUCER_ERROR"
  assert manifest["evidence_complete"] is False
  assert manifest["complete"] is False
  assert manifest["error"] == {"type": "RuntimeError", "message": "compile failed"}
  assert set(manifest["files"]) == {"candidate.json"}
  assert not list(tmp_path.glob(".failed.tmp-*"))


def test_bundle_refuses_to_replace_existing_output(tmp_path):
  out = tmp_path / "existing"
  out.mkdir()
  with pytest.raises(FileExistsError):
    produce_experiment_bundle(canonical_candidate("gated_matrix_v0"), out, experiment_id="e",
                              system_snapshot_id="s", runner=lambda config: _fake_report())
