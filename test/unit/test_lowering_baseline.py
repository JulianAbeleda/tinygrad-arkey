"""Tests for extra/audit/lowering_baseline.py (LR-000 compile-only lowering fingerprint baseline).

Kept CPU-only where possible: the diff/classification logic is pure Python and is tested with synthetic entry
dicts so this file never requires AMD to exercise "does a mutated source hash get reported as a source change,
not a shape change". Only the actual end-to-end compile (build_artifact / run_check against real kernels) needs
AMD; those tests use the repo's existing `_has_amd()` skip idiom (see test/unit/test_mmq_q4k_q8_atom.py).
"""
from __future__ import annotations

import copy
import json

import pytest

from extra.audit import lowering_baseline as lb


def _has_amd() -> bool:
  """Compile-only AMD availability check: never executes a device kernel (matches the LR-000 no-GPU-execution
  contract), just confirms the HIP renderer/compiler can be constructed and a trivial sink can be lowered."""
  try:
    lb._amd_renderer()
    return True
  except Exception:
    return False


# --------------------------------------------------------------------------------------------------------------
# Pure diff/classification logic: CPU-only, no AMD required.
# --------------------------------------------------------------------------------------------------------------

def _base_entry() -> dict:
  return {
    "kernel_id": "decode_q4k_g3_lanemap.ffn_gate_up.8B",
    "route_id": "decode_q4k_g3_generated",
    "model": "8B",
    "shape": {"rows": 12288, "k": 4096, "role": "ffn_gate_up"},
    "program_name": "q4k_g3_lanemap_gemv_12288_4096",
    "source_sha256": "a" * 64,
    "binary_sha256": "b" * 64,
    "resources": {"vgpr": 32, "sgpr": 16, "lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0,
                  "sgpr_spills": 0, "workgroup_threads": 32, "wavefront_size": 32},
    "workgroup": [32, 1, 1],
    "grid": [12288, 1, 1],
    "skipped_reason": None,
  }


def test_identical_entries_produce_no_diff():
  old, new = _base_entry(), _base_entry()
  assert lb._classify_entry_diff(old, new) == []


def test_mutated_source_hash_is_reported_as_source_change_not_shape_change():
  old = _base_entry()
  new = _base_entry()
  new["source_sha256"] = "c" * 64
  changes = lb._classify_entry_diff(old, new)
  assert changes == ["source_hash"]
  assert "shape" not in changes
  assert "route" not in changes


def test_shape_change_is_reported_as_shape_not_source():
  old = _base_entry()
  new = _base_entry()
  new["shape"] = {**new["shape"], "rows": 17408}
  # a real shape change also changes the compiled source/resources; the classifier must still name "shape"
  # first-class rather than only reporting the source-hash symptom.
  new["source_sha256"] = "d" * 64
  new["resources"] = {**new["resources"], "vgpr": 40}
  changes = lb._classify_entry_diff(old, new)
  assert "shape" in changes
  assert "source_hash" in changes


def test_route_change_is_distinguished_from_source_change():
  old = _base_entry()
  new = _base_entry()
  new["route_id"] = "some_other_route"
  changes = lb._classify_entry_diff(old, new)
  assert changes == ["route"]


def test_resource_only_change_is_reported_as_resources():
  old = _base_entry()
  new = _base_entry()
  new["resources"] = {**new["resources"], "vgpr": 999}
  changes = lb._classify_entry_diff(old, new)
  assert changes == ["resources"]


def test_run_check_reports_added_and_removed_kernels(tmp_path, monkeypatch):
  stored = {"header": {"schema": lb.SCHEMA}, "entries": [_base_entry()]}
  out_path = tmp_path / "latest.json"
  out_path.write_text(json.dumps(stored))
  monkeypatch.setattr(lb, "OUT_PATH", out_path)

  removed_entry = _base_entry()
  added_entry = _base_entry()
  added_entry["kernel_id"] = "decode_q4k_g3_lanemap.ffn_down.8B"
  fresh = {"header": {"schema": lb.SCHEMA}, "entries": [added_entry]}
  monkeypatch.setattr(lb, "build_artifact", lambda argv: fresh)

  rc = lb.run_check([])
  assert rc == 1


def test_run_check_passes_when_fresh_matches_stored(tmp_path, monkeypatch):
  entry = _base_entry()
  stored = {"header": {"schema": lb.SCHEMA}, "entries": [entry]}
  out_path = tmp_path / "latest.json"
  out_path.write_text(json.dumps(stored))
  monkeypatch.setattr(lb, "OUT_PATH", out_path)
  monkeypatch.setattr(lb, "build_artifact", lambda argv: copy.deepcopy(stored))

  rc = lb.run_check([])
  assert rc == 0


# --------------------------------------------------------------------------------------------------------------
# End-to-end compile-only tests: require AMD (compile-only; no device execution per LR-000 contract).
# --------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not _has_amd(), reason="AMD renderer/compiler is not available")
def test_build_artifact_is_deterministic_in_process():
  first = lb.build_artifact([])
  second = lb.build_artifact([])
  # header.command/git/python fields are process-stable; compare everything except nothing -- both runs are
  # in the same process against the same tree, so the whole artifact must be byte-identical.
  assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
  assert len(first["entries"]) > 0
  assert all(e["skipped_reason"] is None for e in first["entries"])


@pytest.mark.skipif(not _has_amd(), reason="AMD renderer/compiler is not available")
def test_check_passes_against_a_freshly_written_baseline(tmp_path, monkeypatch):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lb, "OUT_PATH", out_path)
  artifact = lb.build_artifact([])
  out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
  assert lb.run_check([]) == 0


@pytest.mark.skipif(not _has_amd(), reason="AMD renderer/compiler is not available")
def test_check_detects_a_real_mutated_source_hash(tmp_path, monkeypatch):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lb, "OUT_PATH", out_path)
  artifact = lb.build_artifact([])
  mutated = copy.deepcopy(artifact)
  mutated["entries"][0]["source_sha256"] = "0" * 64
  out_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
  assert lb.run_check([]) == 1
