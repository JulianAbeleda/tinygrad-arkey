"""Tests for extra/audit/lowering_fingerprint.py (CPU-only lowering fingerprint gate).

CPU-only throughout: this gate never touches a GPU (every graph is built with device="CPU" explicitly, and
schedule_linear() never renders/executes a kernel), so none of these tests require or check for AMD.
"""
from __future__ import annotations

import copy
import json

from extra.audit import lowering_fingerprint as lf


# --------------------------------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------------------------------

def test_compute_fingerprints_is_deterministic_across_two_in_process_runs():
  first = lf.compute_fingerprints()
  second = lf.compute_fingerprints()
  assert first == second
  assert len(first) == 7
  assert all(isinstance(h, str) and len(h) == 64 for h in first.values())


def test_all_graph_names_present():
  fp = lf.compute_fingerprints()
  expected = {
    "elementwise_reduce", "matmul", "chained_reduce", "broadcast_max", "softmax_like",
    "transpose_matmul", "cast_dtype_roundtrip",
  }
  assert set(fp) == expected


# --------------------------------------------------------------------------------------------------------------
# --check against a freshly written baseline
# --------------------------------------------------------------------------------------------------------------

def test_check_passes_against_a_freshly_written_baseline(tmp_path, monkeypatch):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lf, "OUT_PATH", out_path)
  artifact = lf.build_artifact([])
  out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
  assert lf.run_check([]) == 0


def test_check_detects_a_mutated_hash_and_reports_which_graph_changed(tmp_path, monkeypatch, capsys):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lf, "OUT_PATH", out_path)
  artifact = lf.build_artifact([])
  mutated = copy.deepcopy(artifact)
  mutated["fingerprints"]["matmul"] = "0" * 64
  out_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
  rc = lf.run_check([])
  assert rc == 1
  out = capsys.readouterr().out
  assert "matmul" in out
  assert "CHANGED" in out
  # only the mutated graph should be reported as changed
  for name in artifact["fingerprints"]:
    if name != "matmul":
      assert f"{name:24s} CHANGED" not in out


def test_check_reports_added_and_removed_graphs(tmp_path, monkeypatch):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lf, "OUT_PATH", out_path)
  stored = {"header": {"schema": lf.SCHEMA}, "fingerprints": {"only_in_stored": "a" * 64}}
  out_path.write_text(json.dumps(stored))
  assert lf.run_check([]) == 1


def test_check_fails_loudly_when_no_baseline_stored(tmp_path, monkeypatch):
  out_path = tmp_path / "does_not_exist.json"
  monkeypatch.setattr(lf, "OUT_PATH", out_path)
  assert lf.run_check([]) == 1


# --------------------------------------------------------------------------------------------------------------
# Env-var stripping: the leading suspect for the original ~1/8 flake was a leaked gate/tuning env var. Prove the
# strip actually removes matching keys, and that a leaked PREFILL_* var does not change the fingerprint.
# --------------------------------------------------------------------------------------------------------------

def test_strip_gate_env_vars_removes_matching_prefixes(monkeypatch):
  monkeypatch.setenv("PREFILL_SOFTMAX_REDUCE_FUSE", "0")
  monkeypatch.setenv("SCHED_BEAM", "4")
  monkeypatch.setenv("TC_OPT", "2")
  monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")
  removed = lf.strip_gate_env_vars()
  import os
  assert "PREFILL_SOFTMAX_REDUCE_FUSE" not in os.environ
  assert "SCHED_BEAM" not in os.environ
  assert "TC_OPT" not in os.environ
  assert os.environ.get("SOME_UNRELATED_VAR") == "keep-me"
  assert "PREFILL_SOFTMAX_REDUCE_FUSE" in removed
  assert "SCHED_BEAM" in removed
  assert "TC_OPT" in removed


def test_leaked_prefill_env_var_does_not_change_the_fingerprint(monkeypatch):
  baseline = lf.compute_fingerprints()
  monkeypatch.setenv("PREFILL_SOFTMAX_REDUCE_FUSE", "0")
  with_leak = lf.compute_fingerprints()
  assert with_leak == baseline
  import os
  # strip_gate_env_vars() runs inside compute_fingerprints(); confirm the leaked var is gone afterward too.
  assert "PREFILL_SOFTMAX_REDUCE_FUSE" not in os.environ
