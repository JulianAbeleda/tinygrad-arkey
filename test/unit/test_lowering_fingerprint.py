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
  assert len(first) == len(lf._build_graphs(__import__('tinygrad', fromlist=['Tensor']).Tensor))
  assert all(isinstance(h, str) and len(h) == 64 for h in first.values())


def test_all_graph_names_present():
  fp = lf.compute_fingerprints()
  expected = {
    "elementwise_reduce", "matmul", "chained_reduce", "broadcast_max", "softmax_like",
    "transpose_matmul", "cast_dtype_roundtrip",
    # Added after a review found the gate blind to assign: realize_store_after_src's WAR-hazard branch fires on
    # these and on nothing else in the corpus. Do not drop them without replacing that coverage.
    "assign_self_dependent", "assign_war_hazard", "multi_output_shared_producer",
  }
  assert set(fp) == expected


# --------------------------------------------------------------------------------------------------------------
# --check against a freshly written baseline
# --------------------------------------------------------------------------------------------------------------

def test_check_passes_against_a_freshly_written_baseline(tmp_path, monkeypatch):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lf, "OUT_PATH", out_path)
  artifact = lf.build_artifact([], require_cold=False)
  out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
  assert lf.run_check([]) == 0


def test_check_detects_a_mutated_hash_and_reports_which_graph_changed(tmp_path, monkeypatch, capsys):
  out_path = tmp_path / "latest.json"
  monkeypatch.setattr(lf, "OUT_PATH", out_path)
  artifact = lf.build_artifact([], require_cold=False)
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


# --------------------------------------------------------------------------------------------------------------
# LR-032b: pass order is pinned, and the registry's coverage claim is measured rather than asserted
# --------------------------------------------------------------------------------------------------------------

def test_collapse_drops_matcher_construction_and_consecutive_repeats():
  seq = ["process UPat", "simplify", "simplify", "compile UPat", "simplify", "pad", "pad"]
  # Matcher-construction names are dropped FIRST, so the two simplify runs they separated become one run and
  # collapse together. That ordering is deliberate: 'compile UPat' appears wherever a matcher happens to be built
  # first, so letting it split a run would make the order depend on corpus iteration order.
  assert lf._collapse(seq) == ["simplify", "pad"]


def test_collapse_keeps_a_pass_that_is_genuinely_re_entered():
  """Collapsing must not deduplicate globally -- 'simplify, pad, simplify' means the pipeline came back to
  simplify, which is exactly the kind of structure this gate exists to pin."""
  assert lf._collapse(["simplify", "pad", "simplify"]) == ["simplify", "pad", "simplify"]


def test_pass_orders_are_deterministic_across_two_cold_runs():
  assert lf.pass_orders_in_fresh_process() == lf.pass_orders_in_fresh_process()


def test_a_warm_process_refuses_to_report_a_pass_order():
  """The gate's sharpest edge. A warm run records 132 collapsed steps where a cold run records 981, because
  UOp-keyed caches skip the rewrites entirely -- so a warm order silently looks like a massive pipeline change.
  Returning that would be far worse than refusing."""
  import pytest
  lf._run_corpus()                       # warm this process, whatever ran before
  with pytest.raises(lf.WarmProcessError):
    lf.compute_pass_orders()


def test_every_graph_has_a_nonempty_pass_order():
  orders = lf.pass_orders_in_fresh_process()
  assert set(orders) == set(lf.compute_fingerprints())
  assert all(len(v) > 10 for v in orders.values())


def test_enabling_the_trace_does_not_change_the_fingerprint():
  """The load-bearing claim behind capturing order inside the fingerprint run: tracing is an observer. If
  record_rewrite ever touched the graph, the stored fingerprints would move and this gate would be measuring its
  own instrumentation."""
  import json as _json, pathlib
  stored = _json.loads(pathlib.Path(lf.OUT_PATH).read_text())["fingerprints"]
  assert lf.compute_fingerprints() == stored


def test_order_diff_names_the_pass_that_moved():
  old = {"g": ["a", "b", "c", "d"]}
  new = {"g": ["a", "b", "renamed", "d"]}
  rows = lf._classify_order_diff(old, new)
  assert len(rows) == 1 and rows[0][1] == "REORDERED"
  assert "'c'" in rows[0][2] and "'renamed'" in rows[0][2]


def test_order_diff_detects_an_inserted_pass():
  rows = lf._classify_order_diff({"g": ["a", "b"]}, {"g": ["a", "new", "b"]})
  assert len(rows) == 1 and "len 2 -> 3" in rows[0][2]


def test_order_diff_is_silent_on_an_identical_order():
  assert lf._classify_order_diff({"g": ["a", "b"]}, {"g": ["a", "b"]}) == []


def test_registry_coverage_numbers_are_recomputed_not_asserted():
  """OBSERVED_NAME_JOIN records why the pass registry cannot be asserted against the observed pass sequence
  (function granularity vs call granularity). Those numbers are a finding, so they must be recomputed from the
  live registry and the live trace -- a stale comment claiming 7/64 would be worse than no comment."""
  from tinygrad.codegen.passes import OBSERVED_NAME_JOIN, REGISTRY

  orders = lf.pass_orders_in_fresh_process()
  observed = {n for v in orders.values() for n in v}
  assert len(observed) == OBSERVED_NAME_JOIN["distinct_observed_names"]

  def norm(s: str) -> str: return s.lower().replace(" ", "-").replace("-", "_")
  reg_leaf = {norm(pid.split(".")[-1]) for pid in REGISTRY}
  assert sum(1 for o in observed if norm(o) in reg_leaf) == OBSERVED_NAME_JOIN["matched_by_name"]

  assert len(REGISTRY) == OBSERVED_NAME_JOIN["registry_descriptors"]
  assert len({d.owner_file for d in REGISTRY.values()}) == OBSERVED_NAME_JOIN["registry_owner_files"]

  steps = [n for v in orders.values() for n in v]
  assert len(steps) == OBSERVED_NAME_JOIN["total_collapsed_steps"]
  assert steps.count("<unnamed>") == OBSERVED_NAME_JOIN["unnamed_collapsed_steps"]


def test_unnamed_rewrites_are_a_bounded_and_declared_blind_spot():
  """The order gate cannot see a reorder among rewrites that pass no name=. That limit is acceptable only while it
  is known and bounded; if it grows, this fails and the number in passes.py has to be re-argued."""
  from tinygrad.codegen.passes import OBSERVED_NAME_JOIN
  frac = OBSERVED_NAME_JOIN["unnamed_collapsed_steps"] / OBSERVED_NAME_JOIN["total_collapsed_steps"]
  assert frac < 0.25, f"unnamed rewrites now {frac:.0%} of the pipeline; name them or re-argue the gate's resolution"
