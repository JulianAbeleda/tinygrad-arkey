"""TG8 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): the third of the
three pure policy gates this package addresses is `prefill_candidate_runtime.py:162`.

After inspection this is NOT a live capability/policy admission gate in the TG3 sense: `promoted_candidate_set()`
takes zero arguments (it never consults scanned_device_facts) and is an ARTIFACT-IDENTITY guard, the same kind
as the canonical_identity/legacy_identity/candidate_set_identity hash checks beside it -- it asserts the one
checked-in compact artifact still declares the exact target its searched tile/LDS/thread schedule was compiled
for, never a live-hardware decision. That raise is preserved unchanged (`test_compact_target_mismatch_raises_loudly`).

The real live admission is `automatic_promoted_prefill_graph_policy`, which already derives BOTH the resolved
target (from live scanned_device_facts) and the promoted-target set (from the artifact's own recorded entries,
via `promoted_prefill_graph_targets`) with no hardcoded literal -- capability and policy are the same fused
question for this exact-shape compiled kernel (a searched WMMA/LDS schedule is measured for one exact
(backend, arch, wave_size) triple, not a generically-expressible capability), so it is not split further.
"""
import json

import pytest

from tinygrad.llm import prefill_candidate_runtime
from tinygrad.llm.prefill_candidate_runtime import (
  CandidateAdmission, CandidateRegistry, automatic_promoted_prefill_graph_policy,
  promoted_candidate_registry, promoted_candidate_set, promoted_prefill_graph_targets,
)


def test_compact_target_mismatch_raises_loudly(tmp_path, monkeypatch):
  """The artifact-identity guard never consults live device facts and must still fail loudly on drift."""
  forged = {"schema": prefill_candidate_runtime.COMPACT_SCHEMA, "route_id": prefill_candidate_runtime.ROUTE_ID,
            "candidate_set_identity": "irrelevant", "profile": "p",
            "target": {"backend": "METAL", "arch": "Apple9", "wave_size": 32}, "template": {}, "entries": []}
  path = tmp_path / "forged.json"
  path.write_text(json.dumps(forged))
  monkeypatch.setattr(prefill_candidate_runtime, "ARTIFACT", path)
  promoted_candidate_set.cache_clear()
  try:
    with pytest.raises(ValueError, match="compact target is unsupported"):
      promoted_candidate_set()
  finally:
    promoted_candidate_set.cache_clear()


def test_promoted_prefill_graph_targets_reads_the_checked_in_artifacts_own_target():
  """Today's real artifact was searched/compiled for exactly one target -- confirm the promoted-target set is
  read from the artifact's data, not a python literal, by checking it matches that one recorded target."""
  assert promoted_prefill_graph_targets(promoted_candidate_registry()) == frozenset({("AMD", "gfx1100", 32)})


def test_promoted_prefill_graph_targets_has_no_hardcoded_backend():
  """A synthetic registry (never touching the real AMD artifact) proves the function has zero hardcoded
  backend/arch strings: whatever target a promoted candidate set records is exactly what comes back."""
  admission = CandidateAdmission("f" * 64, {"workload": {"target": {"backend": "METAL", "arch": "Apple9", "wave_size": 32}}},
                                  geometry=None, pipeline_plan=None, active_lds_bytes=0, context=None)
  registry = CandidateRegistry(candidate_set=None, admissions=(admission,), exact_index={})
  assert promoted_prefill_graph_targets(registry) == frozenset({("METAL", "Apple9", 32)})


def test_amd_structural_admission_is_unchanged_by_the_promoted_prefill_graph_targets_refactor():
  """Structural-only (no AMD hardware here, scope section 8): with real AMD gfx1100 wave32 facts, the exact
  same four-role candidate set is admitted through `automatic_promoted_prefill_graph_policy` after the TG8
  refactor as before it -- the refactor only names the existing target-set computation, it does not change it."""
  candidate_set = promoted_candidate_set().to_json()
  rows = []
  for index, entry in enumerate(candidate_set["entries"]):
    workload = entry["payload"]["workload"]
    rows.append({"invocation_id": f"candidate-{index}", "candidate_controlled": True,
                 "role": workload["role"], "shape": workload["shape"]})
  rows.append({"invocation_id": "lm-head", "candidate_controlled": False, "fixed_route_id": "fixed-ggml-linear"})
  inventory = {"inventory_identity": "inventory:sha256:" + "a" * 64, "rows": rows}
  amd_facts = {"backend": "AMD", "architecture": "gfx1100", "capabilities": {"wave_size": 32}}
  policy = automatic_promoted_prefill_graph_policy(inventory, amd_facts)
  assert policy is not None and policy["candidate_id"] == "prefill_wmma_lds_dbuf_generated"
  assert len(policy["graph_gemm"]["policy_rows"]) == 4

  metal_facts = {"backend": "METAL", "architecture": "Apple9", "capabilities": {"wave_size": None}}
  assert automatic_promoted_prefill_graph_policy(inventory, metal_facts) is None
