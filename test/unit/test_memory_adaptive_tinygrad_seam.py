import gc, json, weakref
import pytest
from types import SimpleNamespace

from extra.qk.memory_adaptive_transport import SelectedModelScan
from extra.qk.memory_adaptive_allocation_observer import SCHEMA as CHECKPOINT_OBSERVER_SCHEMA
from extra.qk.memory_adaptive_tinygrad_seam import (TinygradWholeModelSeam, _compiled_resource_artifact,
  _bind_manifest_physical_owners, _drain_manifest_rows, _failed_artifacts, _reconcile_memory_authorities,
  _route_graph_materialization_bound, _selected_runtime_base_terms)
from extra.qk.memory_adaptive_tinygrad_seam import _transport_encode, _validate_transport
from tinygrad.llm.memory_semantics import MemorySemanticClass, MemorySemanticOwner
from tinygrad.llm.prefill_memory_plan import ByteLifetime, ByteTerm, CandidateMemoryCoverage, Strategy
from extra.qk.memory_adaptive_autoscan import AutoscanCandidate
from tinygrad import UOp, dtypes
from extra.qk.physical_memory_ledger import PhysicalMemoryLedger
from tinygrad.llm.physical_memory_ledger import AllocationOwner
from tinygrad.schedule.memory import ScheduleMemoryArena, ScheduleMemoryBuffer, ScheduleMemoryManifest


def _model():
  rows = [{"invocation_id": "i0", "candidate_controlled": True, "shape": {"m": 512, "n": 4, "k": 8}},
          {"invocation_id": "i1", "candidate_controlled": True, "shape": {"m": 512, "n": 8, "k": 4}},
          {"invocation_id": "lm", "candidate_controlled": False, "fixed_route_id": "fixed-ggml-linear",
           "shape": {"m": 1, "n": 32000, "k": 8}}]
  return SelectedModelScan({"content_hash": "sha256:x"}, {"rows": rows},
    (ByteTerm("model", 1, "test", "exact", ByteLifetime.PERSISTENT),), {"prompt_tokens": 544}, {})


def _policy(candidate_id, routes=None):
  return {"candidate_id": candidate_id, "whole_policy_identity": f"identity:{candidate_id}",
          **({} if routes is None else {"routes": routes})}


def test_manifest_rows_release_backing_uop_before_real_ledger_cleanup():
  ledger = PhysicalMemoryLedger(("CPU",))
  def retained_manifest():
    backing = UOp.new_buffer("CPU", 17, dtypes.uint8)
    ledger.bind_uop_owner(backing, AllocationOwner("schedule_arena", "schedule", semantic_owner_id="manifest:0:arena"))
    buffer = backing.buffer
    buffer.allocate()
    manifest = ScheduleMemoryManifest((), (ScheduleMemoryArena("arena:CPU:compute", "CPU", 0, 17, backing),), (), 0)
    return [manifest], weakref.ref(buffer)
  with ledger.active():
    manifests, buffer_ref = retained_manifest()
    rows = _drain_manifest_rows(manifests)
    gc.collect()
    assert manifests == [] and rows[0]["arenas"][0]["physical_owner_id"] is not None
    assert buffer_ref() is None
  assert [(event.event, event.owner.kind if event.owner else None) for event in ledger.events] == [
    ("alloc", "schedule_arena"), ("free", "schedule_arena")]
  assert ledger.complete, ledger.issues


def test_candidates_are_structural_complete_and_name_free():
  seam = TinygradWholeModelSeam()
  specs = seam.enumerate_candidate_specs(_model(), SimpleNamespace(backend="AMD",
    capabilities=SimpleNamespace(global_allocation_granularity=64)))
  assert [x.strategy for x in specs] == [Strategy.DIRECT_PACKED_FALLBACK, Strategy.FULL_RESIDENT_OVERLAY]
  assert all(x.covered_invocations == ("i0", "i1", "lm") for x in specs)
  encoded = json.dumps([x.policy for x in specs])
  assert all(word not in encoded for word in ("model_path", "profile", "size_label"))
  assert specs[1].memory_terms[0].bytes == (4*8 + 8*4)*2
  for spec in specs:
    assert spec.policy["routes"]["lm"] == "fixed-ggml-linear"
    capability = spec.kernel_capability()
    assert capability.full_m_values == (512,)
    assert [(x.logical_m, x.physical_m, x.minimum_prompt_tokens) for x in capability.remainder_mappings] == [(32, 512, 544)]
    assert capability.correctness_m_values == (512,)
    assert capability.invocation_bytes[0].m == 512
    assert capability.invocation_bytes[0].activation_bytes == 512 * (8 + 4) * 2


def test_pre_run_terms_cover_runtime_geometry_and_graph_without_names_or_tiers():
  kv = {"general.architecture": "arch", "arch.block_count": 3, "arch.embedding_length": 16,
        "arch.attention.head_count": 4, "arch.attention.head_count_kv": 2,
        "arch.attention.key_length": 4, "arch.rope.dimension_count": 2,
        "general.name": "display-only 14B label"}
  terms = _selected_runtime_base_terms(kv, backing_bytes=1000, max_context=10, alignment=64)
  assert [term.bytes for term in terms] == [1000, 3*320, 128]
  assert "14B" not in json.dumps([term.to_dict() for term in terms])
  rows = ({"shape": {"m": 3, "n": 5, "k": 7}, "tensor_identity": "renamable"},
          {"shape": {"m": 1, "n": 2, "k": 3}, "tensor_identity": "also-renamable"})
  assert _route_graph_materialization_bound(rows, 16) == (48+32)+(16+16)
  renamed = tuple({**row, "tensor_identity": "different"} for row in rows)
  assert _route_graph_materialization_bound(renamed, 16) == _route_graph_materialization_bound(rows, 16)


def test_missing_artifacts_are_explicit_failures_never_passes():
  out = _failed_artifacts(("i0",), ["missing required artifact: resource"])
  assert out["resource"]["status"] == "FAIL"
  assert out["route_census"]["complete"] is False
  assert out["end_to_end_timing"]["samples"] == []
  assert "PASS" not in json.dumps(out)


def test_worker_artifact_json_round_trips_every_typed_semantic_owner():
  owners = [MemorySemanticOwner(semantic_class) for semantic_class in MemorySemanticClass
            if semantic_class is not MemorySemanticClass.CANDIDATE_WORKSPACE]
  owners.append(MemorySemanticOwner(MemorySemanticClass.CANDIDATE_WORKSPACE, "structural:tile-8"))
  row = {"schedule_manifests": [{"buffers": [{"semantic_owner": owner} for owner in owners]}]}
  wire = json.dumps(_transport_encode(row), sort_keys=True, separators=(",", ":"))
  decoded = _validate_transport(json.loads(wire))
  assert [(x["semantic_owner"]["semantic_class"], x["semantic_owner"]["candidate_id"]) for x in decoded["schedule_manifests"][0]["buffers"]] == [
    (semantic_class.value, None) for semantic_class in MemorySemanticClass
    if semantic_class is not MemorySemanticClass.CANDIDATE_WORKSPACE] + [
    ("candidate_workspace", "structural:tile-8")]
  with pytest.raises(TypeError, match="unknown artifact transport object"):
    _transport_encode(object())
  with pytest.raises(ValueError, match="invalid semantic owner transport"):
    _validate_transport({"semantic_class": "unknown", "candidate_id": None})


def test_worker_artifact_json_preserves_schedule_evidence_semantic_totals():
  row = {"schedule_evidence": [{"peak_by_semantic_class": [
    {"semantic_class": "candidate_workspace", "candidate_id": "structural:tile-8", "physical_bytes": 256},
    {"semantic_class": "prefill_activation", "candidate_id": None, "physical_bytes": 512},
  ]}]}
  decoded = _validate_transport(json.loads(json.dumps(_transport_encode(row))))
  assert decoded == row
  with pytest.raises(ValueError, match="invalid semantic owner transport"):
    _validate_transport({"semantic_class": "unknown", "candidate_id": None, "physical_bytes": 1})


def test_worker_artifact_transport_preserves_unknown_owner_only_for_fail_closed_reconciliation():
  row = {"schedule_manifests": [{"buffers": [{"identity": "buffer:x", "semantic_owner": "unknown"}]}]}
  assert _validate_transport(json.loads(json.dumps(_transport_encode(row)))) == row
  with pytest.raises(ValueError, match="semantic_owner transport"):
    _validate_transport({"semantic_owner": "prefill_scratch"})


def test_worker_blocker_downgrades_partial_passing_resource_evidence():
  partial = {"resource": {"status": "PASS", "vgpr_count": 91, "complete": True},
             "route_census": {"status": "PASS", "complete": True}}
  out = _failed_artifacts(("i0",), ["allocation evidence incomplete"], partial)
  assert out["resource"] == {"status": "FAIL", "vgpr_count": 91, "complete": False,
                              "blocker": "allocation evidence incomplete"}
  # Independent evidence remains available for diagnosis; a worker blocker only makes the guarded resource gate fail closed.
  assert out["route_census"] == partial["route_census"]


def _physical_schedule_row(owner="manifest:0:arena:CPU:compute:backing:test", physical=256):
  return {"schema": "tinygrad.physical_memory_ledger.v1", "complete": True, "blockers": [],
    "peak_physical_bytes": physical, "peak_physical_bytes_per_device": {"CPU": physical}, "lifetimes": [{
      "allocation_id": 1, "physical_base_id": 1, "device": "CPU", "alloc_sequence": 1, "free_sequence": 2,
      "requested_nbytes": 129, "physical_nbytes": physical, "mapped": False,
      "owner": {"kind": "schedule_arena", "lifetime": "schedule", "candidate_id": None,
                "semantic_owner_id": owner}}]}


def _manifest(owner="prefill_activation"):
  return ({"peak_physical_bytes": 129, "arenas": [{"identity": "arena:CPU:compute", "device": "CPU", "lane": 0,
    "size": 129, "shared_rewritten_backing": True,
    "physical_owner_id": "manifest:0:arena:CPU:compute:backing:test"}], "buffers": [{"identity": "b", "device": "CPU",
    "arena_identity": "arena:CPU:compute", "byte_range": [0, 129], "first_index": 0, "last_index": 0,
    "semantic_owner": owner}]},
    {"complete": owner != "unknown", "blockers": [] if owner != "unknown" else [
      "buffer 'b' has unknown or malformed ownership: unknown semantic class 'unknown'"],
     "peak_physical_bytes": 129, "peak_by_semantic_class": [], "indices": []})


def test_exact_authorities_reconcile_shared_arena_without_semantic_class_on_physical_owner():
  manifest, evidence = _manifest()
  out = _reconcile_memory_authorities(_physical_schedule_row(), [manifest], [evidence],
    {"schema": CHECKPOINT_OBSERVER_SCHEMA, "complete": True, "peak_growth_bytes": 129,
     "post_run_retained_bytes": 0, "blockers": []}, selected_device="CPU",
    granularity=256, free_vram_bytes=4096, planned_peak_bytes=512)
  assert out["complete"] and out["peak_bytes"] == 256
  assert out["allocations"][0]["owner"] == {"kind": "schedule_arena", "lifetime": "schedule",
    "candidate_id": None, "semantic_owner_id": "manifest:0:arena:CPU:compute:backing:test"}


def test_reconciliation_fails_closed_on_unknown_schedule_semantics_and_mismatch():
  manifest, evidence = _manifest("unknown")
  out = _reconcile_memory_authorities(_physical_schedule_row(owner="manifest:0:arena:CPU:other:backing:test"), [manifest], [evidence],
    {"schema": CHECKPOINT_OBSERVER_SCHEMA, "complete": True, "peak_growth_bytes": 3,
     "post_run_retained_bytes": 1, "blockers": []}, selected_device="CPU",
    granularity=256, free_vram_bytes=4096, planned_peak_bytes=128)
  assert not out["complete"]
  assert any("unknown or malformed ownership" in x for x in out["blockers"])
  assert any("physical/manifest mismatch" in x for x in out["blockers"])
  assert any("exceeds planned peak" in x for x in out["blockers"])
  assert any("cleanup" in x for x in out["blockers"])


def test_collection_preserves_real_partial_run_and_precise_blockers(monkeypatch):
  seam = TinygradWholeModelSeam()
  candidate = AutoscanCandidate(CandidateMemoryCoverage("direct", Strategy.DIRECT_PACKED_FALLBACK, (), ("i0",), ("i0",)),
                                _policy("direct", {"i0": "direct"}))
  worker = {"actual_whole_model_run": True, "blockers": ["missing required artifact: final resource"],
            "whole_policy_identity": "identity:direct", "artifacts": {"whole_policy_identity": "identity:direct",
              "route_census": {"whole_policy_identity": "identity:direct"},
              "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [1, 2, 3]}}}
  monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(worker)+"\n", stderr=""))
  out = seam.collect_whole_model_artifacts("chosen.gguf", _model(), candidate, samples=3)
  assert out["actual_whole_model_run"] is True
  assert out["artifacts"]["end_to_end_timing"]["samples"] == [1, 2, 3]
  assert out["artifacts"]["resource"]["status"] == "FAIL"
  assert "missing required artifact: execution" in out["blockers"]


def test_worker_launch_isolated_and_candidate_binding_is_exact(monkeypatch):
  seam = TinygradWholeModelSeam(python="python-test")
  candidate = AutoscanCandidate(CandidateMemoryCoverage("direct", Strategy.DIRECT_PACKED_FALLBACK, (), ("i0",), ("i0",)),
                                {**_policy("direct", {"i0": "route.structural"}),
                                 "workload_choice": {"remainder_m": 32, "remainder_physical_m": 512,
                                                     "total_call_count": 1}})
  seen = []
  def run(cmd, **kwargs):
    request = json.loads(kwargs["input"])
    seen.append({"cmd": cmd, "request": request, "timeout": kwargs["timeout"],
                 "profile": kwargs["env"]["PROFILE"]})
    artifacts = _failed_artifacts(("i0",), ["resource unavailable"])
    artifacts["whole_policy_identity"] = request["whole_policy_identity"]
    artifacts["route_census"]["whole_policy_identity"] = request["whole_policy_identity"]
    return SimpleNamespace(returncode=0, stdout=json.dumps({"actual_whole_model_run": True,
      "whole_policy_identity": request["whole_policy_identity"],
      "blockers": ["resource unavailable"], "artifacts": artifacts})+"\n", stderr="")
  monkeypatch.setattr("subprocess.run", run)
  seam.collect_whole_model_artifacts("chosen.gguf", _model(), candidate, samples=3)
  assert len(seen) == 2 and [x["profile"] for x in seen] == ["0", "0"]
  assert [x["request"]["lifecycle_phase"] for x in seen] == ["evidence", "timing"]
  assert seen[0]["cmd"][-1] == "--worker" and seen[0]["cmd"][0] == "python-test"
  assert seen[0]["request"]["routes"] == {"i0": "route.structural"}
  assert seen[0]["request"]["whole_policy_identity"] == "identity:direct"
  assert seen[0]["request"]["samples"] == 3
  assert seen[0]["request"]["workload_choice"]["remainder_physical_m"] == 512
  assert seen[0]["request"]["planned_peak_bytes"] == 1


def test_collection_preserves_boundary_compatible_allocation_and_fails_closed(monkeypatch):
  seam = TinygradWholeModelSeam()
  candidate = AutoscanCandidate(CandidateMemoryCoverage("direct", Strategy.DIRECT_PACKED_FALLBACK, (), ("i0",), ("i0",)),
                                _policy("direct", {"i0": "direct"}))
  allocation = {"peak_bytes": 2, "planned_peak_bytes": 1, "allocations": [], "complete": False,
                "blockers": ["measured peak 2 exceeds planned peak 1"]}
  worker = _worker_row([[1], [1], [1]])
  worker["measured_allocation"] = allocation
  worker["artifacts"]["resource"]["measured_allocation"] = allocation
  monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(worker)+"\n", stderr=""))
  out = seam.collect_whole_model_artifacts("chosen.gguf", _model(), candidate, samples=3)
  assert out["measured_allocation"] == allocation
  assert out["artifacts"]["resource"]["measured_allocation"] == allocation
  assert "measured peak 2 exceeds planned peak 1" in out["blockers"]


@pytest.mark.parametrize("worker_identity", [None, "identity:other"])
def test_parent_rejects_missing_or_mismatched_worker_policy_identity_before_evidence(monkeypatch, worker_identity):
  seam = TinygradWholeModelSeam()
  candidate = AutoscanCandidate(CandidateMemoryCoverage("direct", Strategy.DIRECT_PACKED_FALLBACK, (), ("i0",), ("i0",)),
                                _policy("direct", {"i0": "direct"}))
  worker = _worker_row([[1], [1], [1]], worker_identity)
  monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(worker)+"\n", stderr=""))
  out = seam.collect_whole_model_artifacts("chosen.gguf", _model(), candidate, samples=3)
  assert out["actual_whole_model_run"] is False
  assert any("whole_policy_identity missing or mismatched" in blocker for blocker in out["blockers"])
  assert seam._baselines == {}
  assert all(phase["status"] == "failed" for phase in out["artifacts"]["execution"]["phases"])
  assert out["artifacts"]["resource"]["status"] == "FAIL"
  assert out["artifacts"]["route_census"]["status"] == "FAIL"


def _worker_row(outputs, identity="identity:direct"):
  return {"actual_whole_model_run": True, "whole_policy_identity": identity,
    "blockers": ["awaiting parent baseline comparison",
    "missing required artifact: exact runtime route census for every selected invocation"], "run": {
      "whole_policy_identity": identity,
      "deterministic_output_evidence": {"outputs": outputs, "input_digest_before": "digest", "input_digest_after": "digest"}},
    "artifacts": {"whole_policy_identity": identity, "execution": {"phases": [
      {"phase": "compile", "status": "passed", "evidence": {}},
      {"phase": "execution", "status": "passed", "evidence": {"dispatch_state": "completed", "health": {}}},
      {"phase": "correctness", "status": "failed", "evidence": {}}]},
      "resource": {"status": "PASS"}, "route_census": {"status": "FAIL", "complete": False,
        "whole_policy_identity": identity},
      "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": [1, 1, 1]}}}


def test_parent_retains_baseline_and_authorizes_exact_greedy_outputs(monkeypatch):
  seam = TinygradWholeModelSeam()
  baseline = AutoscanCandidate(CandidateMemoryCoverage("base", Strategy.DIRECT_PACKED_FALLBACK, (), ("i0",), ("i0",)),
                               _policy("base", {}))
  candidate = AutoscanCandidate(CandidateMemoryCoverage("candidate", Strategy.FULL_RESIDENT_OVERLAY, (), ("i0",), ("i0",)),
                                _policy("candidate", {}))
  rows = iter((_worker_row([[4, 5], [4, 5], [4, 5]], "identity:base"),
               _worker_row([[4, 5], [4, 5], [4, 5]], "identity:base"),
               _worker_row([[4, 5], [4, 5], [4, 5]], "identity:candidate"),
               _worker_row([[4, 5], [4, 5], [4, 5]], "identity:candidate")))
  monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(next(rows))+"\n", stderr=""))
  seam.collect_whole_model_artifacts("chosen.gguf", _model(), baseline, samples=3)
  out = seam.collect_whole_model_artifacts("chosen.gguf", _model(), candidate, samples=3)
  correctness = next(x for x in out["artifacts"]["execution"]["phases"] if x["phase"] == "correctness")
  assert correctness["status"] == "passed"
  assert correctness["evidence"]["comparison"] | {"atol": 0.0, "rtol": 0.0, "greedy_equal": True} == correctness["evidence"]["comparison"]


def test_parent_rejects_candidate_output_mismatch_and_input_mutation():
  seam = TinygradWholeModelSeam(); model = _model()
  baseline = AutoscanCandidate(CandidateMemoryCoverage("base", Strategy.DIRECT_PACKED_FALLBACK, (), ("i0",), ("i0",)),
                               _policy("base"))
  candidate = AutoscanCandidate(CandidateMemoryCoverage("candidate", Strategy.FULL_RESIDENT_OVERLAY, (), ("i0",), ("i0",)),
                                _policy("candidate"))
  seam._authorize_correctness(model, baseline, _worker_row([[1], [1], [1]]))
  artifacts, errors = seam._authorize_correctness(model, candidate, _worker_row([[2], [2], [2]]))
  assert "candidate greedy output differs" in errors[0]
  mutated = _worker_row([[1], [1], [1]]); mutated["run"]["deterministic_output_evidence"]["input_digest_after"] = "changed"
  _, errors = seam._authorize_correctness(model, candidate, mutated)
  assert errors == ["guarded input changed during whole-model execution"]


def test_profile_code_objects_are_deduplicated_and_resource_violations_fail(monkeypatch):
  from tinygrad.device import Compiled, ProfileProgramEvent
  start = len(Compiled.profile_events)
  Compiled.profile_events.extend((ProfileProgramEvent("AMD", "kernel", b"elf", None),
                                  ProfileProgramEvent("AMD", "kernel duplicate", b"elf", None)))
  metadata = {"vgpr": 12, "sgpr": 8, "lds_bytes": 256, "scratch_bytes": 0,
              "vgpr_spills": 0, "sgpr_spills": 0, "dynamic_stack": False}
  monkeypatch.setattr("extra.qk.amdgpu_metadata.parse_amdgpu_metadata", lambda binary: dict(metadata))
  artifact, failures = _compiled_resource_artifact(start, "AMD")
  assert failures == [] and artifact["status"] == "PASS" and artifact["program_count"] == 1
  monkeypatch.setattr("extra.qk.amdgpu_metadata.parse_amdgpu_metadata",
                      lambda binary: {**metadata, "scratch_bytes": 16})
  artifact, failures = _compiled_resource_artifact(start, "AMD")
  assert artifact["status"] == "FAIL" and any("scratch_bytes=16" in x for x in failures)
  del Compiled.profile_events[start:]
def test_manifest_binds_known_dedicated_semantic_to_physical_uop():
  ledger = PhysicalMemoryLedger(("CPU",))
  backing = UOp.new_buffer("CPU", 17, dtypes.uint8)
  row = ScheduleMemoryBuffer("buffer:x", "CPU", 0, 256, 0, 0, "dedicated:buffer:x", 256, 0, (0, 256),
                             MemorySemanticOwner(MemorySemanticClass.RUNTIME_SCRATCH))
  manifest = ScheduleMemoryManifest((row,), (ScheduleMemoryArena("dedicated:buffer:x", "CPU", 0, 256, backing),), (), 256)
  _bind_manifest_physical_owners(ledger, manifest, 0)
  with ledger.active(): backing.buffer.allocate(); backing.buffer.deallocate()
  event = next(x for x in ledger.events if x.event == "alloc")
  assert event.owner == AllocationOwner("schedule_transient", "schedule")
