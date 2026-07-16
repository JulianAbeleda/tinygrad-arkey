import copy, hashlib, json

import pytest

from extra.qk.prefill import candidate_inventory_execution as driver
from extra.qk.prefill.workload_inventory import CANDIDATE_INVENTORY_SCHEMA, INVENTORY_SCHEMA
from extra.qk.runtime_specs import FULL_KERNEL_CANDIDATE_SET_SCHEMA, FullKernelCandidateSet
from extra.qk.prefill.execution_bridge_contracts import ExecutionResult, PhaseResult


def _entry(profile, role, quant, shape, marker):
  payload = {"workload": {"profile": profile, "role": role,
    "shape": dict(zip(("m", "n", "k"), shape)), "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}},
    "schedule": {"marker": marker, "pipeline": {"buffer_count": 1, "stage_count": 1}},
    "operand_sources": {"b": {"quant_format": quant}}}
  identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  return {"canonical_identity": identity, "payload": payload}


def artifact(profile="generic_profile_not_in_registry"):
  specs = [("ffn_down", "Q6_K", (7, 16, 256), "z"),
           ("attn_qo", "Q4_K", (3, 32, 256), "a"),
           ("attn_qo", "Q6_K", (5, 48, 256), "b")]
  rows, bindings, sets = [], [], {}
  for role, quant, shape, marker in specs:
    entry = _entry(profile, role, quant, shape, marker)
    key = [role, quant, *shape]
    rows.append({"role": role, "quant_format": quant, "shape": dict(zip(("m", "n", "k"), shape)),
                 "tensor_identities": [f"tensor.{marker}"]})
    bindings.append({"inventory_key": key, "canonical_identity": entry["canonical_identity"]})
    sets.setdefault(quant, {"schema": FULL_KERNEL_CANDIDATE_SET_SCHEMA, "entries": []})["entries"].append(entry)
  return {"schema": CANDIDATE_INVENTORY_SCHEMA,
    "inventory": {"schema": INVENTORY_SCHEMA, "profile": profile, "rows": rows},
    "candidate_sets": sets, "bindings": bindings}


def test_join_uses_inventory_order_not_partition_or_role_sort_order_and_supports_generic_profile():
  value = artifact()
  joined = driver.validate_and_join(value)
  assert [(x.role, x.quant_format, x.shape) for x in joined] == [
    ("ffn_down", "Q6_K", (7, 16, 256)), ("attn_qo", "Q4_K", (3, 32, 256)),
    ("attn_qo", "Q6_K", (5, 48, 256))]
  assert all(x.payload["workload"]["profile"] == "generic_profile_not_in_registry" for x in joined)
  entries = {x.legacy_identity_alias: x.canonical_identity for raw_set in value["candidate_sets"].values()
             for x in FullKernelCandidateSet.from_json(raw_set).entries}
  assert all(x.canonical_identity == entries[binding["canonical_identity"]]
             for x, binding in zip(joined, value["bindings"]))


def test_legacy_binding_alias_is_canonicalized_in_request_and_execution_evidence(tmp_path):
  value = artifact()
  entry = FullKernelCandidateSet.from_json(value["candidate_sets"]["Q6_K"]).entries[0]
  assert value["bindings"][0]["canonical_identity"] == entry.legacy_identity_alias
  assert entry.legacy_identity_alias != entry.canonical_identity
  joined = driver.validate_and_join(value)
  assert joined[0].canonical_identity == entry.canonical_identity
  request = driver.make_request(joined[0], "input.npz", phase="compile-only")
  assert request.candidate_id == request.comparator_id == entry.canonical_identity
  assert request.compiler_context["canonical_identity"] == entry.canonical_identity
  out = driver.run_inventory(value, phase="compile-only", artifact_dir=str(tmp_path),
    prepare_fn=lambda payload, identity, *, device: (None, {"canonical_identity": identity}))
  row = out["results"][0]
  assert row["identity"]["canonical_identity"] == entry.canonical_identity
  assert row["request"]["candidate_id"] == entry.canonical_identity


def test_filters_preserve_inventory_order_and_reject_unknown_values():
  joined = driver.validate_and_join(artifact())
  assert [x.quant_format for x in driver.select_candidates(joined, roles=["attn_qo"])] == ["Q4_K", "Q6_K"]
  assert [x.role for x in driver.select_candidates(joined, quant_formats=["Q6_K"])] == ["ffn_down", "attn_qo"]
  with pytest.raises(ValueError, match="unknown role filters"): driver.select_candidates(joined, roles=["model_layer_7"])


@pytest.mark.parametrize("mutation,match", [
  (lambda x: x["bindings"].pop(), "missing bindings"),
  (lambda x: x["bindings"].append(copy.deepcopy(x["bindings"][0])), "duplicate binding"),
  (lambda x: x["bindings"][0].update(canonical_identity="0" * 64), "canonical identity drift"),
  (lambda x: x["candidate_sets"]["Q6_K"]["entries"][0]["payload"]["operand_sources"]["b"].update(quant_format="Q4_K"), "identity_mismatch|identity differs"),
  (lambda x: x["candidate_sets"].update(UNKNOWN={"schema": FULL_KERNEL_CANDIDATE_SET_SCHEMA,
    "entries": [_entry("generic_profile_not_in_registry", "lm_head", "UNKNOWN", (1, 8, 256), "u")]}), "unknown candidate sets keys"),
])
def test_schema_and_identity_drift_fail_closed(mutation, match):
  value = artifact(); mutation(value)
  with pytest.raises(ValueError, match=match): driver.validate_and_join(value)


def test_unknown_top_level_field_is_rejected():
  value = artifact(); value["model_name"] = "must not become authority"
  with pytest.raises(ValueError, match="unknown or missing"): driver.validate_and_join(value)


def test_nested_inventory_schema_drift_is_rejected():
  value = artifact(); value["inventory"]["schema"] = "future.inventory.v2"
  with pytest.raises(ValueError, match="containers are malformed"): driver.validate_and_join(value)


def test_compile_only_prepares_serially_and_never_calls_dispatch(tmp_path):
  prepared, dispatched = [], []
  def prepare(payload, identity, *, device):
    prepared.append(identity)
    assert device == "AMD"
    return None, {"canonical_identity": identity}
  def dispatch(request):
    dispatched.append(request.candidate_id)
    raise AssertionError("compile-only dispatched")
  out = driver.run_inventory(artifact(), phase="compile-only", artifact_dir=str(tmp_path),
    prepare_fn=prepare, execute_fn=dispatch)
  assert len(prepared) == 3 and not dispatched and out["completed_count"] == 3
  assert all(row["identity"]["canonical_identity"] == row["request"]["candidate_id"] for row in out["results"])


def test_default_compile_only_is_independent_of_missing_npz(monkeypatch, tmp_path):
  calls = []
  def compile_prepare(payload, identity, *, device):
    calls.append((payload, identity, device))
    assert not any(tmp_path.iterdir())
    return object(), {"canonical_identity": identity, "compiled_device": device}
  monkeypatch.setattr(driver, "prepare_current_prefill_compile", compile_prepare)
  out = driver.run_inventory(artifact(), phase="compile-only", artifact_dir=str(tmp_path),
    roles=["ffn_down"])
  assert len(calls) == 1 and calls[0][2] == "AMD"
  assert out["results"][0]["status"] == "passed" and not any(tmp_path.iterdir())


def test_default_execution_registers_one_scoped_production_adapter(monkeypatch, tmp_path):
  registered, executed = [], []
  def register(registry): registered.append(registry)
  def execute(request, *, registry):
    executed.append((request.candidate_id, registry))
    return ExecutionResult(request.experiment_id, request.candidate_id, request.digest,
      (PhaseResult("correctness", "passed", evidence={"health": {"preflight": True, "postflight": True}}),))
  monkeypatch.setattr(driver, "register_current_prefill_adapter", register)
  monkeypatch.setattr(driver, "execute_request", execute)
  out = driver.run_inventory(artifact(), phase="correctness", artifact_dir=str(tmp_path))
  assert len(registered) == 1 and len(executed) == 3
  assert all(registry is registered[0] for _, registry in executed)
  assert out["completed_count"] == 3


def test_serial_correctness_failure_stops_before_next_candidate(tmp_path):
  calls = []
  def execute(request):
    calls.append(request.candidate_id)
    status = "passed" if len(calls) == 1 else "failed"
    return ExecutionResult(request.experiment_id, request.candidate_id, request.digest,
      (PhaseResult("correctness", status, evidence={"health": {"preflight": True, "postflight": status == "passed"}}),))
  out = driver.run_inventory(artifact(), phase="correctness", artifact_dir=str(tmp_path), execute_fn=execute)
  assert len(calls) == 2 and out["completed_count"] == 2 and out["selected_count"] == 3


def test_build_inputs_is_injected_ordered_and_identity_bound(tmp_path):
  calls = []
  def build(quant, path, shape):
    calls.append((quant, shape)); return {"quant_format": quant, "path": path, "shape": list(shape)}
  out = driver.run_inventory(artifact(), phase="build-input", artifact_dir=str(tmp_path),
    roles=["attn_qo"], build_fn=build)
  assert calls == [("Q4_K", (3, 32, 256)), ("Q6_K", (5, 48, 256))]
  assert [x["identity"]["inventory_key"][1] for x in out["results"]] == ["Q4_K", "Q6_K"]


def test_request_digests_change_with_workload_schedule_and_candidate_facts():
  one = driver.validate_and_join(artifact())[0]
  req = driver.make_request(one, "x.npz", phase="correctness")
  changed = copy.deepcopy(artifact())
  changed["inventory"]["rows"][0]["tensor_identities"] = ["different.tensor"]
  req2 = driver.make_request(driver.validate_and_join(changed)[0], "x.npz", phase="correctness")
  assert req.workload_digest != req2.workload_digest and req.experiment_id != req2.experiment_id
