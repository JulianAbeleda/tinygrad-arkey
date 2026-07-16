import copy, json
from pathlib import Path

import pytest

from extra.qk.prefill import candidate_inventory_execution as driver
from extra.qk.prefill.workload_inventory import generate_candidate_inventory
from extra.qk.prefill.execution_bridge_contracts import ExecutionResult, PhaseResult


def artifact(first_tensor="tensor.z"):
  specs = [("ffn_down", "Q6_K", (512, 5120, 17408), "z"),
           ("attn_qo", "Q4_K", (512, 5120, 5120), "a"),
           ("attn_qo", "Q6_K", (512, 5120, 5120), "b")]
  rows = []
  for role, quant, shape, marker in specs:
    block_bytes = 144 if quant == "Q4_K" else 210
    rows.append({"role":role, "quant_format":quant, "shape":dict(zip(("m", "n", "k"), shape)),
      "layout":{"logical":"transposed_row_major", "packed":"ggml_k_blocks", "block_elems":256,
                "block_bytes":block_bytes}, "tensor_identities":[f"tensor.{marker}"], "call_count":1,
      "source_bytes":shape[1] * shape[2] // 256 * block_bytes, "logical_flop":2 * shape[0] * shape[1] * shape[2],
      "memory_lifetime":"model_resident"})
  rows[0]["tensor_identities"] = [first_tensor]
  from extra.qk.prefill.workload_inventory import INVENTORY_SCHEMA, _canonical_inventory_identity
  inventory = {"schema":INVENTORY_SCHEMA, "rows":rows, "inventory_identity":_canonical_inventory_identity(rows)}
  raw = json.loads(Path("bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/candidate-set.json").read_text())
  templates = {x["payload"]["workload"]["role"]:x["payload"] for x in raw["entries"]}
  return generate_candidate_inventory(inventory, templates)


def test_join_uses_inventory_order_not_partition_or_role_sort_order():
  value = artifact()
  joined = driver.validate_and_join(value)
  assert [(x.role, x.quant_format, x.shape) for x in joined] == [
    ("ffn_down", "Q6_K", (512, 5120, 17408)), ("attn_qo", "Q4_K", (512, 5120, 5120)),
    ("attn_qo", "Q6_K", (512, 5120, 5120))]
  assert all(x.payload["workload"]["profile"] == value["inventory_identity"] for x in joined)


def test_legacy_binding_alias_is_rejected(tmp_path):
  value = artifact()
  entry = value["candidate_sets"]["Q6_K"]["entries"][0]
  from extra.qk.runtime_specs import FullKernelCandidateSetEntry
  parsed = FullKernelCandidateSetEntry(entry["canonical_identity"], entry["payload"])
  value["bindings"][0]["canonical_identity"] = parsed.legacy_identity_alias
  assert parsed.legacy_identity_alias != parsed.canonical_identity
  with pytest.raises(ValueError, match="canonical identity drift"): driver.validate_and_join(value)


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
  (lambda x: x["candidate_sets"].update(UNKNOWN=x["candidate_sets"]["Q6_K"]), "quant partition drift"),
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


def test_default_q4_compile_only_uses_five_buffer_target(monkeypatch, tmp_path):
  calls = []
  def compile_prepare(payload, identity, *, target):
    calls.append((payload, identity, target))
    assert not any(tmp_path.iterdir())
    return object(), {"canonical_identity": identity, "compile_target": target}
  monkeypatch.setattr(driver, "prepare_q4k_q8_five_buffer_compile", compile_prepare)
  out = driver.run_inventory(artifact(), phase="compile-only", artifact_dir=str(tmp_path),
    quant_formats=["Q4_K"])
  assert len(calls) == 1 and calls[0][2] == "AMD:ISA:gfx1100"
  assert out["results"][0]["status"] == "passed" and not any(tmp_path.iterdir())


def test_request_selects_adapter_from_admitted_abi():
  q4, q6 = driver.validate_and_join(artifact())[1:]
  q4_request = driver.make_request(q4, "q4.npz", phase="compile-only")
  q6_request = driver.make_request(q6, "q6.npz", phase="compile-only")
  assert q4_request.compiler_context["adapter_id"] == driver.FIVE_BUFFER_ADAPTER_ID
  assert q6_request.compiler_context["adapter_id"] == driver.ADAPTER_ID


def test_default_execution_registers_both_scoped_production_adapters(monkeypatch, tmp_path):
  registered, executed = [], []
  def register(registry): registered.append(registry)
  def execute(request, *, registry):
    executed.append((request.candidate_id, registry))
    return ExecutionResult(request.experiment_id, request.candidate_id, request.digest,
      (PhaseResult("correctness", "passed", evidence={"health": {"preflight": True, "postflight": True}}),))
  monkeypatch.setattr(driver, "register_current_prefill_adapter", register)
  monkeypatch.setattr(driver, "register_q4k_q8_five_buffer_adapter", register)
  monkeypatch.setattr(driver, "execute_request", execute)
  out = driver.run_inventory(artifact(), phase="correctness", artifact_dir=str(tmp_path))
  assert len(registered) == 2 and len(executed) == 3
  assert registered[0] is registered[1]
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
  assert calls == [("Q4_K", (512, 5120, 5120)), ("Q6_K", (512, 5120, 5120))]
  assert [x["identity"]["inventory_key"][1] for x in out["results"]] == ["Q4_K", "Q6_K"]


def test_request_digests_change_with_workload_schedule_and_candidate_facts():
  one = driver.validate_and_join(artifact())[0]
  req = driver.make_request(one, "x.npz", phase="correctness")
  changed = artifact("different.tensor")
  req2 = driver.make_request(driver.validate_and_join(changed)[0], "x.npz", phase="correctness")
  assert req.workload_digest != req2.workload_digest and req.experiment_id != req2.experiment_id
