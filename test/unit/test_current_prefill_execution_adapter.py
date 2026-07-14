import hashlib, json, pickle

import numpy as np
import pytest

from extra.qk.prefill import current_prefill_execution_adapter as adapter
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry
from extra.qk.route_manifest import promoted_prefill_candidate_policy
from tinygrad.runtime.execution_bridge_contracts import ExecutionRequest, TransportPlan
from tinygrad.uop.ops import Ops


def _candidate():
  row = json.loads(open(promoted_prefill_candidate_policy()["candidate_set_path"]).read())
  return next(entry for entry in row["entries"] if entry["payload"]["workload"]["role"] == "attn_qo")


def _request(entry, **compiler_changes):
  context = {"adapter_id": adapter.ADAPTER_ID, "candidate_payload": entry["payload"],
             "canonical_identity": entry["canonical_identity"], "input_npz": "/unused/in/unit/test.npz"}
  context.update(compiler_changes)
  return ExecutionRequest("exp", "candidate", "baseline", "w" * 64, "s" * 64,
    TransportPlan("lds", "s" * 64), {"provider": "tinygrad", "target": "AMD:gfx1100"}, context)


def test_adapter_prepares_exact_promoted_attn_qo_as_spawn_safe_bundle_without_gpu(monkeypatch):
  entry = _candidate()
  arrays = ({"a": np.ones((2, 2), dtype=np.float16), "b": np.ones((2, 2), dtype=np.float16)},
            np.ones((2, 2), dtype=np.float16))
  monkeypatch.setattr(adapter, "_arrays", lambda *_: arrays)
  monkeypatch.setattr(adapter, "_input_artifact_identities", lambda *_: {"input_artifact_sha256": "i" * 64,
                                                                        "reference_sha256": "r" * 64})
  compile_calls = []
  def fake_compile(payload, identity, *, device):
    compile_calls.append((payload, identity, device))
    return object(), {"schema": "prefill-transport-compile.v1", "passed": True,
                      "binary_sha256": "b" * 64, "canonical_identity": identity}

  prepared = adapter.CurrentPrefillAdapter(compile_prepare=fake_compile).prepare(_request(entry))
  assert compile_calls == [(entry["payload"], entry["canonical_identity"], "AMD")]
  assert prepared.compile_evidence["canonical_identity"] == entry["canonical_identity"]
  assert prepared.output_dtype == np.float16
  assert prepared.health_probe is not None and prepared.health_probe.device == "AMD"
  restored = pickle.loads(pickle.dumps(prepared.builder))
  assert restored.build is adapter.build_current_prefill_bundle
  assert restored.kwargs["canonical_identity"] == entry["canonical_identity"]
  assert restored.kwargs["compile_device"] == "AMD"
  assert restored.kwargs["runtime_device"] == "AMD"


def test_adapter_reuses_admission_and_rejects_identity_or_typed_transport_drift(monkeypatch):
  entry = _candidate()
  monkeypatch.setattr(adapter, "_arrays", lambda *_: ({"a": np.ones(2)}, np.ones(2)))
  monkeypatch.setattr(adapter, "_input_artifact_identities", lambda *_: {})
  instance = adapter.CurrentPrefillAdapter(compile_prepare=lambda *_args, **_kwargs: (None, {"passed": True}))
  with pytest.raises(ValueError, match="canonical SHA-256"):
    instance.prepare(_request(entry, canonical_identity="0" * 64))
  request = _request(entry)
  request = ExecutionRequest(request.experiment_id, request.candidate_id, request.comparator_id,
    request.workload_digest, request.schedule_digest, TransportPlan("direct_l2", request.schedule_digest),
    request.target_context, request.compiler_context)
  with pytest.raises(ValueError, match="does not match admitted"):
    instance.prepare(request)


def test_registration_is_explicit_and_has_no_route_name_fallback():
  registry = AdapterRegistry()
  assert registry.ids() == ()
  adapter.register_current_prefill_adapter(registry)
  assert registry.ids() == (adapter.ADAPTER_ID,)
  assert registry.resolve("prefill_wmma_lds_dbuf_generated") is None


def test_spawn_child_requires_contract_and_rejects_normalized_identity_drift_before_bundle(monkeypatch):
  entry = _candidate()
  base = {"passed": True, "canonical_identity": entry["canonical_identity"], "source_sha256": "a" * 64,
          "binary_sha256": "b" * 64, "target": "gfx1100", "compile_target": "AMD"}
  with pytest.raises(ValueError, match="contract is missing"):
    adapter.build_current_prefill_bundle(payload=entry["payload"], canonical_identity=entry["canonical_identity"],
                                         compile_evidence=base)
  base["child_recompile_binary_identity_contract"] = {"enabled": True,
    "reject_sha256_mismatch_before_dispatch": True, "canonical_identity": entry["canonical_identity"],
    "source_sha256": "a" * 64, "binary_sha256": "b" * 64, "target": "gfx1100",
    "compile_target": "AMD"}
  monkeypatch.setattr(adapter, "prepare_current_prefill_compile", lambda *_args, **_kwargs:
    (object(), {**base, "source_sha256": "c" * 64}))
  monkeypatch.setattr(adapter, "build_tinygrad_bundle", lambda **_kwargs: pytest.fail("bundle must not be constructed"))
  with pytest.raises(ValueError, match="source_sha256 differs"):
    adapter.build_current_prefill_bundle(payload=entry["payload"], canonical_identity=entry["canonical_identity"],
                                         compile_evidence=base)


def test_input_identity_hashes_exact_npz_and_loaded_reference(tmp_path):
  path = tmp_path / "inputs.npz"
  reference = np.arange(6, dtype=np.float16).reshape(2, 3)
  np.savez(path, a=np.ones((2, 2), dtype=np.float16), b=np.ones((3, 2), dtype=np.float16), reference=reference)
  identity = adapter._input_artifact_identities(str(path), reference)
  assert identity["input_artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
  assert identity["reference_sha256"] == hashlib.sha256(b"<f2:2,3:" + reference.tobytes()).hexdigest()


def test_promoted_attn_qo_compile_only_produces_one_bound_final_program_and_proof_resources():
  entry = _candidate()
  program, evidence = adapter.prepare_current_prefill_compile(
    entry["payload"], entry["canonical_identity"], device="AMD")

  assert program.op is Ops.PROGRAM
  assert getattr(program.src[0].arg.candidate_context, "canonical_identity", None) == entry["canonical_identity"]
  assert evidence["passed"] is True
  assert evidence["canonical_identity"] == entry["canonical_identity"]
  assert evidence["capture"] == {"mode": "compile_only", "dispatch_permitted": False,
                                  "resource_authority": "compiled_program_descriptor"}
  assert evidence["final_isa_manifest"]["candidate_id"] == entry["canonical_identity"]
  assert evidence["final_isa_manifest"]["abi_metadata"]["argument_order"] == ["output", "a", "b"]
  assert evidence["final_isa_manifest"]["ownership_metadata"]["semantic_operands"][1]["operand_id"] == "A"
  assert evidence["resource_summary"]["canonical_identity"] == entry["canonical_identity"]
  assert evidence["resource_summary"]["lds_bytes"] == evidence["resource_summary"]["admitted_active_lds_bytes"] == 40960
  assert evidence["resource_summary"]["vgpr"] == 188
  assert evidence["resource_summary"]["allocated_vgpr"] == 248
  assert evidence["resource_summary"]["sgpr"] == 18
  json.dumps(evidence)
  assert evidence["isa_structure"]["counts"]["wmma"] == 32
  assert evidence["isa_structure"]["counts"]["ds_load"] == 48
  # Semantic operand ownership is now derived from the exact shipping code object via ABI dataflow;
  # the clean global-load->LDS-stage A/B flow is attributed and the double-buffered LDS windows stay
  # explicit unknown with a named discriminator (no route-name or alternate-binary inference).
  st = evidence["isa_structure"]
  assert st["operand_ownership_authority"] == "abi_dataflow_v1"
  assert st["operand_ownership_binary_sha256"] == evidence["binary_sha256"]
  paths = st["operand_paths"]
  assert paths and all(p["binary_sha256"] == evidence["binary_sha256"] for p in paths)
  attributed = [p for p in paths if p["operand_id"] != "unknown"]
  assert st["attributed_row_count"] == len(attributed) >= 16   # 8 global loads + 8 ds stores at minimum
  gl = [p for p in paths if p["kind"] == "global_load" and p["operand_id"] in ("A", "B")]
  assert sum(p["operand_id"] == "A" for p in gl) == 4 and sum(p["operand_id"] == "B" for p in gl) == 4
  assert all(p["source_operands"] == ["A", "B"] for p in paths if p["kind"] == "wmma" and "source_operands" in p)
  assert all("missing" in p for p in paths if p["operand_id"] == "unknown")
  assert "double_buffered_lds_window_binding" in st["missing_evidence"]
  assert evidence["artifacts"]["final_isa_manifest"]["status"] == "partial"
  assert evidence["artifacts"]["final_isa_manifest"]["attributed_rows"] == st["attributed_row_count"]
