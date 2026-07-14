import hashlib, json, pickle

import numpy as np
import pytest

from extra.qk.prefill import current_prefill_execution_adapter as adapter
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry
from extra.qk.route_manifest import promoted_prefill_candidate_policy
from extra.qk.runtime_specs import derive_packed_weight_candidate
from tinygrad.runtime.execution_bridge_contracts import ExecutionRequest, TransportPlan
from tinygrad.uop.ops import Ops


def _candidate():
  row = json.loads(open(promoted_prefill_candidate_policy()["candidate_set_path"]).read())
  return next(entry for entry in row["entries"] if entry["payload"]["workload"]["role"] == "attn_qo")


def _packed_candidate(quant_format):
  """Derive a strict packed-B candidate without borrowing an identity from the promoted dense payload."""
  entry = derive_packed_weight_candidate(_candidate()["payload"], quant_format)
  return entry.to_json()["payload"], entry.canonical_identity


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


@pytest.mark.parametrize("quant_format,np_dtype,units", (("Q4_K", np.uint32, 4096*16*36),
                                                          ("Q6_K", np.uint16, 4096*16*105)))
def test_packed_input_artifact_binds_exact_slot2_storage_contract(tmp_path, quant_format, np_dtype, units):
  payload, identity = _packed_candidate(quant_format)
  admission = adapter.admit_current_prefill(payload, identity)
  path = tmp_path / "packed.npz"
  np.savez(path, a=np.zeros((512,4096), np.float16), b=np.zeros(units, np_dtype),
           reference=np.zeros((512,4096), np.float16))
  inputs, _ = adapter._arrays(str(path), (512,4096,4096), admission.context.packed_weight)
  assert inputs["b"].dtype == np.dtype(np_dtype)
  assert inputs["b"].nbytes == admission.context.packed_weight.packed_bytes

  wrong = tmp_path / "wrong.npz"
  np.savez(wrong, a=np.zeros((512,4096), np.float16), b=np.zeros(1, np.float16),
           reference=np.zeros((512,4096), np.float16))
  with pytest.raises(ValueError, match="packed prefill B"):
    adapter._arrays(str(wrong), (512,4096,4096), admission.context.packed_weight)


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


@pytest.mark.parametrize("quant_format,storage_dtype,dtype_name,block_bytes,units_per_block", (
  ("Q4_K", "uint32", "unsigned int", 144, 36),
  ("Q6_K", "uint16", "unsigned short", 210, 105),
))
def test_packed_attn_qo_compile_only_is_one_fp16_wmma_program_with_packed_b_abi(
    quant_format, storage_dtype, dtype_name, block_bytes, units_per_block):
  payload, identity = _packed_candidate(quant_format)
  program, admission = adapter.compile_current_prefill_program(payload, identity, device="AMD:ISA:gfx1100")

  programs = [u for u in program.toposort() if u.op is Ops.PROGRAM]
  assert programs == [program], "packed decode must be fused, with no prerequisite dequant PROGRAM"
  assert getattr(program.src[0].arg.candidate_context, "canonical_identity", None) == identity
  assert (tuple(program.arg.globals), tuple(program.arg.outs), tuple(program.arg.ins)) == ((0, 1, 2), (0,), (1, 2))

  transform = admission.context.packed_weight
  assert transform is not None
  assert transform.to_json() == {
    "quant_format": quant_format, "rows": 4096, "k": 4096, "block_elems": 256,
    "block_bytes": block_bytes, "storage_dtype": dtype_name,
    "storage_width": 4 if quant_format == "Q4_K" else 2,
    "units_per_block": units_per_block,
    "packed_bytes": 4096 * (4096 // 256) * block_bytes,
  }

  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  assert "wmma_f32_16x16x16_f16" in source
  params = {u.arg.slot: u.dtype for u in program.src[0].toposort() if u.op is Ops.PARAM}
  assert set(params) == {0, 1, 2}
  assert params[0].base.name == params[1].base.name == "half"
  assert params[2].base.name == dtype_name


@pytest.mark.parametrize("quant_format,storage_dtype", (("Q4_K", "uint32"), ("Q6_K", "uint16")))
def test_packed_shipping_binary_passes_resource_and_representation_gate(quant_format, storage_dtype):
  payload, identity = _packed_candidate(quant_format)
  _, evidence = adapter.prepare_current_prefill_compile(payload, identity, device="AMD")
  assert evidence["packed_wmma_compile_gate"]["passed"] is True
  assert evidence["isa_structure"]["counts"]["wmma"] == 32
  assert evidence["resource_summary"]["lds_bytes"] == 40960
  assert evidence["resource_summary"]["workgroup_threads"] == 256
  assert all(evidence["resource_summary"][field] == 0 for field in ("scratch_bytes", "vgpr_spills", "sgpr_spills"))
  packed_b = evidence["final_isa_manifest"]["ownership_metadata"]["semantic_operands"][2]
  assert packed_b["representation"] == "packed_scalar_decoder"
  assert (packed_b["quant_format"], packed_b["storage_dtype"]) == (quant_format, storage_dtype)
