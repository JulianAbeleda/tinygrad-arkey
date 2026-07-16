import hashlib
import numpy as np
import pytest

from extra.qk.prefill import q4k_q8_five_buffer_execution_adapter as adapter
from extra.qk.prefill.execution_bridge_contracts import ExecutionRequest, TransportPlan
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import admitted_buffer_descriptors
from extra.qk.runtime_specs import FULL_KERNEL_CANDIDATE_SCHEMA, derive_q4k_q8_1_five_buffer_candidate


def _entry(shape=(16, 16, 256), role="ffn_gate_up"):
  m, n, k = shape
  payload = {"schema_version": FULL_KERNEL_CANDIDATE_SCHEMA,
    "workload":{"profile":"execution_adapter_test", "role":role, "shape":{"m":m,"n":n,"k":k},
      "dtypes":{"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"},
      "layout":{"a":"row_major","b":"transposed_row_major","c":"row_major"},
      "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}},
    "schedule":{"tile":{"m":128,"n":128,"k":32},"waves":{"m":4,"n":2},"threads":256,
      "lane_ownership":"rdna3_wmma_f32_16x16x16_f16_lds2_static",
      "cooperative_load":{x:{"lane_mapping":"cooperative_row_stride_64_b128","vector_width":8,"alignment":16} for x in ("a","b")},
      "lds":{"windows":{"a":[0,10240],"b":[10240,20480]},"strides":{"a":80,"b":80},
        "padding":16,"banks":32,"store_vector_width":8,"load_vector_width":8},
      "pipeline":{"buffer_count":1,"stage_count":1,"epoch_graph":[]},
      "wmma":{"instruction_family":"wmma_f32_16x16x16_f16","fragment_layout":"rdna3_wmma_f32_16x16x16_f16_lds2_static",
        "accumulator_ownership":"wmma_accum_wm_x_wn_8_vgprs"},
      "dependency_policy":{"waitcnt":{"vm":0,"lgkm":0},"barriers":[]},
      "residency":{"preload":["a","b"],"resident":["accumulator"],"reuse":{"a":4,"b":2}},
      "epilogue":{"lane_mapping":"wmma_accumulator_scalar_b16","vector_width":1},"numerical_mode":"ieee_fp16_acc_fp32"},
    "static_constraints":{"max_lds_bytes":65536,"max_vgpr_per_thread":256,"allow_spill":False},
    "applicability":{"exact_shape":True,"profiles":["execution_adapter_test"],"roles":[role],
      "targets":["AMD:gfx1100:wave32"]}}
  return derive_q4k_q8_1_five_buffer_candidate(payload)


def _request(entry, path):
  return ExecutionRequest(experiment_id="exp", candidate_id="candidate", comparator_id="base",
    workload_digest="workload", schedule_digest="schedule", transport_plan=TransportPlan("direct_global", "schedule"),
    target_context={}, compiler_context={"adapter_id": adapter.ADAPTER_ID, "candidate_payload": entry.payload,
      "canonical_identity": entry.canonical_identity, "input_npz": str(path)})


def _npz(tmp_path, admission):
  rows = {}
  for descriptor in admitted_buffer_descriptors(admission):
    name = "reference" if descriptor.direction == "out" else descriptor.name
    rows[name] = np.zeros(descriptor.flat_shape, dtype=getattr(np, descriptor.storage_dtype))
  path = tmp_path / "five.npz"; np.savez(path, **rows)
  return path


def test_exact_npz_contract_and_content_identities(tmp_path):
  entry = _entry((16, 16, 256))
  admission = adapter.admit_q4k_q8_five_buffer_compile(entry.payload, entry.canonical_identity)
  path = _npz(tmp_path, admission)
  inputs, reference, detail = adapter.load_q4k_q8_five_buffer_npz(str(path), admission)
  assert tuple(inputs) == ("q4_packed_words", "q8_ds4_values", "q8_scales", "q8_weighted_sums")
  assert reference.shape == (16, 16) and reference.dtype == np.float32
  assert detail["input_artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
  assert set(detail["content_sha256"]) == {*inputs, "reference"}

  wrong = tmp_path / "wrong.npz"
  np.savez(wrong, **{**inputs, "reference": reference, "extra": np.zeros(1)})
  with pytest.raises(ValueError, match="exactly"):
    adapter.load_q4k_q8_five_buffer_npz(str(wrong), admission)


def test_prepare_derives_argument_order_and_build_spec_without_runtime(monkeypatch, tmp_path):
  entry = _entry((16, 16, 256)); admission = adapter.admit_q4k_q8_five_buffer_compile(entry.payload, entry.canonical_identity)
  path = _npz(tmp_path, admission)
  evidence = {"passed": True, "canonical_identity": entry.canonical_identity, "source_sha256": "a"*64,
    "binary_sha256": "b"*64, "abi_digest": "c"*64, "target": "gfx1100", "compile_target": adapter.AMD_ISA_TARGET}
  monkeypatch.setattr(adapter, "make_tiny_health_probe", lambda **kw: (lambda: True))
  prepared = adapter.Q4KQ8FiveBufferAdapter(compile_prepare=lambda *_a, **_k: (object(), evidence)).prepare(_request(entry, path))
  assert tuple(prepared.inputs) == tuple(x.name for x in admitted_buffer_descriptors(admission) if x.direction == "in")
  assert prepared.output_dtype == np.float32
  assert prepared.builder.kwargs["canonical_identity"] == entry.canonical_identity
  assert prepared.compile_evidence["input_identity"].startswith("sha256:")


def test_child_recompile_mismatch_rejected_before_bundle(monkeypatch):
  entry = _entry((16, 16, 256)); admission = adapter.admit_q4k_q8_five_buffer_compile(entry.payload, entry.canonical_identity)
  digest = adapter._abi_digest(admitted_buffer_descriptors(admission))
  evidence = {"canonical_identity": entry.canonical_identity, "source_sha256": "a"*64, "binary_sha256": "b"*64,
    "abi_digest": digest, "target": "gfx1100", "compile_target": adapter.AMD_ISA_TARGET}
  evidence["child_recompile_binary_identity_contract"] = {**evidence, "enabled": True,
    "reject_sha256_mismatch_before_dispatch": True}
  monkeypatch.setattr(adapter, "prepare_q4k_q8_five_buffer_compile", lambda *_a, **_k:
    (object(), {**evidence, "source_sha256": "z"*64}))
  monkeypatch.setattr(adapter, "build_tinygrad_bundle", lambda **kw: pytest.fail("must reject before runtime bundle"))
  with pytest.raises(ValueError, match="source_sha256 differs"):
    adapter.build_q4k_q8_five_buffer_bundle(payload=entry.payload, canonical_identity=entry.canonical_identity,
      compile_evidence=evidence)


def test_registration_is_explicit_and_unique():
  registry = AdapterRegistry(); adapter.register_q4k_q8_five_buffer_adapter(registry)
  assert registry.ids() == (adapter.ADAPTER_ID,)
  assert registry.resolve("manual_five_buffer") is None


def test_static_compile_evidence_binds_exact_zero_resource_code_object_and_role(monkeypatch):
  from extra.qk import mmq_compile_evidence
  from tinygrad.renderer.amd import elf
  class Descriptor: group_segment_fixed_size = 0
  monkeypatch.setattr(mmq_compile_evidence, "parse_amdgpu_metadata", lambda _binary: {
    "vgpr": 48, "sgpr": 8, "lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0,
    "sgpr_spills": 0, "wavefront_size": 32})
  monkeypatch.setattr(elf, "kernel_descriptor_from_elf", lambda _binary: Descriptor())
  monkeypatch.setattr(elf, "descriptor_register_counts", lambda _desc, **_kwargs: (48, None))
  entry = _entry((16, 16, 256), role="attn_kv")
  _, evidence = adapter.prepare_q4k_q8_five_buffer_compile(entry.payload, entry.canonical_identity)
  resources = evidence["resource_summary"]
  assert evidence["surface"]["role"] == "attn_kv"
  assert resources["authority"] == "final_code_object_metadata_descriptor_and_program_launch"
  assert resources["lds_bytes"] == resources["admitted_active_lds_bytes"] == 0
  assert resources["scratch_bytes"] == resources["vgpr_spills"] == resources["sgpr_spills"] == 0
  assert resources["wavefront_size"] == 32 and resources["workgroup_threads"] == 32
  assert resources["source_sha256"] == evidence["source_sha256"]
  assert resources["binary_sha256"] == evidence["binary_sha256"]
