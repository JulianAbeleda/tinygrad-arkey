import pytest

from extra.qk.prefill.pure_register_evaluation_gate import (COMPILE_SCHEMA, REGISTER_STORAGE, compile_only,
  evaluate, final_resources, machine_search, runtime_compile_resource_eligibility, validate_role_attribution)


IDENTITY = "a" * 64
BINARY = "b" * 64


def _compile(**overrides):
  resource = {"schema": "tinygrad.amd.resource_artifact.v1", "target": "gfx1100", "abi": "amdgpu_kernel",
              "source_sha256": "c" * 64, "binary_sha256": BINARY, "candidate_identity": IDENTITY,
              "resource_stage": "final_program",
              "resources": {"vgpr": 100, "sgpr": 80, "lds_bytes": 0, "scratch_bytes": 0,
                            "vgpr_spills": 0, "sgpr_spills": 0, "workgroup_threads": 256, "wavefront_size": 32},
              "physical_intervals": [{"logical_role": "A", "bank": "vgpr", "start": 0, "end": 8},
                                     {"logical_role": "B", "bank": "vgpr", "start": 8, "end": 16}]}
  row = {"schema": COMPILE_SCHEMA, "canonical_identity": IDENTITY, "binary_sha256": BINARY,
         "passed": True, "surface": {"strict_pure": True, "ops_ins_count": 0, "source_kind": "compiler_rendered"},
         "capture": {"mode": "compile_only", "dispatch_permitted": False,
                     "resource_authority": "final_code_object_descriptor", "allocator_authority": "final_regalloc"},
         "target_evidence": {"authority": "final_program", "target": "gfx1100", "abi": "amdgpu_kernel"},
         "instruction_order_proof": {"schema": "prefill-pure-register-instruction-order.v1",
           "authority": "final_disassembly", "passed": True, "errors": [],
           "disassembly_sha256": "d" * 64,
           "positions": {"global_load": 0, "vmcnt0_wait": 1, "stage_write": 2, "stage_read": 3, "wmma": 4},
           "lds_instruction_lines": []},
         "pipeline": {"storage_kind": REGISTER_STORAGE, "lds_bytes": 0,
                      "consumer_identity": "amd.rdna3.wmma.fp16.v1",
                      "register_mapping": {"backend": "amd_vgpr", "addressing": "sequential",
                                            "required_roles": ["A", "B"]},
                      "wait_required_edges": [["A", 0, 1], ["B", 0, 1]]},
         "wait": {"typed": True, "kind": "targeted_vmcnt",
                  "coverage": {"passed": True, "errors": [], "covered": [["A", 0, 1], ["B", 0, 1]]}},
         "abi": {"wave_size": 32, "fragment_carrier": "half.vec(16)", "accumulator_carrier": "float.vec(8)"},
         "resources": {"stage": "final_program", "vgpr": 100, "sgpr": 80, "lds_bytes": 0,
                       "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
                       "workgroup_threads": 256, "wave_count": 8}, "resource_artifact": resource}
  row.update(overrides)
  return row


def _correctness():
  return {"passed": True, "canonical_identity": IDENTITY, "binary_sha256": BINARY,
          "nonconstant_cases": True, "all_output_parity": True}


def _timing():
  return {"passed": True, "canonical_identity": IDENTITY, "binary_sha256": BINARY,
          "clock_pin": True, "tok_s": 2.0,
          "protocol": {"scope": "kernel_only", "compile_excluded": True}}


def _runtime_binding():
  return {"profile": "qwen3_8b_q4k_m_gfx1100", "role": "attn_qo",
          "shape": {"m": 512, "n": 4096, "k": 4096},
          "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}}


def _roles():
  return {role: {"passed": True, "strict_pure": True, "fallback_used": False, "route_family": "pure",
                 "canonical_identity": IDENTITY, "binary_sha256": BINARY,
                 "consumer_identity": "amd.rdna3.wmma.fp16.v1",
                 "candidate_fields": ["storage_kind", "wait_kind", "buffer_count", "slot_addressing", "consumer_identity"],
                 "policy": {"storage_kind": REGISTER_STORAGE, "wait_kind": "targeted_vmcnt"},
                 "search_space": "typed_policy_fields"}
          for role in ("attn_qo", "ffn_down", "attn_kv", "ffn_gate_up")}


def test_missing_register_artifact_blocks_before_resource_or_search_claims():
  report = evaluate({"canonical_identity": IDENTITY})
  assert report["passed"] is False and report["blocked_at"] == "compile"
  assert "register compile artifact is unavailable" in report["blockers"]["compile"]


def test_runtime_register_warmstart_missing_evidence_is_default_closed():
  binding = _runtime_binding()
  row = runtime_compile_resource_eligibility({"canonical_identity": IDENTITY}, None,
    profile=binding["profile"], role=binding["role"], shape=(512,4096,4096), target=binding["target"])
  assert row["passed"] is False
  assert any("compile artifact is unavailable" in error for error in row["errors"])


def test_runtime_register_warmstart_rejects_invalid_exact_evidence():
  binding = _runtime_binding()
  artifact = _compile(runtime_binding={**binding, "shape": {"m":512,"n":4096,"k":2048}})
  row = runtime_compile_resource_eligibility({"canonical_identity": IDENTITY}, artifact,
    profile=binding["profile"], role=binding["role"], shape=(512,4096,4096), target=binding["target"])
  assert row["passed"] is False
  assert any("exact workload/target match" in error for error in row["errors"])


def test_runtime_register_warmstart_accepts_valid_exact_compile_resource_evidence():
  binding = _runtime_binding()
  row = runtime_compile_resource_eligibility({"canonical_identity": IDENTITY}, _compile(runtime_binding=binding),
    profile=binding["profile"], role=binding["role"], shape=(512,4096,4096), target=binding["target"])
  assert row["passed"] is True
  assert row["canonical_identity"] == IDENTITY and row["binary_sha256"] == BINARY


def test_runtime_register_warmstart_profile_is_provenance_only():
  binding = _runtime_binding()
  artifact = _compile(runtime_binding={**binding, "profile": "renamed evidence fixture"})
  row = runtime_compile_resource_eligibility({"canonical_identity": IDENTITY}, artifact,
    profile="different legacy caller label", role=binding["role"], shape=(512,4096,4096), target=binding["target"])
  assert row["passed"] is True


def test_lds_candidate_is_not_admitted_as_register_compile():
  artifact = _compile(pipeline={"storage_kind": "lds", "lds_bytes": 20480})
  row = compile_only({"canonical_identity": IDENTITY}, artifact)
  assert row["passed"] is False
  assert any("global_register_resident" in error for error in row["errors"])
  assert any("claims LDS" in error for error in row["errors"])


def test_register_compile_requires_physical_mapping_and_wait_edges():
  no_mapping = _compile(pipeline={"storage_kind": REGISTER_STORAGE, "lds_bytes": 0},
                        wait={"typed": True, "kind": "targeted_vmcnt"})
  row = compile_only({"canonical_identity": IDENTITY}, no_mapping)
  assert row["passed"] is False
  assert any("physical VGPR mapping" in error for error in row["errors"])
  assert any("wait dependency coverage" in error for error in row["errors"])


def test_register_compile_requires_generic_consumer_identity():
  artifact = _compile(pipeline={"storage_kind": REGISTER_STORAGE, "lds_bytes": 0,
                                "register_mapping": {"backend": "amd_vgpr", "addressing": "sequential",
                                                      "required_roles": ["A", "B"]},
                                "wait_required_edges": [["A", 0, 1], ["B", 0, 1]]})
  row = compile_only({"canonical_identity": IDENTITY}, artifact)
  assert row["passed"] is False
  assert any("consumer identity" in error for error in row["errors"])


def test_register_compile_mapping_must_cover_artifact_roles():
  artifact = _compile(pipeline={"storage_kind": REGISTER_STORAGE, "lds_bytes": 0,
                                "register_mapping": {"backend": "amd_vgpr", "addressing": "sequential",
                                                      "required_roles": ["A", "B", "accumulator"]},
                                "wait_required_edges": [["A", 0, 1], ["B", 0, 1]]})
  row = compile_only({"canonical_identity": IDENTITY}, artifact)
  assert row["passed"] is False
  assert any("missing required logical register roles" in error for error in row["errors"])


def test_final_resource_gate_rejects_host_estimates_and_spills():
  host_resource = _compile()["resource_artifact"] | {"resource_stage": "host_estimate"}
  artifact = _compile(resource_artifact=host_resource)
  assert compile_only({"canonical_identity": IDENTITY}, artifact)["passed"] is False
  spill_resource = _compile()["resource_artifact"] | {"resources": _compile()["resource_artifact"]["resources"] | {"vgpr_spills": 1}}
  artifact = _compile(resource_artifact=spill_resource)
  compiled = compile_only({"canonical_identity": IDENTITY}, artifact)
  assert compiled["passed"] is True and final_resources(compiled)["passed"] is False


def test_compile_gate_rejects_estimated_or_dispatch_capable_capture():
  estimated = _compile(capture={"mode": "compile_only", "dispatch_permitted": False,
                                "resource_authority": "host_estimate", "allocator_authority": "final_regalloc"})
  assert any("estimated" in error for error in compile_only({"canonical_identity": IDENTITY}, estimated)["errors"])
  dispatching = _compile(capture={"mode": "compile_only", "dispatch_permitted": True,
                                  "resource_authority": "final_code_object_descriptor", "allocator_authority": "final_regalloc"})
  assert any("non-dispatching" in error for error in compile_only({"canonical_identity": IDENTITY}, dispatching)["errors"])


@pytest.mark.parametrize("field", ["target_evidence", "resource_artifact"])
def test_compile_gate_rejects_target_or_abi_identity_mismatch(field):
  artifact = _compile()
  if field == "target_evidence":
    artifact["target_evidence"] = {"authority": "final_program", "target": "gfx1200", "abi": "amdgpu_kernel"}
  else:
    artifact["resource_artifact"] = artifact["resource_artifact"] | {"abi": "amdgpu_wave"}
  assert compile_only({"canonical_identity": IDENTITY}, artifact)["passed"] is False


def test_compile_gate_rejects_non_sha256_binary_identity():
  assert compile_only({"canonical_identity": IDENTITY}, _compile(binary_sha256="z" * 64))["passed"] is False


def test_machine_search_requires_every_role_and_typed_policy_fields():
  rows = _roles()
  del rows["attn_kv"]
  report = machine_search(rows)
  assert report["passed"] is False and any("attn_kv" in error for error in report["errors"])


def test_machine_search_rejects_missing_consumer_identity():
  rows = _roles()
  rows["attn_qo"] = {**rows["attn_qo"], "consumer_identity": None}
  report = machine_search(rows)
  assert report["passed"] is False and any("attn_qo" in error and "consumer identity" in error for error in report["errors"])


def test_machine_search_rejects_untyped_candidate_field_set():
  rows = _roles()
  rows["attn_qo"] = {**rows["attn_qo"], "candidate_fields": ["storage_kind", "wait_kind"]}
  report = machine_search(rows)
  assert report["passed"] is False and any("attn_qo" in error and "candidate fields" in error for error in report["errors"])


def test_machine_search_joins_role_consumer_identity_when_report_supplies_it():
  route = {"prefill_role_routes": {role: "register" for role in _roles()},
           "prefill_role_consumers": {role: "amd.rdna3.wmma.fp16.v1" for role in _roles()}}
  assert machine_search(_roles(), route_report=route)["passed"] is True
  route["prefill_role_consumers"]["ffn_down"] = "amd.rdna3.dot2.fp16.v1"
  report = machine_search(_roles(), route_report=route)
  assert report["passed"] is False and any("ffn_down" in error and "does not match" in error for error in report["errors"])


def test_role_attribution_requires_all_roles_and_does_not_trust_top_level_pure_flag():
  mixed = {"route_attribution": {"prefill_route_pure": True},
           "prefill_role_routes": {"ffn_gate_up": "candidate_lds_single_buffer"}}
  report = validate_role_attribution(mixed)
  assert report["passed"] is False
  assert any("attn_qo" in error for error in report["errors"])
  pure = {"route_attribution": {"prefill_route_pure": True},
          "prefill_role_routes": {role: "register" for role in ("attn_qo", "ffn_down", "attn_kv", "ffn_gate_up")}}
  assert validate_role_attribution(pure, require_pure=True)["passed"] is True


def test_machine_search_rejects_mixed_whole_prefill_role_map():
  mixed = {"prefill_role_routes": {role: "register" for role in ("attn_qo", "ffn_down", "attn_kv", "ffn_gate_up")}}
  mixed["prefill_role_routes"]["ffn_gate_up"] = "candidate_lds_single_buffer"
  report = machine_search(_roles(), route_report=mixed)
  assert report["passed"] is False and any("role attribution" in error for error in report["errors"])


def test_all_gates_join_for_synthetic_register_artifact():
  report = evaluate({"canonical_identity": IDENTITY}, compile_artifact=_compile(),
                    correctness=_correctness(), timing=_timing(), role_evidence=_roles(), baseline_tok_s=1.0)
  assert report["passed"] is True and report["blocked_at"] is None
  assert all(report["stages"][name]["passed"] for name in ("compile", "resources", "correctness_timing", "machine_search"))
