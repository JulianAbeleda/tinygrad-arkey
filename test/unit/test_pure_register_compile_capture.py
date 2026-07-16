import pytest

from extra.qk.amd_resource_artifact import AMDPhysicalInterval, AMDResourceFacts
from extra.qk.prefill.pure_register_compile_capture import (FinalCompileEvidence, capture_compile_only,
  instruction_order_proof)
from extra.qk.prefill.pure_register_evaluation_gate import REGISTER_STORAGE, compile_only, final_resources

IDENTITY = "a" * 64
DISASSEMBLY = """\
buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_pack_b32_f16 v200, v40, v41
v_mov_b32_e32 v42, v200
v_wmma_f32_16x16x16_f16_w32 v[0:7], v[200:207], v[216:223], v[0:7]
"""


def _evidence(**overrides):
  values = {"candidate_identity": IDENTITY, "target": "gfx1100", "abi": "amdgpu_kernel",
            "source": "compiler source", "binary": b"final binary", "disassembly": DISASSEMBLY,
            "resources": AMDResourceFacts(232, 32, 0, 0, 0, 0, 256, 32),
            "intervals": (AMDPhysicalInterval("A", "vgpr", 200, 216, "register_stage"),
                          AMDPhysicalInterval("B", "vgpr", 216, 232, "register_stage"))}
  values.update(overrides)
  return FinalCompileEvidence(**values)


def _capture(evidence=None):
  return capture_compile_only(evidence or _evidence(),
    pipeline={"storage_kind": REGISTER_STORAGE, "lds_bytes": 0,
      "consumer_identity": "amd.rdna3.wmma.fp16.v1",
      "register_mapping": {"backend": "amd_vgpr", "addressing": "sequential", "required_roles": ["A", "B"]},
      "wait_required_edges": [["A", 0, 1], ["B", 0, 1]]},
    wait={"typed": True, "kind": "targeted_vmcnt",
      "coverage": {"passed": True, "errors": [], "covered": [["A", 0, 1], ["B", 0, 1]]}},
    abi_contract={"wave_size": 32, "fragment_carrier": "half.vec(16)", "accumulator_carrier": "float.vec(8)"},
    surface={"strict_pure": True, "ops_ins_count": 0, "source_kind": "compiler_rendered"})


def test_capture_joins_final_binary_resources_intervals_and_order_without_dispatch():
  artifact = _capture()
  compiled = compile_only({"canonical_identity": IDENTITY}, artifact)
  assert artifact["capture"] == {"mode": "compile_only", "dispatch_permitted": False,
    "resource_authority": "final_code_object_descriptor", "allocator_authority": "final_regalloc"}
  assert artifact["resource_artifact"]["physical_intervals"][0]["logical_role"] == "A"
  assert artifact["instruction_order_proof"]["positions"] == {
    "global_load": 0, "vmcnt0_wait": 1, "stage_write": 2, "stage_read": 3, "wmma": 4}
  assert compiled["passed"] is True and final_resources(compiled)["passed"] is True


@pytest.mark.parametrize("disassembly", [
  DISASSEMBLY.replace("s_waitcnt vmcnt(0)\n", ""),
  DISASSEMBLY.replace("s_waitcnt vmcnt(0)\nv_pack_b32_f16 v200, v40, v41\n",
                      "v_pack_b32_f16 v200, v40, v41\ns_waitcnt vmcnt(0)\n"),
  DISASSEMBLY + "ds_read_b128 v[0:3], v0\n",
])
def test_instruction_order_proof_fails_closed_on_missing_reordered_or_lds(disassembly):
  assert instruction_order_proof(disassembly, _evidence().intervals)["passed"] is False


def test_instruction_order_does_not_accept_pack_or_move_outside_allocator_leases():
  disassembly = DISASSEMBLY.replace("v_pack_b32_f16 v200", "v_pack_b32_f16 v100")
  assert instruction_order_proof(disassembly, _evidence().intervals)["passed"] is False


def test_direct_global_fragment_order_needs_no_stage_copy():
  disassembly = """\
global_load_b128 v[200:203], v0, v0, s[0:1]
s_waitcnt vmcnt(0)
v_wmma_f32_16x16x16_f16 v[8:15], v[200:207], v[208:215], v[8:15]
"""
  intervals = (AMDPhysicalInterval("A", "vgpr", 200, 208, "direct_wmma_fragment"),
               AMDPhysicalInterval("B", "vgpr", 208, 216, "direct_wmma_fragment"))
  proof = instruction_order_proof(disassembly, intervals)
  assert proof["passed"] is True
  assert proof["fragment_transport"] == "direct_global"
  assert proof["positions"] == {"global_load": 0, "vmcnt0_wait": 1, "wmma": 2}
  assert instruction_order_proof(disassembly.replace("s_waitcnt vmcnt(0)\n", ""), intervals)["passed"] is False


@pytest.mark.parametrize(("field", "value"), [
  ("resource_authority", "host_estimate"), ("allocator_authority", "planned_intervals")])
def test_capture_rejects_nonfinal_authorities(field, value):
  with pytest.raises(ValueError): _evidence(**{field: value})


def test_capture_marks_spills_failed_and_gate_rejects_it():
  artifact = _capture(_evidence(resources=AMDResourceFacts(232, 32, 0, 16, 1, 0, 256, 32)))
  assert artifact["passed"] is False
  assert compile_only({"canonical_identity": IDENTITY}, artifact)["passed"] is False


def test_capture_rejects_candidate_identity_mismatch_at_gate():
  assert compile_only({"canonical_identity": "b" * 64}, _capture())["passed"] is False


@pytest.mark.parametrize(("field", "value"), [("target", "gfx1200"), ("abi", "amdgpu_wave")])
def test_capture_rejects_non_milestone_target_or_abi(field, value):
  with pytest.raises(ValueError): _evidence(**{field: value})
