import pytest

from extra.qk.amd_resource_artifact import AMDPhysicalInterval, AMDResourceFacts
from extra.qk.prefill.pure_register_compile_capture import capture_final_program_compile_only


IDENTITY = "c" * 64
ISA = """buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_pack_b32_f16 v200, v40, v41
v_mov_b32_e32 v42, v200
v_wmma_f32_16x16x16_f16_w32 v[0:7], v[200:207], v[216:223], v[0:7]
"""


def _program(**overrides):
  row = {"candidate_identity": IDENTITY, "target": "gfx1100", "abi": "amdgpu_kernel",
    "source": "actual assembler source", "binary": b"actual code object", "disassembly": ISA,
    "descriptor": {"authority": "final_code_object_descriptor",
      "resources": {"vgpr": 232, "sgpr": 32, "lds_bytes": 0, "scratch_bytes": 0,
        "vgpr_spills": 0, "sgpr_spills": 0, "workgroup_threads": 256, "wavefront_size": 32}},
    "allocator": {"authority": "final_regalloc", "intervals": [
      {"logical_role": "A", "bank": "vgpr", "start": 200, "end": 216, "purpose": "register_stage"},
      {"logical_role": "B", "bank": "vgpr", "start": 216, "end": 232, "purpose": "register_stage"}]}}
  row.update(overrides)
  return row


def _capture(program):
  return capture_final_program_compile_only(program, pipeline={"storage_kind": "global_register_resident"},
    wait={"typed": True}, abi_contract={"wave_size": 32}, surface={"strict_pure": True})


def test_adapter_captures_actual_final_outputs_and_preserves_gate_contract():
  artifact = _capture(_program())
  assert artifact["passed"] is True
  assert artifact["capture"]["dispatch_permitted"] is False
  assert artifact["resource_artifact"]["resources"]["vgpr"] == 232
  assert artifact["resource_artifact"]["physical_intervals"][1]["end"] == 232


@pytest.mark.parametrize("field", ["descriptor", "allocator"])
def test_adapter_fails_closed_without_final_authority(field):
  program = _program(**{field: {"authority": "host_estimate"}})
  with pytest.raises(ValueError, match="authoritative"):
    _capture(program)


def test_adapter_rejects_missing_or_untyped_final_facts():
  with pytest.raises(ValueError, match="typed resources"):
    _capture(_program(descriptor={"authority": "final_code_object_descriptor", "resources": {"vgpr": 232}}))
  with pytest.raises(ValueError, match="typed intervals"):
    _capture(_program(allocator={"authority": "final_regalloc", "intervals": []}))
