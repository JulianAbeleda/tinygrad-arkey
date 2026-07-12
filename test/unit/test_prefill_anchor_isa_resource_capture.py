from tinygrad.dtype import dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.prefill import anchor_isa_resource_capture as cap


def _program(linear, source="kernel source"):
  return UOp(Ops.PROGRAM, src=(UOp.sink(), UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=linear),
                               UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=b"binary")),
             arg=KernelInfo(name="k"))


def test_program_surface_classifies_scheduler_generated_program_as_pure():
  out = cap._program_surface(_program((UOp(Ops.NOOP, dtypes.void),)))
  assert out == {"ops_ins_count": 0, "source_kind": "compiler_rendered", "forbidden_markers": [], "strict_pure": True}


def test_program_surface_classifies_ops_ins_assembly_oracle_as_forbidden():
  out = cap._program_surface(_program((UOp(Ops.INS, arg="s_endpgm"),), ".text\nk:"))
  assert out["strict_pure"] is False
  assert out["source_kind"] == "native_isa"
  assert out["forbidden_markers"] == ["Ops.INS", "assembly_source"]


def test_capture_program_binds_hashes_resources_and_purity(monkeypatch):
  monkeypatch.setattr(cap, "parse_amdgpu_metadata", lambda _: {"symbol": "k.kd", "vgpr": 7})
  monkeypatch.setattr(cap, "disassemble_amdgpu", lambda _: ("s_endpgm", "objdump"))
  out = cap.capture_program(_program((UOp(Ops.NOOP, dtypes.void),)), candidate_id="c", route_id="r", expected_pure=True)
  assert out["purity_matches_expectation"] is True
  assert len(out["program"]["binary_sha256"]) == 64
  assert out["resources"]["vgpr"] == 7
