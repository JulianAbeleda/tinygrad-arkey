import pytest

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, Ops, ProgramInfo, UOp

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


def test_capture_anchor_records_repository_state(monkeypatch):
  monkeypatch.setattr(cap, "_git_state", lambda: {"revision": "abc", "dirty": False})
  monkeypatch.setattr(cap, "build_pure_program", lambda: _program((UOp(Ops.NOOP, dtypes.void),)))
  monkeypatch.setattr(cap, "build_s9_oracle_program", lambda: _program((UOp(Ops.INS, arg="s_endpgm"),), ".text\nk:"))
  monkeypatch.setattr(cap, "parse_amdgpu_metadata", lambda _: {"symbol": "k.kd"})
  monkeypatch.setattr(cap, "disassemble_amdgpu", lambda _: ("s_endpgm", "objdump"))
  assert cap.capture_anchor()["git"] == {"revision": "abc", "dirty": False}


def test_candidate_resource_authority_binds_context_binary_and_emitted_lds(monkeypatch):
  from test.unit.test_pure_single_buffer_evaluation_gate import _payload
  from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash
  payload = _payload(); identity = canonical_candidate_hash(payload)
  base = _program((UOp(Ops.NOOP, dtypes.void),))
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=20480, addrspace=AddrSpace.LOCAL), (), "lds")
  program = base.replace(src=(
    UOp.sink(lds, arg=KernelInfo(name="k", candidate_context=KernelCandidateContext(payload["schema_version"], identity))),
    *base.src[1:]), arg=ProgramInfo(name="k", global_size=(1, 1, 1), local_size=(256, 1, 1)))
  metadata = {"symbol": "k.kd", "vgpr": 120, "sgpr": 32, "vgpr_spills": 0, "sgpr_spills": 0,
              "lds_bytes": 20480, "scratch_bytes": 0, "max_workgroup_threads": 256, "wavefront_size": 32,
              "target": "amdgcn-amd-amdhsa--gfx1100"}
  monkeypatch.setattr(cap, "parse_amdgpu_metadata", lambda _: metadata)
  monkeypatch.setattr(cap, "disassemble_amdgpu", lambda _: ("ds_store_b128 v0, v[1:4]\nds_load_b128 v[5:8], v0\n", "objdump"))
  out = cap.capture_candidate_program(program, payload, identity)
  assert out["passed"] is True
  assert out["resources"]["lds_bytes"] == 20480
  assert out["isa"]["compiler_emitted_single_buffer_lds"] is True


def test_candidate_resource_authority_fails_closed_on_context_or_binary_resources(monkeypatch):
  from test.unit.test_pure_single_buffer_evaluation_gate import _payload
  from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash
  payload = _payload(); identity = canonical_candidate_hash(payload)
  base = _program((UOp(Ops.NOOP, dtypes.void),))
  program = base.replace(src=(UOp.sink(arg=KernelInfo(name="k", candidate_context=KernelCandidateContext(
    payload["schema_version"], "0" * 64))), *base.src[1:]))
  with pytest.raises(RuntimeError, match="context does not match"):
    cap.capture_candidate_program(program, payload, identity)
