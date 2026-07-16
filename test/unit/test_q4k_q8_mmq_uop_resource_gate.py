import hashlib

import pytest

from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp

import extra.qk.q4k_q8_mmq_uop_resource_gate as gate


def _program_and_sink():
  sink = gate.build_sink(16, 16, 256)
  return to_program(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))), sink


def _metadata(program):
  return {"vgpr": 48, "sgpr": 32, "lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0,
          "sgpr_spills": 0, "wavefront_size": 32,
          "max_workgroup_threads": 32, "symbol": program.arg.function_name + ".kd"}


def _isa(mnemonic=gate.SIGNED_WMMA_MNEMONIC):
  return {"instruction_count": 10, "scratch_sites": 0, "instructions": [{"mnemonic": mnemonic}] * 4}


def test_mock_final_program_evidence_separates_uops_from_isa(monkeypatch):
  program, sink = _program_and_sink()
  monkeypatch.setattr(gate, "parse_amdgpu_metadata", lambda _: _metadata(program))
  monkeypatch.setattr(gate, "disassemble_amdgpu", lambda _: ("exact disassembly\n", "mock-objdump"))
  monkeypatch.setattr(gate, "analyze_final_isa", lambda text, wavefront_size: _isa())
  row = gate.capture_program_evidence(program, sink)
  assert row["uops"]["authored_wmma"] == 0
  assert row["uops"]["final_program_lowered_wmma"] == 1
  assert row["final_isa"]["wmma_count"] == 4
  assert row["final_isa"]["wmma_mnemonic"] == "v_wmma_i32_16x16x16_iu8"
  assert row["occupancy"]["status"] == "unavailable" and row["occupancy_gate_called"] is False
  assert row["identity"]["rendered_source_sha256"] == hashlib.sha256(program.src[3].arg.encode()).hexdigest()
  assert row["identity"]["binary_sha256"] == hashlib.sha256(program.src[4].arg).hexdigest()


@pytest.mark.parametrize("failure", ["metadata", "symbol", "mnemonic", "isa"])
def test_mock_gate_fails_closed(monkeypatch, failure):
  program, sink = _program_and_sink()
  metadata = _metadata(program)
  if failure == "metadata": metadata.pop("scratch_bytes")
  if failure == "symbol": metadata["symbol"] = "wrong.kd"
  monkeypatch.setattr(gate, "parse_amdgpu_metadata", lambda _: metadata)
  monkeypatch.setattr(gate, "disassemble_amdgpu", lambda _: ("disassembly", "mock"))
  mnemonic = "v_wmma_i32_16x16x16_uu8" if failure == "mnemonic" else gate.SIGNED_WMMA_MNEMONIC
  monkeypatch.setattr(gate, "analyze_final_isa", lambda text, wavefront_size: _isa(mnemonic) if failure != "isa" else
                      {"instruction_count": 6, "scratch_sites": 0, "instructions": []})
  with pytest.raises(ValueError): gate.capture_program_evidence(program, sink)


def test_rejects_non_program_and_unapproved_shape():
  with pytest.raises(ValueError, match="Ops.PROGRAM"): gate.capture_program_evidence(UOp(Ops.SINK), UOp(Ops.SINK))
  with pytest.raises(ValueError, match="unsupported evidence shape"): gate.capture_shape(16, 32, 256, device="CPU")
