from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from extra.qk import mmq_frozen_target_artifact as frozen


def _program() -> UOp:
  sizes_dtypes = (
    (8_912_896, __import__("tinygrad").dtypes.float32),
    (626_688, __import__("tinygrad").dtypes.uint32),
    (131_072, __import__("tinygrad").dtypes.int8),
    (4_096, __import__("tinygrad").dtypes.float32),
    (4_096, __import__("tinygrad").dtypes.float32),
  )
  params = tuple(UOp.placeholder((size,), dtype, slot) for slot, (size, dtype) in enumerate(sizes_dtypes))
  sink = UOp(Ops.SINK, src=params)
  linear = UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=s_endpgm()),))
  shell = UOp(Ops.PROGRAM, src=(sink,))
  binary = assemble_linear(shell, linear, "gfx1100")
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD:ISA:gfx1100"), linear,
    UOp(Ops.SOURCE, arg="generated source\n"), UOp(Ops.BINARY, arg=binary)),
    arg=ProgramInfo(name=frozen.FUNCTION_NAME, global_size=(136, 4, 1), local_size=(256, 1, 1),
                    globals=tuple(range(5))))


def _fixture() -> dict:
  return {"schema": frozen.FIXTURE_SCHEMA, "shape": [512, 17_408, 5_120],
          "q4": {"source_sha256": "a" * 64, "epoch_major_sha256": "b" * 64},
          "q8": {"source_sha256": "c" * 64}}


def _disassembly() -> tuple[str, str]:
  return "s_endpgm // 0000000000000100: BFB00000\ns_code_end // 0000000000000104: BF9F0000\n", "cpu-test-objdump"


def test_frozen_target_producer_compiles_once_and_loads_without_recompile(tmp_path: Path):
  calls = []
  def compile_once():
    calls.append("compile")
    return SimpleNamespace(emitted=True, program=_program(), blocker=None)

  output, archive = tmp_path / "bundle", tmp_path / "bundle.tar"
  manifest = frozen.produce_frozen_target_artifact(
    output, archive=archive, compile_once=compile_once,
    disassemble=lambda binary: ("s_endpgm // 100: BFB00000\n", "cpu-test-objdump"),
    fixture_builder=_fixture)
  assert calls == ["compile"]
  assert manifest["compile_calls"] == 1
  assert manifest["accumulate"] is True
  assert manifest["accumulation"] == frozen.ACCUMULATION
  assert manifest["gpu_runtime_initialized"] is False and manifest["gpu_dispatch_performed"] is False

  directory_loaded = frozen.load_frozen_target_artifact(output)
  archive_loaded = frozen.load_frozen_target_artifact(archive)
  assert calls == ["compile"]
  assert directory_loaded.binary == archive_loaded.binary
  assert directory_loaded.program.key == archive_loaded.program.key
  assert directory_loaded.manifest["program"]["global_size"] == [136, 4, 1]
  assert directory_loaded.manifest["program"]["local_size"] == [256, 1, 1]
  assert directory_loaded.manifest["program"]["function"] == frozen.FUNCTION_NAME
  assert directory_loaded.manifest["program"]["target"] == "AMD:ISA:gfx1100"
  assert [row["slot"] for row in directory_loaded.manifest["program"]["abi"]] == list(range(5))
  assert isinstance(directory_loaded.manifest["compiler_environment"], dict)


def test_frozen_target_loader_rejects_retained_hsaco_tampering(tmp_path: Path):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=lambda: SimpleNamespace(emitted=True, program=_program(), blocker=None),
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  binary_path = output / frozen.FILE_NAMES["binary"]
  binary_path.write_bytes(binary_path.read_bytes() + b"tamper")
  with pytest.raises(ValueError, match="inventory identity mismatch"):
    frozen.load_frozen_target_artifact(output)


def test_frozen_target_producer_fails_closed_on_non_accumulating_function(tmp_path: Path):
  program = _program().replace(arg=ProgramInfo(name="mmq_llama_five_buffer_full_grid",
    global_size=(136, 4, 1), local_size=(256, 1, 1), globals=tuple(range(5))))
  with pytest.raises(ValueError, match="target function changed"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "bundle", compile_once=lambda: SimpleNamespace(emitted=True, program=program, blocker=None),
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)


def test_frozen_target_verify_cli_consumes_bundle_without_compiler(tmp_path: Path, capsys):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=lambda: SimpleNamespace(emitted=True, program=_program(), blocker=None),
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  assert frozen.main(["verify", str(output)]) == 0
  row = json.loads(capsys.readouterr().out)
  assert row["schema"] == frozen.SCHEMA and row["consumer"]["requires_recompile"] is False


def test_frozen_target_audit_joins_static_hashes_to_manifest(tmp_path: Path, capsys, monkeypatch):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=lambda: SimpleNamespace(emitted=True, program=_program(), blocker=None),
    disassemble=lambda binary: _disassembly(), fixture_builder=_fixture)

  expected_key = frozen.load_frozen_target_artifact(output).program.key.hex()
  load_calls, audit_calls = [], []
  real_load = frozen.load_frozen_target_artifact
  def counted_load(path):
    load_calls.append(path)
    return real_load(path)
  monkeypatch.setattr(frozen, "load_frozen_target_artifact", counted_load)
  import extra.qk.mmq_hsaco_static_audit as static_audit
  real_audit = static_audit.audit_hsaco
  def counted_audit(binary, disassembly):
    audit_calls.append((binary, disassembly))
    return real_audit(binary, disassembly)
  monkeypatch.setattr(static_audit, "audit_hsaco", counted_audit)

  assert frozen.main(["audit", str(output)]) == 0
  row = json.loads(capsys.readouterr().out)
  assert load_calls == [output] and len(audit_calls) == 1
  assert row["schema"] == frozen.AUDIT_SCHEMA
  assert row["passed"] is True and row["verdict"] == "PASS"
  assert row["identity"]["passed"] is True
  assert row["identity"]["observed"] == row["identity"]["expected"]
  assert row["static_audit"]["passed"] is True
  assert row["bundle"]["program_key"] == expected_key


def test_frozen_target_audit_cli_blocks_on_static_audit_or_identity_failure(tmp_path: Path, capsys, monkeypatch):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=lambda: SimpleNamespace(emitted=True, program=_program(), blocker=None),
    disassemble=lambda binary: _disassembly(), fixture_builder=_fixture)

  import extra.qk.mmq_hsaco_static_audit as static_audit
  real_audit = static_audit.audit_hsaco
  def mismatched_audit(binary, disassembly):
    result = real_audit(binary, disassembly)
    result["binary_sha256"] = "0" * 64
    return result
  monkeypatch.setattr(static_audit, "audit_hsaco", mismatched_audit)

  assert frozen.main(["audit", str(output)]) == 1
  row = json.loads(capsys.readouterr().out)
  assert row["passed"] is False and row["verdict"] == "BLOCKED"
  assert row["static_audit"]["passed"] is True
  assert row["identity"]["passed"] is False
  assert row["findings"] == ["static audit binary_sha256 differs from frozen manifest"]
