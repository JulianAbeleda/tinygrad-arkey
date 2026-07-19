from __future__ import annotations

from dataclasses import replace
from functools import cache
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.amd.dsl import s
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.runtime.autogen.amd.rdna3.ins import s_branch, s_endpgm, s_mov_b32
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from extra.qk import mmq_frozen_target_artifact as frozen
from extra.qk import mmq_llama_five_buffer_full_kernel as full_kernel
from extra.qk import mmq_target_epoch_orchestrator as orchestrator
from extra.qk.mmq_exact_role_spec import DEFAULT_EXACT_ROLE_SPEC, ExactRoleSpec, exact_role_spec
from extra.qk.mmq_target_epoch_orchestrator import compile_target_kernel


def _program(role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC) -> UOp:
  tg_dtypes = __import__("tinygrad").dtypes
  sizes_dtypes = tuple(zip(role_spec.program.abi_elements,
                           (tg_dtypes.float32, tg_dtypes.uint32, tg_dtypes.int8, tg_dtypes.float32, tg_dtypes.float32)))
  params = tuple(UOp.placeholder((size,), dtype, slot) for slot, (size, dtype) in enumerate(sizes_dtypes))
  sink = UOp(Ops.SINK, src=params)
  linear = UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=s_endpgm()),))
  shell = UOp(Ops.PROGRAM, src=(sink,))
  binary = assemble_linear(shell, linear, "gfx1100")
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), linear,
    UOp(Ops.SOURCE, arg="generated source\n"), UOp(Ops.BINARY, arg=binary)),
    arg=ProgramInfo(name=frozen.FUNCTION_NAME, global_size=role_spec.program.grid, local_size=(256, 1, 1),
                    globals=tuple(range(5))))


@cache
def _kernel(role_spec: ExactRoleSpec):
  return full_kernel.build_llama_five_buffer_full_kernel(
    *role_spec.program.shape, accumulate=True)


def _compiled(role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, program: UOp | None = None):
  program = _program(role_spec) if program is None else program
  kernel = _kernel(role_spec)
  with patch.object(full_kernel, "build_llama_five_buffer_full_kernel", return_value=kernel), \
       patch.object(full_kernel, "compile_llama_five_buffer_full_kernel",
                    side_effect=lambda built, *, target: replace(
                      built, program=program, emitted=True, blocker="")):
    return compile_target_kernel(
      accumulate=True, target=full_kernel.AMD_ISA_TARGET, role_spec=role_spec)


def _fixture(role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC) -> dict:
  return {"schema": frozen.FIXTURE_SCHEMA, "role": role_spec.role, "shape": list(role_spec.shape),
          "q4": {"source_sha256": "a" * 64, "epoch_major_sha256": "b" * 64},
          "q8": {"source_sha256": "c" * 64}}


def _disassembly() -> tuple[str, str]:
  return "s_endpgm // 0000000000000100: BFB00000\ns_code_end // 0000000000000104: BF9F0000\n", "cpu-test-objdump"


def test_final_stream_disassembly_sign_extends_backedges_and_advances_by_encoded_size():
  linear = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=inst) for inst in (
    s_branch(simm16=0xffff), s_mov_b32(s[0], 0x12345678), s_endpgm())))
  lines = AMDISARenderer._final_disassembly(linear, start_pc=0x100).splitlines()
  assert lines == [
    "s_branch -1 // 0000000000000100: BFA0FFFF",
    "s_mov_b32 s0, lit, 305419896 // 0000000000000104: BE8000FF 12345678",
    "s_endpgm 0 // 000000000000010c: BFB00000",
  ]


def test_frozen_target_producer_compiles_once_and_loads_without_recompile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setenv("PYTHONHASHSEED", "0")
  monkeypatch.setenv("REGALLOC_ADDR_REMAT", "1")
  calls = []
  def compile_once():
    calls.append("compile")
    return _compiled()

  output, archive = tmp_path / "bundle", tmp_path / "bundle.tar"
  manifest = frozen.produce_frozen_target_artifact(
    output, archive=archive, compile_once=compile_once, fixture_builder=_fixture)
  assert calls == ["compile"]
  assert manifest["compile_calls"] == 1
  assert manifest["accumulate"] is True
  assert manifest["accumulation"] == frozen.ACCUMULATION
  assert manifest["gpu_runtime_initialized"] is False and manifest["gpu_dispatch_performed"] is False
  assert manifest["artifacts"]["disassembly_tool"] == "renderer-final-stream-byte-reassembled"
  assert manifest["schema"] == frozen.SCHEMA and manifest["schema"].endswith(".v2")
  assert manifest["source_sink"]["authority"] == "same_session_pre_lowering_sink_passed_to_compiler"

  directory_loaded = frozen.load_frozen_target_artifact(output)
  archive_loaded = frozen.load_frozen_target_artifact(archive)
  assert calls == ["compile"]
  assert directory_loaded.binary == archive_loaded.binary
  assert directory_loaded.program.key == archive_loaded.program.key
  assert directory_loaded.sink is not None and archive_loaded.sink is not None
  assert directory_loaded.sink.key == archive_loaded.sink.key
  assert directory_loaded.sink.key.hex() == manifest["source_sink"]["key"]
  assert directory_loaded.manifest["program"]["global_size"] == [136, 4, 1]
  assert directory_loaded.manifest["program"]["local_size"] == [256, 1, 1]
  assert directory_loaded.manifest["program"]["function"] == frozen.FUNCTION_NAME
  assert directory_loaded.manifest["program"]["device"] == "AMD"
  assert directory_loaded.manifest["program"]["compile_target"] == "AMD:ISA:gfx1100"
  assert [row["slot"] for row in directory_loaded.manifest["program"]["abi"]] == list(range(5))
  assert directory_loaded.manifest["compiler_environment"]["PYTHONHASHSEED"] == "0"
  assert directory_loaded.manifest["compiler_environment"]["REGALLOC_ADDR_REMAT"] == "1"
  assert set(directory_loaded.manifest["compiler_environment"]) == set(frozen.COMPILER_ENV)
  assert directory_loaded.manifest["compiler_environment"]["SCHED_MODULO"] is None
  assert directory_loaded.disassembly.startswith("s_endpgm 0 // 0000000000000100: BFB00000")
  assert frozen.audit_frozen_target_artifact(output)["passed"] is True


def test_frozen_role_geometry_supports_attn_kv_and_shares_attn_qo_ffn_down_program(tmp_path: Path):
  kv, qo, down = (exact_role_spec(role) for role in ("attn_kv", "attn_qo", "ffn_down"))
  bundles = {}
  for role_spec in (kv, qo, down):
    output = tmp_path / role_spec.role
    manifest = frozen.produce_frozen_target_artifact(
      output, role_spec=role_spec,
      compile_once=lambda spec=role_spec: _compiled(spec),
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"),
      fixture_builder=lambda spec=role_spec: _fixture(spec))
    loaded = frozen.load_frozen_target_artifact(output)
    assert manifest["shape"] == list(role_spec.program.shape)
    assert manifest["full_role_shape"] == list(role_spec.shape)
    assert manifest["program"]["global_size"] == list(role_spec.program.grid)
    assert [row["elements"] for row in manifest["program"]["abi"]] == list(role_spec.program.abi_elements)
    assert loaded.fixture["role"] == role_spec.role
    bundles[role_spec.role] = loaded
  assert bundles["attn_kv"].manifest["program"]["global_size"] == [8, 4, 1]
  assert bundles["attn_qo"].manifest["program"]["global_size"] == [40, 4, 1]
  assert bundles["attn_qo"].binary == bundles["ffn_down"].binary
  assert bundles["attn_qo"].program.key == bundles["ffn_down"].program.key
  assert bundles["attn_qo"].fixture["shape"] != bundles["ffn_down"].fixture["shape"]


def test_frozen_producer_rejects_full_role_fixture_mismatch_even_when_program_geometry_is_shared(tmp_path: Path):
  qo, down = exact_role_spec("attn_qo"), exact_role_spec("ffn_down")
  with pytest.raises(ValueError, match="fixture full-role shape differs"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "bundle", role_spec=down,
      compile_once=lambda: _compiled(qo),
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"),
      fixture_builder=lambda: _fixture(qo))


def test_frozen_target_loader_rejects_retained_hsaco_tampering(tmp_path: Path):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  binary_path = output / frozen.FILE_NAMES["binary"]
  binary_path.write_bytes(binary_path.read_bytes() + b"tamper")
  with pytest.raises(ValueError, match="inventory identity mismatch"):
    frozen.load_frozen_target_artifact(output)


def test_frozen_target_loader_retains_sparse_legacy_environment_compatibility(tmp_path: Path):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  manifest_path = output / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["compiler_environment"] = {}
  manifest_path.write_text(json.dumps(manifest))
  assert frozen.load_frozen_target_artifact(output).manifest["compiler_environment"] == {}


def test_frozen_target_loader_retains_v1_compatibility_without_source_sink(tmp_path: Path):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  manifest_path = output / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  sink_name = frozen.FILE_NAMES["sink"]
  (output / sink_name).unlink()
  manifest["schema"] = frozen.LEGACY_SCHEMA
  manifest.pop("source_sink")
  manifest["files"].pop(sink_name)
  manifest_path.write_text(json.dumps(manifest))
  loaded = frozen.load_frozen_target_artifact(output)
  assert loaded.manifest["schema"] == frozen.LEGACY_SCHEMA
  assert loaded.sink is None


def test_frozen_target_v2_rejects_missing_or_tampered_source_sink(tmp_path: Path):
  with pytest.raises(ValueError, match="requires the exact issued compile_target_kernel proof"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "missing", compile_once=lambda: _program(),
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)

  output = tmp_path / "tampered"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  sink_path = output / frozen.FILE_NAMES["sink"]
  sink_path.write_bytes(sink_path.read_bytes() + b"tamper")
  with pytest.raises(ValueError, match="inventory identity mismatch"):
    frozen.load_frozen_target_artifact(output)


def test_frozen_target_v2_rejects_copied_proof_with_mismatched_sink_and_program(tmp_path: Path):
  qo, kv = exact_role_spec("attn_qo"), exact_role_spec("attn_kv")
  proof = _compiled(qo)
  mismatched = replace(proof, sink=_kernel(kv).sink)
  with pytest.raises(ValueError, match="requires the exact issued compile_target_kernel proof"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "mismatch", role_spec=qo, compile_once=lambda: mismatched,
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"),
      fixture_builder=lambda: _fixture(qo))


def test_frozen_target_v2_rejects_replace_with_self_consistent_full_tuple(tmp_path: Path):
  qo, kv = exact_role_spec("attn_qo"), exact_role_spec("attn_kv")
  original, replacement = _compiled(qo), _compiled(kv)
  forged = replace(original, kernel=replacement.kernel, sink=replacement.sink, program=replacement.program)
  with pytest.raises(ValueError, match="requires the exact issued compile_target_kernel proof"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "full-tuple-forgery", role_spec=kv, compile_once=lambda: forged,
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"),
      fixture_builder=lambda: _fixture(kv))


def test_frozen_target_v2_consumes_issued_compile_authority_once(tmp_path: Path):
  proof = _compiled()
  frozen.produce_frozen_target_artifact(
    tmp_path / "first", compile_once=lambda: proof,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  with pytest.raises(ValueError, match="requires the exact issued compile_target_kernel proof"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "second", compile_once=lambda: proof,
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)


def test_frozen_target_v2_rejects_issued_proof_from_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  proof, issuing_pid = _compiled(), orchestrator.os.getpid()
  monkeypatch.setattr(orchestrator.os, "getpid", lambda: issuing_pid + 1)
  with pytest.raises(ValueError, match="cannot cross a process boundary"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "cross-process", compile_once=lambda: proof,
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)


def test_frozen_target_rejects_program_device_drift(tmp_path: Path):
  program = _program()
  changed = program.replace(src=(program.src[0], UOp(Ops.DEVICE, arg="AMD:ISA:gfx1100"), *program.src[2:]))
  with pytest.raises(ValueError, match="device changed"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "bundle", compile_once=lambda: _compiled(program=changed),
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)


def test_frozen_target_loader_rejects_manifest_compile_target_drift(tmp_path: Path):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  manifest_path = output / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["program"]["compile_target"] = "AMD:ISA:gfx1200"
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="launch identity differs"):
    frozen.load_frozen_target_artifact(output)


def test_frozen_target_producer_fails_closed_on_non_accumulating_function(tmp_path: Path):
  program = _program().replace(arg=ProgramInfo(name="mmq_llama_five_buffer_full_grid",
    global_size=(136, 4, 1), local_size=(256, 1, 1), globals=tuple(range(5))))
  with pytest.raises(ValueError, match="target function changed"):
    frozen.produce_frozen_target_artifact(
      tmp_path / "bundle", compile_once=lambda: _compiled(program=program),
      disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)


def test_frozen_target_verify_cli_consumes_bundle_without_compiler(tmp_path: Path, capsys):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
    disassemble=lambda binary: ("s_endpgm\n", "cpu-test-objdump"), fixture_builder=_fixture)
  assert frozen.main(["verify", str(output)]) == 0
  row = json.loads(capsys.readouterr().out)
  assert row["schema"] == frozen.SCHEMA and row["consumer"]["requires_recompile"] is False


def test_frozen_target_audit_joins_static_hashes_to_manifest(tmp_path: Path, capsys, monkeypatch):
  output = tmp_path / "bundle"
  frozen.produce_frozen_target_artifact(
    output, compile_once=_compiled,
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
    output, compile_once=_compiled,
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
