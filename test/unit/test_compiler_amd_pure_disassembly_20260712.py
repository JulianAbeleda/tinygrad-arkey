from types import SimpleNamespace

import pytest

from tinygrad.runtime.support import compiler_amd


def test_pure_amd_disassembly_returns_data_and_tool_identity(monkeypatch):
  monkeypatch.setattr(compiler_amd, "_find_llvm_objdump", lambda: "/fake/llvm-objdump")
  monkeypatch.setattr(compiler_amd.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
    stdout=b"header\ns_code_end\n"))

  result = compiler_amd.amdgpu_disassemble_result(b"already-built-code-object")

  assert result.ok is True
  assert result.disassembly == "header"
  assert result.tool == "/fake/llvm-objdump"
  assert result.error is None


def test_pure_amd_disassembly_missing_objdump_is_structured_and_fail_closed(monkeypatch):
  def missing():
    raise FileNotFoundError("llvm-objdump not found")
  monkeypatch.setattr(compiler_amd, "_find_llvm_objdump", missing)

  result = compiler_amd.amdgpu_disassemble_result(b"already-built-code-object")

  assert result.ok is False
  assert result.disassembly == ""
  assert result.error == {"kind": "tool_unavailable", "message": "llvm-objdump not found"}


def test_debug_api_still_prints_returned_disassembly(monkeypatch, capsys):
  monkeypatch.setattr(compiler_amd, "amdgpu_disassemble_result",
                      lambda lib: compiler_amd.AMDDisassemblyResult(True, "isa", "/fake/objdump"))

  assert compiler_amd.amdgpu_disassemble(b"code") is None
  assert capsys.readouterr().out == "isa\n"


def test_debug_api_keeps_failure_fail_closed(monkeypatch):
  monkeypatch.setattr(compiler_amd, "amdgpu_disassemble_result",
                      lambda lib: compiler_amd.AMDDisassemblyResult(False,
                        error={"kind": "tool_unavailable", "message": "missing"}))

  with pytest.raises(RuntimeError, match="missing"):
    compiler_amd.amdgpu_disassemble(b"code")
