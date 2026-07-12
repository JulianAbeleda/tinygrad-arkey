import pytest

from tinygrad.codegen import to_program, to_program_cache
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, UOp
from tinygrad.codegen.opt import postrange


def _sink(identity: str) -> UOp:
  out = UOp.param(0, dtypes.float.ptr(1))
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", identity)
  return out.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(dtypes.float, 1)).sink(
    arg=KernelInfo(name="candidate", opts_to_apply=(), candidate_context=context))


def test_two_full_kernel_candidates_remain_distinct_in_one_process():
  renderer = PythonRenderer(Target("PYTHON"))
  first, second = _sink("1" * 64), _sink("2" * 64)
  assert first.key != second.key
  to_program_cache.clear()
  first_program, second_program = to_program(first, renderer), to_program(second, renderer)
  assert first_program is not second_program
  assert first_program.key != second_program.key
  assert first_program.src[0].arg.candidate_context.canonical_identity == "1" * 64
  assert second_program.src[0].arg.candidate_context.canonical_identity == "2" * 64


def test_candidate_context_enters_compiler_cache_identity(monkeypatch):
  compiler = Compiler("candidate_context_test")
  seen = []
  monkeypatch.setattr("tinygrad.device.diskcache_get", lambda table, key: seen.append(key))
  monkeypatch.setattr("tinygrad.device.diskcache_put", lambda table, key, value: value)
  compiler.compile_cached("same source", ("boltbeam.full_kernel_candidate.v1", "1" * 64))
  compiler.compile_cached("same source", ("boltbeam.full_kernel_candidate.v1", "2" * 64))
  assert seen[0] != seen[1]
  assert all(isinstance(key, str) and key.startswith("candidate:boltbeam.full_kernel_candidate.v1:") for key in seen)


def test_legacy_default_cache_identity_is_neutral(monkeypatch):
  compiler = Compiler("candidate_context_legacy_test")
  seen = []
  monkeypatch.setattr("tinygrad.device.diskcache_get", lambda table, key: seen.append(key))
  monkeypatch.setattr("tinygrad.device.diskcache_put", lambda table, key, value: value)
  compiler.compile_cached("legacy source")
  assert seen == ["legacy source"]


def test_unsupported_or_malformed_candidate_context_fails_closed():
  with pytest.raises(ValueError, match="unsupported kernel candidate context schema"):
    KernelCandidateContext("boltbeam.full_kernel_candidate.v2", "1" * 64)
  with pytest.raises(ValueError, match="lowercase SHA-256"):
    KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "A" * 64)


def test_warmstart_candidate_context_reaches_optimized_kernel(monkeypatch):
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "3" * 64)
  key = (frozenset(), 1)
  monkeypatch.setattr(postrange, "_WARMSTART_OPTS", {key: ()})
  monkeypatch.setattr(postrange, "_WARMSTART_CANDIDATE_CONTEXTS", {key: context})
  optimized = postrange.apply_opts(_sink("4" * 64).replace(arg=KernelInfo()), PythonRenderer(Target("PYTHON")))
  assert optimized.arg.candidate_context == context
