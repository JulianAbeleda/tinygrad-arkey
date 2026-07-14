import pytest

from tinygrad.codegen import to_program, to_program_cache
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, KernelLDSWindow, KernelTileGeometry, UOp
from tinygrad.codegen.opt import postrange


def _sink(identity: str) -> UOp:
  out = UOp.param(0, dtypes.float.ptr(1))
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", identity)
  return out.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(dtypes.float, 1)).sink(
    arg=KernelInfo(name="candidate", opts_to_apply=(), candidate_context=context))

def _geometry() -> KernelTileGeometry:
  return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))


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


@pytest.mark.parametrize("factory,error", (
  (lambda: KernelLDSWindow("C", 0, 16, 16), "role"),
  (lambda: KernelLDSWindow("A", -16, 16, 16), "non-empty non-negative"),
  (lambda: KernelLDSWindow("A", 0, 15, 16), "b128 aligned"),
  (lambda: KernelLDSWindow("A", 0, 16, 0), "positive"),
  (lambda: KernelTileGeometry((128, 128), (4, 2), 256, 32, _geometry().lds_windows), "three positive"),
  (lambda: KernelTileGeometry([128, 128, 32], (4, 2), 256, 32, _geometry().lds_windows), "three positive"),
  (lambda: KernelTileGeometry((128, 128, 32), (4, 0), 256, 32, _geometry().lds_windows), "two positive"),
  (lambda: KernelTileGeometry((128, 128, 32), (4, 2), 32, 32, _geometry().lds_windows), "account for threads"),
  (lambda: KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("B", 0, 10240, 80), KernelLDSWindow("A", 10240, 20480, 80))), "ordered A and B"),
  (lambda: KernelTileGeometry((128, 128, 32), (4, 2), 256, 32, (object(), object())), "frozen KernelLDSWindow"),
  (lambda: KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 16, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80))), "contiguous from byte zero"),
))
def test_kernel_tile_geometry_rejects_malformed_fields(factory, error):
  with pytest.raises(ValueError, match=error): factory()


def test_candidate_geometry_survives_program_and_warmstart_propagation(monkeypatch):
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "5" * 64, _geometry())
  sink = _sink("4" * 64).replace(arg=KernelInfo(name="candidate", opts_to_apply=(), candidate_context=context))
  program = to_program(sink, PythonRenderer(Target("PYTHON")))
  assert program.src[0].arg.candidate_context.geometry == _geometry()
  key = (frozenset(), 1)
  monkeypatch.setattr(postrange, "_WARMSTART_OPTS", {key: ()})
  monkeypatch.setattr(postrange, "_WARMSTART_CANDIDATE_CONTEXTS", {key: context})
  optimized = postrange.apply_opts(_sink("4" * 64).replace(arg=KernelInfo()), PythonRenderer(Target("PYTHON")))
  assert optimized.arg.candidate_context.geometry == _geometry()


def test_warmstart_candidate_context_reaches_optimized_kernel(monkeypatch):
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "3" * 64)
  key = (frozenset(), 1)
  monkeypatch.setattr(postrange, "_WARMSTART_OPTS", {key: ()})
  monkeypatch.setattr(postrange, "_WARMSTART_CANDIDATE_CONTEXTS", {key: context})
  optimized = postrange.apply_opts(_sink("4" * 64).replace(arg=KernelInfo()), PythonRenderer(Target("PYTHON")))
  assert optimized.arg.candidate_context == context


def test_warmstart_candidate_state_restores_profiles_sequentially(monkeypatch):
  old_key, first_key, second_key = (frozenset({1}), 1), (frozenset({2}), 2), (frozenset({3}), 3)
  old_context, first_context, second_context = (_sink("a" * 64).arg.candidate_context,
                                                _sink("b" * 64).arg.candidate_context,
                                                _sink("c" * 64).arg.candidate_context)
  old_opts, old_contexts = {old_key: ()}, {old_key: old_context}
  monkeypatch.setattr(postrange, "_WARMSTART_OPTS", old_opts)
  monkeypatch.setattr(postrange, "_WARMSTART_CANDIDATE_CONTEXTS", old_contexts)
  for key, context in ((first_key, first_context), (second_key, second_context)):
    with postrange.warmstart_candidate_state({key: ()}, {key: context}):
      assert postrange._WARMSTART_OPTS == {key: ()}
      assert postrange._WARMSTART_CANDIDATE_CONTEXTS == {key: context}
    assert (postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS) == (old_opts, old_contexts)


def test_warmstart_candidate_state_fails_closed_on_context_collision(monkeypatch):
  key = (frozenset({2}), 2)
  monkeypatch.setattr(postrange, "_WARMSTART_CANDIDATE_CONTEXTS", {key: _sink("d" * 64).arg.candidate_context})
  with pytest.raises(RuntimeError, match="candidate context collision"):
    with postrange.warmstart_candidate_state({key: ()}, {key: _sink("e" * 64).arg.candidate_context}): pass


def test_warmstart_candidate_state_restores_after_capture_failure(monkeypatch):
  sentinel = object()
  monkeypatch.setattr(postrange, "_WARMSTART_OPTS", sentinel)
  with pytest.raises(ValueError, match="capture failed"):
    with postrange.warmstart_candidate_state({}): raise ValueError("capture failed")
  assert postrange._WARMSTART_OPTS is sentinel
