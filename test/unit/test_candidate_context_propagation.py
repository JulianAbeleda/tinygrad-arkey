import pytest

from tinygrad.uop.ops import KernelInfo, UOp
from extra.qk.kernel_vocabulary import KernelCandidateContext
from tinygrad.codegen.opt import postrange


def _ctx(tag="candidate"):
  return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "a" * 64)


def test_typed_candidate_context_survives_sink_kernel_info():
  context = _ctx()
  sink = UOp.sink(arg=KernelInfo(name="typed", candidate_context=context))
  assert sink.arg.candidate_context is context
  assert sink.arg.candidate_context.canonical_identity == "a" * 64


def test_warmstart_candidate_state_restores_after_success():
  old = (postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS)
  with postrange.warmstart_candidate_state({("shape", 1): ("opt",)}, {("shape", 1): _ctx()}):
    assert postrange._WARMSTART_CANDIDATE_CONTEXTS[("shape", 1)].canonical_identity == "a" * 64
  assert (postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS) == old


def test_warmstart_candidate_state_restores_after_exception():
  old = (postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS)
  with pytest.raises(RuntimeError, match="boom"):
    with postrange.warmstart_candidate_state({("shape", 2): ("opt",)}, {("shape", 2): _ctx()}):
      raise RuntimeError("boom")
  assert (postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS) == old


def test_warmstart_candidate_context_requires_matching_opts():
  with pytest.raises(RuntimeError, match="lack schedule opts"):
    with postrange.warmstart_candidate_state({}, {("missing", 3): _ctx()}):
      pass
