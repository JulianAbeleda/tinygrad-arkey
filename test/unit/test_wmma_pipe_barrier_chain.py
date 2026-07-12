import pytest
from tinygrad.uop.ops import KernelCandidateContext, Ops
from extra.qk.wmma_pipe_spec import WMMAPipeSpec, build_wmma_pipe_barrier_chain


def _spec(role="attn_qo", shape=(512, 4096, 4096)):
  return WMMAPipeSpec(*shape, tile_m=128, tile_n=128, role=role)


def test_attn_qo_barrier_chain_has_ordered_graph_and_typed_context():
  sink = build_wmma_pipe_barrier_chain(_spec(), KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "a" * 64))
  ops = sink.toposort()
  assert not any(u.op is Ops.INS for u in ops)
  barrier = next(u for u in ops if u.op is Ops.BARRIER)
  wmma = next(u for u in ops if u.op is Ops.WMMA)
  store = next(u for u in ops if u.op is Ops.STORE)
  assert barrier in wmma.backward_slice
  assert wmma in store.backward_slice
  assert sink.arg.candidate_context.canonical_identity == "a" * 64




def test_barrier_chain_rejects_non_attn_qo_shapes():
  with pytest.raises(ValueError, match="only supports attn_qo"):
    build_wmma_pipe_barrier_chain(_spec(role="attn_kv", shape=(512, 1024, 4096)),
      KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "b" * 64))
