import pytest
from tinygrad.uop.ops import KernelCandidateContext, Ops
from extra.qk.wmma_pipe_spec import WMMAPipeSpec, build_wmma_pipe_barrier_chain, build_wmma_pipe_wait_chain
from tinygrad.helpers import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.uop.spec import spec_tensor, type_verify


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

def test_attn_qo_wait_chain_has_proven_edges_and_no_raw_isa():
  sink = build_wmma_pipe_wait_chain(_spec(), KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "c" * 64))
  ops = sink.toposort()
  waits = [u for u in ops if u.op is Ops.WAIT]
  assert len(waits) == 1 and waits[0].arg.vmcnt == 0
  assert waits[0].tag[:2] == ("wait_coverage", (("A", 0, 1), ("B", 0, 1)))
  assert waits[0].tag[3] == (("global_load_A", "wmma", "A", 0, 1), ("global_load_B", "wmma", "B", 0, 1))
  assert not any(u.op is Ops.INS for u in ops)
  assert waits[0] in next(u for u in ops if u.op is Ops.STORE).backward_slice

def test_attn_qo_wait_node_passes_spec_and_llvm_intrinsic_compiles():
  sink = build_wmma_pipe_wait_chain(_spec(), KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "d" * 64))
  type_verify(sink, spec_tensor)
  wait = next(u for u in sink.toposort() if u.op is Ops.WAIT)
  ren = AMDLLVMRenderer(Target.parse("AMD:LLVM:gfx1100"))
  src = ren.render([wait])
  assert "@llvm.amdgcn.s.waitcnt" in src
  assert ren.compiler.compile_to_obj(src)
