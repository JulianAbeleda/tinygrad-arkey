from extra.llm_research.prefill.nv_q4_imma_provider import (
  M, N, K, DYNAMIC_SHARED_BYTES, PARTIAL_SLOTS, Provider, main_source, fixup_source, production_slotmap,
)


def test_provider_is_shape_locked_and_one_symbol_per_cubin():
  assert (M, N, K) == (512, 12288, 4096)
  assert DYNAMIC_SHARED_BYTES == 57856
  assert PARTIAL_SLOTS == 340
  main, fixup = main_source(), fixup_source()
  assert main.count('extern "C" __global__') == 1
  assert fixup.count('extern "C" __global__') == 1
  assert "total=6144,owners=170" in main
  assert "bar.sync 0, 256" in main
  assert "atomicAdd" not in fixup


def test_provider_boundary_map_is_deterministic():
  assert production_slotmap().tobytes() == production_slotmap().tobytes()


def test_provider_accepts_allocator_padding_but_rejects_undersize():
  class B:
    def __init__(self, size): self.size=size
  required=(M*N*4,PARTIAL_SLOTS*128*128*4,PARTIAL_SLOTS*4,N*(K//256)*36*4,M*K,M*(K//32)*4,M*(K//32)*4,384*2*4)
  p=Provider(None,None,production_slotmap())
  p.validate_buffers(*(B(x+4096) for x in required))
  bad=[B(x) for x in required];bad[2]=B(required[2]-1)
  try:p.validate_buffers(*bad)
  except ValueError:pass
  else:raise AssertionError("undersized ABI buffer accepted")
