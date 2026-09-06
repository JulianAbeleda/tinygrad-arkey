from pathlib import Path

from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import active_fixup_source, transform_compiler_q4k_to_streamk

FIXTURE=Path("/home/ubuntu/boltbeam-runs/packed-q8-completion-20260830/q4-emitter-baseline.cu")

def test_transform_preserves_imma_body_and_adds_exact_owner_workspace_contract():
  if not FIXTURE.exists(): return
  original=FIXTURE.read_text()
  transformed=transform_compiler_q4k_to_streamk(original)
  assert transformed.count("mma.sync.aligned.m16n8k32") == original.count("mma.sync.aligned.m16n8k32")
  assert 'q4k_imma_stream(' in transformed
  assert "blockIdx.x; /* 170 persistent owners */" in transformed
  assert "for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++)" in transformed
  assert "partial_ids[slot]=tile" in transformed
  assert "partial_ids[owner*2]=-1" in transformed
  assert "partials+(slot*16384)" in transformed
  assert "owner*2+((owner_tail&&owner_has_head)?1:0)" in transformed

def test_active_fixup_uses_active_tile_map_and_guards_missing_slots():
  source=active_fixup_source()
  assert "tile=active[blockIdx.x]" in source and "if(s0<0)return" in source

def test_transform_owns_qualified_unroll_choice():
  if not FIXTURE.exists(): return
  transformed=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),unroll=8)
  assert "#pragma unroll 8\n  for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++)" in transformed

def test_double_buffer_uses_parity_bank_and_keeps_publish_barrier():
  if not FIXTURE.exists(): return
  original=FIXTURE.read_text()
  transformed=transform_compiler_q4k_to_streamk(original,double_buffer=True)
  assert "signed char buf1[40960]" in transformed
  assert transformed.count("__syncthreads();")==1
  assert "buf1+((Ridx0&1)*20480)+" in transformed

def test_fragment_load_to_use_moves_exact_q4_group_after_q8_loads():
  if not FIXTURE.exists(): return
  transformed=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),fragment_load_to_use=True)
  assert transformed.index("unsigned int val23 =") < transformed.index("unsigned int val0 =")
  assert transformed.index("uint4 val25 =") < transformed.index("unsigned int val0 =")
  assert transformed.index("unsigned int val10 =") < transformed.index("__syncthreads();")
  assert transformed.count("unsigned int val0 =")==1 and transformed.count("unsigned int val10 =")==1

def test_shared_load_to_pack_stages_each_scalar_at_its_single_consumer():
  if not FIXTURE.exists(): return
  transformed=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),shared_load_to_pack=True)
  lines=transformed.splitlines()
  assert lines.index(next(x for x in lines if "signed char val33 =" in x))+16 == \
    lines.index(next(x for x in lines if "signed_char16 cast17 =" in x))
  assert lines.index(next(x for x in lines if "signed char val153 =" in x))+8 == \
    lines.index(next(x for x in lines if "signed_char8 cast24 =" in x))
  assert transformed.count("signed char val26 =")==1 and transformed.count("signed char val345 =")==1

def test_shared_load_to_pack_can_stage_fragments_or_scales_independently():
  if not FIXTURE.exists(): return
  fragments=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),shared_load_to_pack="fragments")
  scales=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),shared_load_to_pack="scales")
  assert fragments.index("signed char val33 =") > fragments.index("__syncthreads();")
  assert fragments.index("signed char val218 =") < fragments.index("signed_char16 cast17 =")
  assert scales.index("signed char val33 =") < scales.index("signed_char16 cast17 =")
  assert scales.index("signed char val218 =") > scales.index("int4 wmma31 =")


def test_sliced_fixup_matches_same_partials_on_nv():
  import numpy as np
  import pytest
  from tinygrad import Device, Tensor, dtypes
  if Device.DEFAULT.split(':')[0] != 'NV': pytest.skip('requires NV')
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  from extra.llm_research.prefill.nv_native_program_uop import native_nv_program, call_native
  # Distinct values across all four slices and three contributors expose an
  # omitted slice offset, transposition, or incorrect reduction order.
  partial_np = np.random.default_rng(42).standard_normal((3,128,128)).astype(np.float32)
  partial = Tensor(partial_np.reshape(-1), device='NV').realize()
  slots = Tensor([2,0,1], dtype=dtypes.int32, device='NV').realize()
  active = Tensor([0], dtype=dtypes.int32, device='NV').realize()
  expected = (partial_np[2]+partial_np[0])+partial_np[1]
  compiler = NVRTCCompiler(Device['NV'].arch, ptx=False)
  for sliced in (False,True):
    out = Tensor.empty(128*128, dtype=dtypes.float32, device='NV').realize()
    prg = native_nv_program('q4k_imma_fixup_active', compiler.compile(active_fixup_source(max_contributors=3,sliced=sliced)),
      global_size=(1,4 if sliced else 1,1), local_size=(128 if sliced else 256,1,1),
      globals=(0,1,2,3), outs=(0,), ins=(1,2,3), vals=(128,128))
    call_native(prg,out,partial,slots,active,wait=True)
    np.testing.assert_array_equal(out.numpy().reshape(128,128),expected)
