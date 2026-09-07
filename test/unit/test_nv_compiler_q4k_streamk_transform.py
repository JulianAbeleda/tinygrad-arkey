from pathlib import Path
import re
import pytest

from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import _pack_q4_publication, active_fixup_source, transform_compiler_q4k_to_streamk

FIXTURE=Path("/home/ubuntu/boltbeam-runs/packed-q8-completion-20260830/q4-emitter-baseline.cu")

def _q4_publication_fixture() -> str:
  lines=[]
  for val,bank in (("val22",0),("val23",5120)):
    for comp_i,comp in enumerate("xyzw"):
      for shift_i,shift in enumerate(("cast4","cast1","cast2","cast3")):
        offset=bank+comp_i*4+shift_i; addr="alu8" if offset==0 else f"(alu8+{offset})"
        lines.append(f"    *(buf1+{addr}) = ((signed char)((({val}.{comp}>>{shift})&15u)));" )
  return "\n".join(lines)+"\n    __syncthreads();\n"

def test_q4_packed_publication_matches_exact_ownership_and_fails_closed():
  fixture=_q4_publication_fixture()
  packed=_pack_q4_publication(fixture)
  assert packed.count("reinterpret_cast<unsigned int*>(buf1+")==8
  assert "signed char" not in packed
  with pytest.raises(ValueError,match="exact lane/address ownership"):
    _pack_q4_publication(fixture.replace("(alu8+1)","(alu8+2)",1))
  with pytest.raises(ValueError,match="32 scalar nibble stores"):
    _pack_q4_publication(fixture.replace(fixture.splitlines()[0]+"\n","",1))

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

def test_swapped_transform_exports_semantic_weight_then_record_abi():
  if not FIXTURE.exists(): return
  original=FIXTURE.read_text()
  # Simulate swapped compiler ownership by renaming slot capacities: weight is
  # slot 1 and activation record is slot 2. Runtime remains (weight, record).
  original=original.replace("data1_655360", "data1_7077888").replace("data2_7077888", "data2_594432")
  transformed=transform_compiler_q4k_to_streamk(original,operand_order="weight_a_activation_b")
  signature=transformed[transformed.index("q4k_imma_stream("):transformed.index(") {",transformed.index("q4k_imma_stream("))]
  assert signature.index("data1_7077888") < signature.index("data2_594432")

def test_swapped_physical_grid_drives_owner_work_units():
  if not FIXTURE.exists(): return
  swapped=FIXTURE.read_text().replace("int gidx0 = blockIdx.x; /* 96 */", "int gidx0 = blockIdx.x; /* 4 */") \
    .replace("int gidx1 = blockIdx.y; /* 4 */", "int gidx1 = blockIdx.y; /* 96 */")
  transformed=transform_compiler_q4k_to_streamk(swapped,tiles_n=4,tiles_m=96,output_stride=512,
    operand_order="weight_a_activation_b")
  assert "owner*24576" in transformed and "int owner = blockIdx.x" in transformed

def test_partial_store_offset_remap_is_not_cascaded():
  from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import _partial_store_block
  source="  int alu242 = old;\n  x=data0_6291456+alu242+4096;\n  y=data0_6291456+alu242+16384;\n"
  transformed=_partial_store_block(source,output_stride=512)
  assert "alu242+1024" in transformed and "alu242+4096" in transformed

def test_swapped_xor4_epilogue_maps_each_mma_value_once():
  coords=[]; sources=[]
  for lane in range(32):
    g,q,hi=lane>>2,lane&3,(lane>>2)&1
    for p in range(2):
      for half in range(2):
        coords.append((2*q+p,g+7*hi+half))
        # hi0 consumes C[p] from itself/XOR4; hi1 consumes C[p+2].
        source_lane=lane if half==hi else lane^4
        source_elem=p+2*hi
        sources.append((source_lane,source_elem))
  assert len(coords)==len(set(coords))==128
  assert len(sources)==len(set(sources))==128
  assert set(coords)=={(r,c) for r in range(8) for c in range(16)}
  assert set(sources)=={(lane,e) for lane in range(32) for e in range(4)}

def test_swapped_xor4_epilogue_keeps_direct_and_partial_local_indices_identical():
  from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import _logical_transpose_store_blocks
  source=""; value=0
  for row in range(0,64,8):
    for col in range(0,32,8):
      off=row*512+col; addr="alu247" if off==0 else f"(alu247+{off})"
      source+=f"  *((float2*)((data0_6291456+{addr}))) = make_float2((*(buf0+{value})),(*(buf0+{value+32})));\n";value+=1
  direct,partial=_logical_transpose_store_blocks(source)
  assert direct.count("float2*)")==partial.count("float2*)")==32
  assert direct.count("__shfl_xor_sync")==partial.count("__shfl_xor_sync")==64
  # Both destinations use the same local row/column expressions.
  assert "((lidx2<<5)+(alu5<<1)+" in direct and "((lidx2<<5)+(alu5<<1)+" in partial
  assert "((lidx1<<6)+" in direct and "((lidx1<<6)+" in partial

def test_wide_swapped_epilogue_rewrites_only_terminal_stores():
  from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import coalesce_swapped_direct_source
  source='extern "C" __global__ void __launch_bounds__(256) k(float* data0_2097152, unsigned int* data1_1, unsigned int* data2_1) {\n'
  source+='  int alu244 = 0;\n'
  value=0
  for row in range(0,64,8):
    for col in range(0,32,8):
      off=row*512+col; addr="alu244" if off==0 else f"(alu244+{off})"
      source+=f"  *((float2*)((data0_2097152+{addr}))) = make_float2((*(buf0+{value})),(*(buf0+{value+32})));\n";value+=1
  source+='}\n'
  transformed=coalesce_swapped_direct_source(source,logical_stride=4096)
  assert "int alu244 = 0;" in transformed
  assert transformed.count("__shfl_xor_sync")==64 and transformed.count("float2*)")==32
  assert "*4096" in transformed and "*12288" not in transformed
  assert "data0_2097152+(alu244" not in transformed

def test_streamk_q_record_capture_state_is_trace_local():
  from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import CompilerQ4StreamKCapture
  capture=CompilerQ4StreamKCapture(None,None,None,None,None,"id",None,1,q_records=[object()])
  fresh=capture.new_capture()
  assert fresh.q_records==[] and capture.q_records!=fresh.q_records
  assert all(getattr(fresh,name)==[] and getattr(fresh,name) is not getattr(capture,name)
             for name in ("records","outputs","partials","partial_ids"))
  fresh.q_records.append(object());fresh.begin_trace()
  assert fresh.q_records==[] and fresh.cursor==0 and fresh.pair_record is None

def test_transform_accepts_renderer_renumbered_output_index():
  if not FIXTURE.exists(): return
  source=FIXTURE.read_text().replace("alu242", "alu246")
  transformed=transform_compiler_q4k_to_streamk(source)
  assert "int alu246 = ((alu5<<1)+(lidx2<<5)+(alu2*128)+(lidx1*8192));" in transformed
  assert "partials+(slot*16384)+(alu246" in transformed

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

def test_interleaved_wmma_schedule_bounds_results_and_preserves_updates():
  if not FIXTURE.exists(): return
  original=transform_compiler_q4k_to_streamk(FIXTURE.read_text())
  transformed=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),interleave_wmma_updates=True)
  assert transformed.count("int4 wmma")==32 and transformed.count("(*(buf0+")==original.count("(*(buf0+")
  first=transformed.index("int4 wmma28 =")
  last=transformed.index("(*(buf0+51)) =",first)
  assert transformed.count("int4 wmma",first,last)==8
  assert all(transformed.count(f"float cast{i} =")==1 for i in range(33,97))
  for i in range(64):
    needle=f"(*(buf0+{i})) ="
    assert transformed[transformed.index(needle):transformed.index(needle)+len(original[original.index(needle):].splitlines()[0])] == \
      original[original.index(needle):].splitlines()[0]

def test_interleaved_wmma_schedule_derives_renderer_scale_numbers_and_fails_closed():
  if not FIXTURE.exists(): return
  source=FIXTURE.read_text()
  shifted=re.sub(r"\bcast(3[3-9]|[4-8][0-9]|9[0-6])\b",lambda m:f"cast{int(m.group(1))+100}",source)
  transformed=transform_compiler_q4k_to_streamk(shifted,interleave_wmma_updates=True)
  assert transformed.count("int4 wmma")==32
  assert all(transformed.count(f"float cast{i} =")==1 for i in range(133,197))
  broken=re.sub(r"^    float cast133 = .*;\n","",shifted,count=1,flags=re.M)
  with pytest.raises(ValueError,match="32 wmmas, 64 scales, and 64 updates"):
    transform_compiler_q4k_to_streamk(broken,interleave_wmma_updates=True)


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


def test_streamk_role_cursor_supports_one_and_two_role_populations():
  from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import CompilerQ4StreamKCapture
  one=CompilerQ4StreamKCapture(None,None,None,None,None,"id",None,1,population=36,roles=("attn_output",))
  two=CompilerQ4StreamKCapture(None,None,None,None,None,"id",None,1,population=72,roles=("attn_q","attn_output"))
  assert [one.roles[i%len(one.roles)] for i in range(4)]==["attn_output"]*4
  assert [two.roles[i%len(two.roles)] for i in range(4)]==["attn_q","attn_output","attn_q","attn_output"]
