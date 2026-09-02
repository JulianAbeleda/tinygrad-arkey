import contextlib, functools, hashlib, pathlib, signal, subprocess, tempfile

import pytest

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, Ops, PostBarrierRegion, RegionLoad, RegionLoadBridge, UOp
from extra.llm_research.prefill.nv_ptx_phase_boundary_microgate import analyze_sass, CUOBJDUMP, NVDISASM


TARGET=Target.parse("NV:CUDA:sm_120")
UNMARKED_SHA256="19d2dcc6ec58cd214fcc0f09141b453fbb761f2a6acd973b0020ef460f1b6492"
ORDINARY_REGION_SHA256="fd6cf6b86eca16d01f493a5f34236d1dfff8c5dc63913cc6abc8b61c5a369b61"


@contextlib.contextmanager
def _timeout(seconds:int):
  def expired(_signum,_frame): raise TimeoutError(f"CUDA region bridge microgate exceeded {seconds}s")
  old=signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,seconds)
  try: yield
  finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)


def _ordinary_ast():
  lid=UOp.special(32,"lidx0"); source=UOp.placeholder((32,),dtypes.uint,1)
  scratch=UOp.placeholder((32,),dtypes.uint,2,addrspace=AddrSpace.LOCAL)
  anchor=UOp(Ops.BARRIER,dtypes.void)
  region=anchor.post_barrier_region(UOp.const(dtypes.bool,True),workgroup_uniform=True)
  loaded=source[lid].load().load_in_region(region); stored=scratch[lid].store(loaded)
  return UOp.sink(region.end_region(stored),arg=KernelInfo(name="lexical_load_region",opts_to_apply=()))


def _unmarked_ast():
  lid=UOp.special(32,"lidx0"); out=UOp.placeholder((32,),dtypes.uint,0); source=UOp.placeholder((32,),dtypes.uint,1)
  return UOp.sink(out[lid].store(source[lid].load()),arg=KernelInfo(name="bridge_unmarked_control",opts_to_apply=()))


def _bridge_ast(copies:int=18, *, large:bool=False, qualified:bool=False, mutable:bool=False, extra_consumer:bool=False):
  lid=UOp.special(256,"lidx0"); out=UOp.placeholder((256,),dtypes.uint,0)
  source=UOp.placeholder((18*256,),dtypes.uint,1)
  if qualified: source=source.const_restrict()
  scratch=UOp.placeholder((18*256,),dtypes.uint,2,addrspace=AddrSpace.LOCAL)
  accs=[]
  for i in range(64 if large else 1):
    acc=UOp(Ops.CUSTOMI,dtypes.float,(lid,),f"__uint2float_rn((((unsigned int){{0}}) ^ 0x{((i+1)*0x45D9F3B)&0xffffffff:08x}u) & 0xffffu)")
    for rnd in range(8 if large else 1):
      mul=1.0001+((i*7+rnd*3)%29)*0.000013; add=0.125+((i*13+rnd*11)%127)*0.03125
      acc=UOp(Ops.CUSTOMI,dtypes.float,(acc,),f"__fmaf_rn({{0}}, {mul:.9f}f, {add:.9f}f)")
    accs.append(acc)
  anchor=UOp(Ops.BARRIER,dtypes.void,tuple(accs))
  region=anchor.post_barrier_region(UOp.const(dtypes.bool,True),workgroup_uniform=True)
  loads=[source[lid+i*256].load().load_in_region_bridge(region) for i in range(copies)]
  stores=[scratch[lid+i*256].store(load) for i,load in enumerate(loads)]
  if extra_consumer: stores.append(scratch[lid+copies*256].store(loads[0]))
  ended=region.end_region(*stores)
  publication=UOp(Ops.BARRIER,dtypes.void,(ended,))
  read_lane=(lid+1)%256; published_scratch=scratch.after(publication)
  shared=published_scratch[read_lane].load()
  for i in range(1,18): shared=shared ^ published_scratch[read_lane+i*256].load()
  total=accs[0]
  for acc in accs[1:]: total=total+acc
  roots=[out[lid].store(total.bitcast(dtypes.uint)^shared)]
  if mutable: roots.append(source[lid].store(UOp.const(dtypes.uint,0)))
  return UOp.sink(*roots,arg=KernelInfo(name="cuda_region_load_bridge",opts_to_apply=()))


def _source(ast:UOp, *, fused:bool=False, warp_fence:bool=False) -> str:
  class FusedCUDARenderer(CUDARenderer):
    region_load_bridge_owns_barrier=True
    region_load_bridge_warp_fence=warp_fence
  renderer=(FusedCUDARenderer if fused else CUDARenderer)(TARGET)
  program=to_program(ast,renderer)
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


@functools.lru_cache(None)
def _large_source() -> str: return _source(_bridge_ast(large=True))


def test_bridge_api_is_explicit_shared_and_outside_index():
  topo=_bridge_ast().toposort()
  markers=[u for u in topo if u.op is Ops.AFTER and isinstance(u.arg,RegionLoadBridge)]
  loads=[u for u in topo if u.op is Ops.LOAD and any(s in markers for s in u.src[1:])]
  assert len(markers)==1 and len(loads)==18 and all(load.src[1] is markers[0] for load in loads)
  assert all(markers[0] not in load.src[0].backward_slice_with_self for load in loads)
  assert not any(isinstance(u.arg,RegionLoad) for u in topo)


def test_bridge_marker_survives_full_rewrite_and_stays_outside_index():
  rewritten=full_rewrite_to_sink(_bridge_ast(),CUDARenderer(TARGET))
  topo=rewritten.toposort()
  markers=[u for u in topo if u.op is Ops.AFTER and isinstance(u.arg,RegionLoadBridge)]
  loads=[u for u in topo if u.op is Ops.LOAD and any(s in markers for s in u.src[1:])]
  assert len(markers)==1 and len(loads)==18
  assert markers[0].src[0].op is Ops.IF and isinstance(markers[0].src[0].arg,PostBarrierRegion)
  assert all(markers[0] not in load.src[0].backward_slice_with_self for load in loads)


def test_unmarked_and_ordinary_region_sources_remain_byte_identical():
  assert hashlib.sha256(_source(_unmarked_ast()).encode()).hexdigest()==UNMARKED_SHA256
  assert hashlib.sha256(_source(_ordinary_ast()).encode()).hexdigest()==ORDINARY_REGION_SHA256


def test_bridge_source_is_two_ptx_blocks_around_same_c_barrier_without_qualifiers_or_named_loads():
  source=_large_source()
  first_load=source.index("ld.global.u32"); overwrite=source.index("__syncthreads();",first_load)
  first_store=source.index("st.shared.u32",overwrite); publication=source.index("__syncthreads();",overwrite+1)
  assert first_load < overwrite < first_store < publication
  assert source.count("asm volatile(")==2 and source.count("ld.global.u32")==source.count("st.shared.u32")==18
  assert source.count("__syncthreads();")==2 and "bar.sync" not in source
  assert "const unsigned int *" not in source and "__restrict__" not in source
  assert not any("unsigned int val" in line and "data1_" in line for line in source.splitlines())
  assert source.count("__fmaf_rn(")==64*8 and source.count("region_bridge_copy")==18*3


def test_fused_bridge_owns_the_same_barrier_inside_one_ptx_region():
  source=_source(_bridge_ast(large=True),fused=True)
  first_load=source.index("ld.global.u32");barrier=source.index("bar.sync 0",first_load);first_store=source.index("st.shared.u32",barrier)
  assert first_load < barrier < first_store
  assert source.count("asm volatile(")==1 and source.count("ld.global.u32")==source.count("st.shared.u32")==18
  assert source.count("bar.sync 0")==1 and source.count("__syncthreads();")==1


def test_fused_bridge_warp_fence_is_explicit_and_inside_the_region():
  source=_source(_bridge_ast(large=True),fused=True,warp_fence=True)
  fence=source.index("bar.warp.sync 0xffffffff");first_load=source.index("ld.global.u32");barrier=source.index("bar.sync 0")
  assert fence < first_load < barrier and source.count("bar.warp.sync 0xffffffff")==1


def test_bridge_fails_closed_for_unsupported_or_malformed_groups():
  renderer=CUDARenderer(TARGET); renderer.supports_region_load_bridge=False
  with pytest.raises(RuntimeError,match="cannot preserve split region-load"):
    full_rewrite_to_sink(_bridge_ast(),renderer)
  with pytest.raises(RuntimeError,match="exactly 18 scalar copies"): _source(_bridge_ast(17))
  with pytest.raises(RuntimeError,match="immutable unqualified GLOBAL source"): _source(_bridge_ast(qualified=True))
  with pytest.raises(RuntimeError,match="immutable unqualified GLOBAL source"): _source(_bridge_ast(mutable=True))
  with pytest.raises(RuntimeError,match="one unique direct STORE consumer"): _source(_bridge_ast(extra_consumer=True))


def test_bridge_api_requires_constant_true_uniform_region_and_scalar32_load():
  source=UOp.placeholder((32,),dtypes.uint,1); anchor=UOp(Ops.BARRIER,dtypes.void)
  with pytest.raises(ValueError,match="constant-true workgroup-uniform"):
    source[0].load().load_in_region_bridge(anchor.post_barrier_region(UOp.const(dtypes.bool,True)))
  with pytest.raises(ValueError,match="constant-true workgroup-uniform"):
    source[0].load().load_in_region_bridge(anchor.post_barrier_region(UOp.const(dtypes.bool,False),workgroup_uniform=True))
  half=UOp.placeholder((32,),dtypes.half,2)
  region=anchor.post_barrier_region(UOp.const(dtypes.bool,True),workgroup_uniform=True)
  with pytest.raises(ValueError,match="scalar 32-bit LOAD"): half[0].load().load_in_region_bridge(region)


@pytest.mark.skipif(not NVDISASM.is_file() or not CUOBJDUMP.is_file(),reason="CUDA disassembly tools unavailable")
def test_bridge_large_body_nv_sass_matches_split_microgate_contract():
  source=_large_source()
  with _timeout(45):
    binaries=[NVRTCCompiler("sm_120",ptx=False,cache_key=f"cuda_region_load_bridge_r{i}").compile(source) for i in range(3)]
    hashes=[hashlib.sha256(x).hexdigest() for x in binaries]
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
      f.write(binaries[0]); f.flush()
      sass=subprocess.run([str(NVDISASM),"-c",f.name],check=True,text=True,stdout=subprocess.PIPE,timeout=15).stdout
      resources=subprocess.run([str(CUOBJDUMP),"--dump-resource-usage",f.name],check=True,text=True,stdout=subprocess.PIPE,timeout=15).stdout
  analysis=analyze_sass(sass,resources)
  assert len(set(hashes))==1
  assert analysis["base_hard_pass"], analysis
  assert analysis["families"].get("LDG")==analysis["families"].get("STS")==18
  assert analysis["ldg_opcodes"]=={"LDG.E":18} and analysis["first_ldg_to_first_sts_instructions"]<=160
