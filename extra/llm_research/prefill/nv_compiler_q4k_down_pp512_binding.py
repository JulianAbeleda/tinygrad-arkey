"""Graph-owned Q4_K FFN-down bindings for wide and experimental Stream-K assets."""
from dataclasses import dataclass
from functools import cache
import hashlib
from extra.llm_research.prefill.nv_compiler_q4k_down_asset import DownAsset, binding_for as _asset_for, supports, validate_streamk_inputs, M, N, K, PROJECTIONS_PER_MODEL
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import Ops

STREAMK_PARTIAL_FLOATS = 340*128*128

@dataclass(frozen=True)
class StreamKDownAsset:
  base: DownAsset
  main_program: object
  fixup_program: object
  reduction_slots: Tensor
  active_tiles: Tensor
  candidate_identity: str

  @property
  def producer(self): return self.base.producer

@cache
def _streamk_asset(device, unroll=None, sliced_fixup=False):
  from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import transform_compiler_q4k_to_streamk, active_fixup_source
  from extra.llm_research.prefill.nv_compiler_streamk_codegen import q4_down_fixup_map
  from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  base = _asset_for(device)
  sources = [u.arg for u in base.main_program.src if u.op is Ops.SOURCE]
  if len(sources) != 1: raise ValueError("Q4 down asset must retain one compiler source")
  source = transform_compiler_q4k_to_streamk(sources[0], unroll=unroll, tiles_n=32, k_blocks=192, output_stride=N, kernel_name="q4_down_streamk")
  fixup_source = active_fixup_source(max_contributors=3, sliced=sliced_fixup)
  compiler = NVRTCCompiler(Device[device].arch, ptx=False, cache_key="q4_down_graph_streamk_v1")
  def program(name, text, grid, block, outs, ins, vals=()):
    prg = native_nv_program(name, compiler.compile(text), global_size=grid, local_size=block,
      globals=tuple(range(5 if name == "q4_down_streamk" else 4)), outs=outs, ins=ins, vals=vals)
    return prg.replace(src=tuple(u.replace(arg=text) if u.op is Ops.SOURCE else u for u in prg.src))
  main = program("q4_down_streamk", source, (170,1,1), (32,2,4), (0,1,2), (3,4))
  fixup = program("q4k_imma_fixup_active", fixup_source, (128,4 if sliced_fixup else 1,1), (128 if sliced_fixup else 256,1,1), (0,), (1,2,3), (M,N))
  rows, active = q4_down_fixup_map()
  if len(rows) != 128 or len(active) != 128 or any(not 1 <= len(row) <= 3 for row in rows):
    raise ValueError("Q4 down fixup map does not cover every output tile")
  slots = Tensor([slot for row in rows for slot in (*row, *([-1]*(3-len(row))))], dtype=dtypes.int32, device=device).realize()
  active_tensor = Tensor(active, dtype=dtypes.int32, device=device).realize()
  identity = hashlib.sha256((source+fixup_source+repr(rows)).encode()).hexdigest()
  return StreamKDownAsset(base, main, fixup, slots, active_tensor, identity)

def _streamk_main(asset, out, partial, ids, words, record):
  validate_streamk_inputs(words, record)
  return out.uop_program(partial, ids, words, record, fxn=lambda *_:asset.main_program)

def _project(asset,x,words,*,model_family,role,weight_type="Q4_K"):
  if not supports(model_family=model_family,role=role,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device): raise ValueError("unsupported compiler Q4 down route")
  if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32: raise ValueError("Q4 down requires fp16 activation and uint32 weights")
  record=Tensor.empty((M*K+2*M*(K//32)*4)//4,dtype=dtypes.uint32,device=x.device)
  _,record=x.uop_program(record,fxn=lambda *_:asset.producer)
  if isinstance(asset, StreamKDownAsset):
    out = Tensor.empty(M*N, dtype=dtypes.float32, device=x.device)
    partial = Tensor.empty(STREAMK_PARTIAL_FLOATS, dtype=dtypes.float32, device=x.device)
    ids = Tensor.empty(340, dtype=dtypes.int32, device=x.device)
    out, partial, ids, words, record = _streamk_main(asset, out, partial, ids, words, record)
    out, partial, slots, active = out.uop_program(partial, asset.reduction_slots, asset.active_tiles, fxn=lambda *_:asset.fixup_program)
    return out.reshape(M,N)
  # The producer's graph output is a second AFTER value.  Keep the producer
  # materialized before handing the record to the frozen main: composing the
  # two CALLs in one lazy graph currently loses the dependency and the main
  # sees the zero-filled allocation (the 18-role population gate caught this).
  record.realize()
  # Invoke the frozen compiler PROGRAM directly.  Its ABI is validated at
  # compile time: logical output 0, record input 1, packed words input 2.
  out=Tensor.empty(M*N,dtype=dtypes.float32,device=x.device)
  out,record,words=out.uop_program(record,words,fxn=lambda *_:asset.main_program)
  return out.reshape(M,N)

class DownCapture:
  def __init__(self,asset): self.asset,self.cursor,self.epoch=asset,0,0
  def begin_trace(self): self.epoch+=1; self.cursor=0
  def prepare_records(self,count):
    if count!=PROJECTIONS_PER_MODEL: raise ValueError("Q4 down requires exactly 18 projections")
  @property
  def records(self): return range(self.cursor)
  @property
  def outputs(self): return range(self.cursor)
  def project(self,x,words,**kw):
    if self.epoch==0: raise RuntimeError("begin_trace must establish a capture epoch")
    if self.cursor>=PROJECTIONS_PER_MODEL: raise RuntimeError("Q4 down trace exceeded 18 projections")
    self.cursor+=1
    return _project(self.asset,x,words,**kw)

def binding_for(device="NV", *, variant="wide", streamk_unroll=None, tile_k=64, sliced_fixup=False):
  if variant not in ("wide", "streamk"): raise ValueError(f"unknown Q4 down variant {variant}")
  if streamk_unroll not in (None,1,2,4,8,16,32): raise ValueError("streamk_unroll must be one of 1,2,4,8,16,32")
  if variant == "streamk":
    if tile_k != 64: raise ValueError("streamk tile_k must remain 64")
    return _streamk_asset(device, streamk_unroll, sliced_fixup)
  if sliced_fixup: raise ValueError("sliced fixup requires streamk variant")
  return _asset_for(device, tile_k=tile_k)
def capture_for(device="NV", *, variant="wide", streamk_unroll=None, tile_k=64, sliced_fixup=False): return DownCapture(binding_for(device, variant=variant, streamk_unroll=streamk_unroll, tile_k=tile_k, sliced_fixup=sliced_fixup))
