"""Graph-owned Q4_K FFN-down binding backed by immutable tileK64 asset."""
from extra.llm_research.prefill.nv_compiler_q4k_down_asset import DownAsset, binding_for as _asset_for, supports, M, N, K, PROJECTIONS_PER_MODEL
from tinygrad import Tensor, dtypes

def _project(asset,x,words,*,model_family,role,weight_type="Q4_K"):
  if not supports(model_family=model_family,role=role,weight_type=weight_type,m=x.shape[0],n=N,k=x.shape[1],device=x.device): raise ValueError("unsupported compiler Q4 down route")
  if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32: raise ValueError("Q4 down requires fp16 activation and uint32 weights")
  record=Tensor.empty((M*K+2*M*(K//32)*4)//4,dtype=dtypes.uint32,device=x.device)
  _,record=x.uop_program(record,fxn=lambda *_:asset.producer)
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

def binding_for(device="NV"): return _asset_for(device)
def capture_for(device="NV"): return DownCapture(binding_for(device))
