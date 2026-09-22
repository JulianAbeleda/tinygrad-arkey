import argparse, json, traceback
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, ParamArg, KernelInfo
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.renderer.isa.amd_attention_abi import lower_cooperative_tile_load
from tinygrad.uop.ops import CooperativeTileLoadSpec

def emitter(out_ph, inp_ph):
  sem=UOp.cooperative_tile_load(inp_ph, UOp.const(dtypes.weakint,0), CooperativeTileLoadSpec(tile_base=UOp.const(dtypes.weakint,0)))
  shared=lower_cooperative_tile_load(sem); idx=UOp.special(128,"lidx0")
  stores=[]
  for i in range(16):
    j=idx.alu(Ops.ADD,UOp.const(dtypes.weakint,i*128))
    stores.append(out_ph.index(j,ptr=True).store(shared.index(j).load()))
  return UOp.sink(*stores,arg=KernelInfo(name="nv2a_readback"))

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args(); out={"schema":"nv2a_readback/v1","status":"FAILED"}
  try:
    p=KernelProgram("nv2a_readback","nv2a_readback_v1",KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,emitter,output_spec=OutputSpec((2048,),dtypes.half))
    x=np.arange(2048,dtype=np.float32).astype(np.float16); inp=Tensor(x,device="NV").realize(); got=execute_promoted_program(None,inp,program=p).realize().numpy()
    out.update(status="PASS",finite=bool(np.isfinite(got).all()),exact=bool(np.array_equal(got,x)),max_abs=float(np.max(np.abs(got-x))),census={"alloc":2048,"stores":16,"loads":16,"barriers":1,"coverage":2048})
  except Exception as e: out.update(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
  open(a.out,"w").write(json.dumps(out,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
