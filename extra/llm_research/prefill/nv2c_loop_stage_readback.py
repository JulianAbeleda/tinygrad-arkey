import argparse, json, traceback
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program

def emitter(out, inp):
  tid=UOp.special(128,"lidx0"); rng=UOp.range(2,7900)
  local=UOp(Ops.DEFINE_LOCAL,dtypes.half.ptr(2048,AddrSpace.LOCAL),arg=("nv2c_reused_tile", "single_buffer_barrier_v1"))
  base=rng*UOp.const(dtypes.weakint,2048)
  stores=[]
  for i in range(16):
    j=tid+UOp.const(dtypes.weakint,i*128)
    stores.append(local.index(j,ptr=True).store(inp.index(base+j).load()))
  pub=UOp(Ops.BARRIER,dtypes.void,(UOp.group(*stores),),arg=("nv2c_publish", "single_buffer_barrier_v1"))
  copies=[]
  for i in range(16):
    j=tid+UOp.const(dtypes.weakint,i*128)
    copies.append(out.index(base+j,ptr=True).store(local.after(pub).index(j).load()))
  end=UOp(Ops.BARRIER,dtypes.void,(UOp.group(*copies),),arg=("nv2c_end", "single_buffer_barrier_v1"))
  return UOp.sink(end.end(rng),arg=KernelInfo(name="nv2c_loop_stage_readback"))

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args(); r={"schema":"nv2c_loop_stage_readback/v1","status":"FAILED"}
  try:
    p=KernelProgram("nv2c_loop_stage_readback","nv2c_loop_stage_readback_v1",KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,emitter,output_spec=OutputSpec((4096,),dtypes.half))
    x=np.arange(4096,dtype=np.float32).astype(np.float16); got=execute_promoted_program(None,Tensor(x,device="NV").realize(),program=p).realize().numpy()
    r.update(status="PASS" if np.array_equal(got,x) else "FAIL",exact=bool(np.array_equal(got,x)),finite=bool(np.isfinite(got).all()),max_abs=float(np.max(np.abs(got-x))),census={"range_extent":2,"shared_define_local":1,"shared_elements":2048,"publish_barriers":2,"end_barriers":2,"dynamic_tile_base":"range*2048"})
  except Exception as e:r.update(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
  with open(a.out,"w") as f:json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
if __name__=="__main__":main()
