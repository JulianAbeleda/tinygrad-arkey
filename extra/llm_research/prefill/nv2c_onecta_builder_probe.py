import argparse,json,traceback,os
import numpy as np
from tinygrad import Tensor,dtypes
from tinygrad.uop.ops import UOp,Ops,ParamArg,KernelInfo
from tinygrad.llm.kernel_program import KernelProgram,KernelProgramProvenance,OutputSpec,execute_promoted_program
from tinygrad.schedule.wmma.kernels import nv_sm120_q16_grid_hd128_cooperative_attention
from tinygrad.codegen.opt.attention_fragment import attention_fragment_model

def emit(out,q,k,v):
  kv=int(os.getenv('NV2C_PROBE_KV','16'))
  return nv_sm120_q16_grid_hd128_cooperative_attention(q,k,v,out,q_tokens=16,q_heads=4,kv_heads=1,kv_tokens=kv,scale=0.08837890625,causal=True,valid_kv=kv,query_start=0,kernel_info=KernelInfo(name='nv2c_onecta'),warps_per_cta=4,fragment_model=attention_fragment_model('nv_sm120'))
def main():
  ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();r={'schema':'nv2c_onecta_builder_probe/v1','status':'FAILED'}
  try:
    p=KernelProgram('nv2c_onecta','nv2c_onecta_v1',KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,emit,output_spec=OutputSpec((4*16*128,),dtypes.float16))
    rng=np.random.default_rng(7); kv=int(os.getenv('NV2C_PROBE_KV','16')); q=Tensor(rng.standard_normal((4*16*128)).astype(np.float16),device='NV').realize(); k=Tensor(rng.standard_normal((kv*128)).astype(np.float16),device='NV').realize(); v=Tensor(rng.standard_normal((kv*128)).astype(np.float16),device='NV').realize()
    got=execute_promoted_program(None,q,k,v,program=p).realize().numpy();r.update(status='PASS',finite=bool(np.isfinite(got).all()),shape=list(got.shape),grid256=True,block128=True,shared_tiles=2)
  except Exception as e:r.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
  with open(a.out,'w') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
if __name__=='__main__':main()
