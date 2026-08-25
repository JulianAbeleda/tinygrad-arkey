#!/usr/bin/env python3
"""Progressive full-logit quality gate for Q6_K FFN-down -> Q5_K sidecars."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import Tensor,UOp,dtypes
from tinygrad.llm.gguf import gguf_load_metadata
from tinygrad.llm.kernel_program import KernelProgram,KernelProgramProvenance,OutputSpec,execute_research_program
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
from extra.llm_research.decode.nv_q5k_direct_microgate import emit
from extra.llm_research.decode.nv_q6_to_q5_feasibility import convert

ROWS,K=4096,12288
class Q5Down:
  def __init__(self,control,words,block,owner):self.control,self.words,self.block,self.owner=control,words,block,owner
  def __call__(self,x:Tensor,**epi):
    if getattr(self.owner,"_is_prefill",False):return self.control(x,**epi)
    xv=x[:,0,:].reshape(K).cast(dtypes.float16).contiguous();res=epi.get("normed_h");resadd=res is not None
    prg=KernelProgram("research.q5_down",f"blk{self.block}.q5.r{int(resadd)}",KernelProgramProvenance.RESEARCH_ONLY,emit(ROWS,K,resadd=resadd),output_spec=OutputSpec((ROWS,),dtypes.float32))
    args=[Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device),self.words.to(x.device),xv]
    if resadd:args.append(res[:,0,:].reshape(ROWS).cast(dtypes.float32))
    return execute_research_program(*args,program=prg).reshape(1,1,ROWS)

def install(model,path,blocks):
  _,meta=gguf_load_metadata(path);infos={x[0]:x for x in meta["tensor_infos"]};installed=[]
  for bi in blocks:
    name=f"blk.{bi}.ffn_down.weight";_n,shape,typ,off=infos[name]
    if tuple(shape)!=(K,ROWS) or typ!=14:raise RuntimeError(infos[name])
    n=ROWS*K;n6=(n//256)*210;raw=np.asarray(np.memmap(path,mode="r",dtype=np.uint8,offset=meta["data_start"]+off,shape=(n6,))).copy();_dense,q5=convert(raw,n)
    words=Tensor(q5.view(np.uint32).copy(),dtype=dtypes.uint32,device="NV").realize();owner=model.blk[bi];owner.ffn_down=Q5Down(owner.ffn_down,words,bi,owner);installed.append({"block":bi,"q6_bytes":n6,"q5_bytes":q5.nbytes})
  return installed

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");ap.add_argument("--blocks",default="");ap.add_argument("--depth",type=int,default=128);ap.add_argument("--count",type=int,default=4);ap.add_argument("--max-context",type=int,default=512);ap.add_argument("--out",required=True);a=ap.parse_args()
  blocks=[int(x) for x in a.blocks.split(",") if x];model=_load(a.model,a.max_context);installed=install(model,pathlib.Path(a.model),blocks) if blocks else []
  gen=model.generate(_prompt(a.model,a.depth),chunk_size=32,temperature=0.0)
  try:next(gen)
  finally:gen.close()
  token,temp=Tensor([[1]],dtype="int32").contiguous(),Tensor([0.0]);sp=UOp.variable("start_pos",0,a.max_context-1);logits=[];tokens=[]
  for i in range(a.count):
    sample,full=model.decode_with_logits(token,sp.bind(a.depth+1+i),temp);arr=full.numpy();logits.append(arr);tokens.append(int(arr.argmax(axis=-1).item()))
  stack=np.stack(logits);out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix(".npz"),logits=stack)
  ret={"schema":"tinygrad.nv_q5k_full_logits.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"blocks":blocks,"installed":installed,"depth":a.depth,"count":a.count,"tokens":tokens,"finite":bool(np.isfinite(stack).all()),"logits_sha256":hashlib.sha256(stack.tobytes()).hexdigest()};out.write_text(json.dumps(ret,indent=2,sort_keys=True)+"\n");print(json.dumps(ret));return 0
if __name__=="__main__":raise SystemExit(main())
