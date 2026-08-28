#!/usr/bin/env python3
"""Progressive recurrent-logit screen for post-hoc Q4_K -> U4Z8 gate/up.

This is a fail-closed research sidecar. It never edits the GGUF or production
route. A pass admits acquisition/calibration from a higher-precision source;
it is not sufficient for production promotion by itself.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from tinygrad import Tensor,UOp,dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor,gguf_load_metadata
from tinygrad.llm.kernel_program import KernelProgram,KernelProgramProvenance,OutputSpec,execute_research_program
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
import nv_u4z8_g64_p256_o_microgate as primitive

ROWS,K=12288,4096


def convert(raw:np.ndarray)->np.ndarray:
  n=ROWS*K;dense=ggml_data_to_tensor(Tensor(raw.copy(),dtype=dtypes.uint8),n,12).numpy().reshape(ROWS,K).astype(np.float32)
  return quantize_u4z8(dense)


def quantize_u4z8(dense:np.ndarray,mse:bool=False)->np.ndarray:
  dense=dense.reshape(ROWS,16,4,64).astype(np.float32)
  peak=np.max(np.abs(dense),axis=-1);scale=np.where(peak==0,1.0,peak/7.0)
  if mse:
    best_err=np.full(scale.shape,np.inf,np.float32);best_scale=scale.copy()
    for alpha in np.linspace(0.50,1.0,11,dtype=np.float32):
      trial=np.where(peak==0,1.0,peak*alpha/7.0);q=np.clip(np.rint(dense/trial[...,None]),-8,7)
      err=np.sum((dense-q*trial[...,None])**2,axis=-1);take=err<best_err;best_err=np.where(take,err,best_err);best_scale=np.where(take,trial,best_scale)
    scale=best_scale
  scale=scale.astype(np.float16)
  q=np.clip(np.rint(dense/scale[...,None]),-8,7).astype(np.int16)+8
  q=q.reshape(ROWS,16,4,2,8,4).astype(np.uint32)
  shifts=(np.arange(4,dtype=np.uint32)*8)[None,None,None,None,None,:]+(np.arange(2,dtype=np.uint32)*4)[None,None,None,:,None,None]
  packed=np.bitwise_or.reduce(np.bitwise_or.reduce(q<<shifts,axis=5),axis=3).reshape(ROWS,16,32)
  sb=scale.view(np.uint16).astype(np.uint32);hdr=np.stack((sb[:,:,0]|(sb[:,:,1]<<16),sb[:,:,2]|(sb[:,:,3]<<16)),axis=-1)
  return np.concatenate((hdr,packed),axis=-1).reshape(-1).astype(np.uint32)


def quantize_affine_dense(src:np.ndarray,hessian:np.ndarray|None=None)->tuple[np.ndarray,float]:
  src=src.reshape(ROWS,16,4,64).astype(np.float32)
  lo,hi=src.min(axis=-1),src.max(axis=-1);base=np.where(hi>lo,(hi-lo)/15.0,1.0)
  if hessian is None:scale=base
  else:
    h=hessian.reshape(1,16,4,64).astype(np.float32);best=np.full(base.shape,np.inf,np.float32);scale=base.copy()
    for alpha in np.linspace(0.60,1.20,13,dtype=np.float32):
      trial=base*alpha;zp=np.clip(np.rint(-lo/trial),0,15);q=np.clip(np.rint(src/trial[...,None])+zp[...,None],0,15)
      err=np.sum(h*(src-trial[...,None]*(q-zp[...,None]))**2,axis=-1);take=err<best;best=np.where(take,err,best);scale=np.where(take,trial,scale)
  scale=scale.astype(np.float16).astype(np.float32);zp=np.clip(np.rint(-lo/scale),0,15).astype(np.int8);q=np.clip(np.rint(src/scale[...,None])+zp[...,None],0,15)
  deq=(scale[...,None]*(q-zp[...,None])).reshape(ROWS,K).astype(np.float16)
  rel=float(np.linalg.norm(deq.astype(np.float32)-src.reshape(ROWS,K))/np.linalg.norm(src))
  return deq,rel


def convert_affine_dense(raw:np.ndarray)->tuple[np.ndarray,float]:
  n=ROWS*K;src=ggml_data_to_tensor(Tensor(raw.copy(),dtype=dtypes.uint8),n,12).numpy().reshape(ROWS,K).astype(np.float32)
  return quantize_affine_dense(src)


def convert_q4meta_dense(raw:np.ndarray,bits:int)->tuple[np.ndarray,float]:
  blocks=raw.reshape(-1,144);out=blocks.copy();s=blocks[:,4:16]
  sc=np.empty((len(blocks),8),dtype=np.uint8);mn=np.empty_like(sc)
  sc[:,:4]=s[:,:4]&63;mn[:,:4]=s[:,4:8]&63
  sc[:,4:]=(s[:,8:12]&15)|((s[:,:4]>>6)<<4);mn[:,4:]=(s[:,8:12]>>4)|((s[:,4:8]>>6)<<4)
  levels=(1<<bits)-1;sc=np.rint(sc.astype(np.float32)*(levels/63)).astype(np.uint8);mn=np.rint(mn.astype(np.float32)*(levels/63)).astype(np.uint8)
  sr=np.rint(sc.astype(np.float32)*(63/levels)).astype(np.uint8);mr=np.rint(mn.astype(np.float32)*(63/levels)).astype(np.uint8)
  ns=np.zeros_like(s);ns[:,:4]=sr[:,:4]|((sr[:,4:]>>4)<<6);ns[:,4:8]=mr[:,:4]|((mr[:,4:]>>4)<<6);ns[:,8:12]=(sr[:,4:]&15)|((mr[:,4:]&15)<<4)
  out[:,4:16]=ns
  n=ROWS*K;src=ggml_data_to_tensor(Tensor(raw.copy(),dtype=dtypes.uint8),n,12).numpy().reshape(ROWS,K).astype(np.float32)
  dense=ggml_data_to_tensor(Tensor(out.reshape(-1),dtype=dtypes.uint8),n,12).numpy().reshape(ROWS,K).astype(np.float32)
  rel=float(np.linalg.norm(dense-src)/np.linalg.norm(src));return dense.astype(np.float16),rel


def emitter():
  old=(primitive.ROWS,primitive.K,primitive.K_BLOCKS,primitive.CANDIDATE_WORDS,primitive.CANDIDATE)
  try:
    primitive.ROWS,primitive.K,primitive.K_BLOCKS=ROWS,K,16;primitive.CANDIDATE_WORDS=ROWS*16*34
    primitive.CANDIDATE=f"u4z8_g64_p256_quality_{ROWS}_{K}"
    return primitive.emit_s4_o()
  finally:
    primitive.ROWS,primitive.K,primitive.K_BLOCKS,primitive.CANDIDATE_WORDS,primitive.CANDIDATE=old


EMIT=emitter()
class U4Linear:
  def __init__(self,control,words,block,role,owner):self.control,self.words,self.block,self.role,self.owner=control,words,block,role,owner
  def __call__(self,x:Tensor):
    if getattr(self.owner,"_is_prefill",False):return self.control(x)
    xv=x[:,0,:].reshape(K).cast(dtypes.float16).contiguous();zero=Tensor.zeros(ROWS,dtype=dtypes.float32,device=x.device)
    prg=KernelProgram("research.u4z8_gate_up",f"b{self.block}.{self.role}",KernelProgramProvenance.RESEARCH_ONLY,EMIT,
      output_spec=OutputSpec((ROWS,),dtypes.float32))
    out=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device),self.words.to(x.device),xv,zero,program=prg)
    return out.reshape(1,1,ROWS)

class DenseQualityLinear:
  def __init__(self,control,weight,owner):self.control,self.weight,self.owner=control,weight,owner
  def __call__(self,x:Tensor):
    if getattr(self.owner,"_is_prefill",False):return self.control(x)
    return x.cast(dtypes.float16).matmul(self.weight.transpose()).cast(dtypes.float32)


def install(model,path:pathlib.Path,blocks:list[int],fmt:str,bf16_shard:pathlib.Path|None,calibration:pathlib.Path|None)->list[dict]:
  _,meta=gguf_load_metadata(path);infos={x[0]:x for x in meta["tensor_infos"]};ret=[]
  bf=None
  if bf16_shard is not None:
    from safetensors import safe_open
    bf=safe_open(bf16_shard,framework="pt")
  for bi in blocks:
    owner=model.blk[bi];owner._decode_q4k_w1w3_fusion_promoted=False
    row={"block":bi}
    for role in ("gate","up"):
      name=f"blk.{bi}.ffn_{role}.weight";_n,shape,typ,off=infos[name]
      if tuple(shape)!=(K,ROWS) or typ!=12:raise RuntimeError(infos[name])
      size=ROWS*(K//256)*144;raw=np.asarray(np.memmap(path,mode="r",dtype=np.uint8,offset=meta["data_start"]+off,shape=(size,))).copy()
      control=getattr(owner,f"ffn_{role}")
      if fmt=="bf16dense":
        if bf is None:raise ValueError("bf16dense requires --bf16-shard")
        key=f"model.layers.{bi}.mlp.{role}_proj.weight";dense=bf.get_tensor(key).float().numpy().astype(np.float16)
        weight=Tensor(dense,device="NV").realize();setattr(owner,f"ffn_{role}",DenseQualityLinear(control,weight,owner));rel=0.0
      elif fmt in ("u4z8","u4z8mse"):
        if bf is None: packed=convert(raw);rel=None
        else:
          key=f"model.layers.{bi}.mlp.{role}_proj.weight";dense=bf.get_tensor(key).float().numpy();packed=quantize_u4z8(dense,mse=fmt=="u4z8mse")
          rel=None
        words=Tensor(packed,dtype=dtypes.uint32,device="NV").realize();setattr(owner,f"ffn_{role}",U4Linear(control,words,bi,role,owner))
      else:
        if fmt in ("u4affine","u4affineh") and bf is not None:
          key=f"model.layers.{bi}.mlp.{role}_proj.weight";h=np.load(calibration) if fmt=="u4affineh" and calibration else None;dense,rel=quantize_affine_dense(bf.get_tensor(key).float().numpy(),h)
        else:dense,rel=(convert_affine_dense(raw) if fmt=="u4affine" else convert_q4meta_dense(raw,4 if fmt=="q4meta4" else 5))
        weight=Tensor(dense,device="NV").realize();setattr(owner,f"ffn_{role}",DenseQualityLinear(control,weight,owner))
      bpb=256 if fmt=="bf16dense" else 136 if fmt in ("u4z8","u4z8mse") else 142 if fmt=="q4meta5" else 140
      row[f"{role}_source_bytes"]=size;row[f"{role}_candidate_bytes"]=ROWS*16*bpb;row[f"{role}_weight_rel_l2"]=rel
    ret.append(row)
  return ret


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");ap.add_argument("--blocks",default="");ap.add_argument("--format",choices=("bf16dense","u4z8","u4z8mse","u4affine","u4affineh","q4meta4","q4meta5"),default="u4z8");ap.add_argument("--bf16-shard",type=pathlib.Path);ap.add_argument("--calibration",type=pathlib.Path);ap.add_argument("--depth",type=int,default=128);ap.add_argument("--count",type=int,default=3);ap.add_argument("--max-context",type=int,default=512);ap.add_argument("--out",required=True);a=ap.parse_args()
  blocks=[int(x) for x in a.blocks.split(",") if x];model=_load(a.model,a.max_context);installed=install(model,pathlib.Path(a.model),blocks,a.format,a.bf16_shard,a.calibration) if blocks else []
  gen=model.generate(_prompt(a.model,a.depth),chunk_size=32,temperature=0.0)
  try:next(gen)
  finally:gen.close()
  token,temp=Tensor([[1]],dtype="int32").contiguous(),Tensor([0.0]);sp=UOp.variable("start_pos",0,a.max_context-1);logits=[];tokens=[]
  for i in range(a.count):
    _sample,full=model.decode_with_logits(token,sp.bind(a.depth+1+i),temp);arr=full.numpy();logits.append(arr);tokens.append(int(arr.argmax(axis=-1).item()))
  stack=np.stack(logits);out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix(".npz"),logits=stack)
  ret={"schema":"tinygrad.nv_u4z8_gate_up_quality.v3","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"format":a.format,"source":"bf16" if a.bf16_shard else "installed_q4k","bf16_shard":str(a.bf16_shard) if a.bf16_shard else None,"blocks":blocks,"installed":installed,"depth":a.depth,"count":a.count,"tokens":tokens,"finite":bool(np.isfinite(stack).all()),"logits_sha256":hashlib.sha256(stack.tobytes()).hexdigest()};out.write_text(json.dumps(ret,indent=2,sort_keys=True)+"\n");print(json.dumps(ret));return 0
if __name__=="__main__":raise SystemExit(main())
