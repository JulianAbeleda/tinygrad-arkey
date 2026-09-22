#!/usr/bin/env python3
"""Real-weight Q5_K one-warp direct GEMV correctness and timing gate."""
from __future__ import annotations
import argparse,json,pathlib,statistics,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import Device,Tensor,dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.decode_kernels import LanePartition,_f16_word,_half4_lane,_q4k_group_params_from_words
from tinygrad.llm.gguf import ggml_data_to_tensor,gguf_load_metadata
from tinygrad.uop.ops import AxisType,KernelInfo,UOp
from extra.llm_research.decode.nv_q6_to_q5_feasibility import convert
from extra.llm_research.decode.nv_q6_to_q4_v_feasibility import _metrics

QK,WARP,Q5_WORDS=256,32,44

def _q5_block_dot(words,x,base,xblock,lane4):
  hdr=words.index(base).load(dtype=dtypes.uint32.vec(4));qh=words[base+4+lane4];out=UOp.const(dtypes.float32,0.)
  for g2 in range(4):
    qw=words[base+12+g2*8+lane4]
    for gp in range(2):
      grp=2*g2+gp;d,dmin,sc,mn=_q4k_group_params_from_words(hdr.gep(0),hdr.gep(1),hdr.gep(2),hdr.gep(3),grp)
      low=qw.rshift(gp*4).bitwise_and(0x0F0F0F0F);xv=x.index(xblock*QK+grp*32+lane4*4).load(dtype=dtypes.float16.vec(4))
      for nib in range(4):
        q=low.rshift(nib*8).bitwise_and(0xf).bitwise_or(qh.rshift(nib*8+grp).bitwise_and(1).lshift(4))
        wt=d*sc.cast(dtypes.float32)*q.cast(dtypes.float32)-dmin*mn.cast(dtypes.float32)
        out=out+wt*_half4_lane(xv,nib)
  return out

def emit(rows,k,resadd:bool=False):
  kb=k//QK;bpg=kb//4
  def kernel(out,words,x,*extra):
    row,lane=UOp.special(rows,"gidx0"),UOp.special(WARP,"lidx0");part=LanePartition(lane,words_per_group=8)
    loop=UOp.range(bpg,0,axis_type=AxisType.REDUCE);blk=part.block_group*bpg+loop;base=(row*kb+blk)*Q5_WORDS
    val=_q5_block_dot(words,x,base,blk,part.word_col);acc=UOp.placeholder((1,),dtypes.float32,20,AddrSpace.REG);acc=acc.after(acc[0].store(0.));acc=acc.after(acc[0].store(acc.after(loop)[0]+val).end(loop))
    total=_warp_reduce_sum_staged(acc[0],lane,WARP);result=total+extra[0][row].cast(dtypes.float32) if resadd else total
    return out[row].store(result).sink(arg=KernelInfo(name=f"q5k_direct{'_resadd' if resadd else ''}_{rows}_{k}",opts_to_apply=()))
  return kernel

def runner(p,o,w,x):
  rt=get_runtime("NV",p);gs,ls=p.arg.launch_dims({});buf=[t.uop.buffer.get_buf("NV") for t in (o,w,x)]
  return lambda wait=False:rt(*[buf[i] for i in p.arg.globals],global_size=gs,local_size=ls,vals=(),wait=wait)

def one(model,meta,name,rows,k,reps):
  info={x[0]:x for x in meta["tensor_infos"]}[name];_n,shape,typ,off=info
  n=rows*k;n6=(n//256)*210;raw=np.asarray(np.memmap(model,mode="r",dtype=np.uint8,offset=meta["data_start"]+off,shape=(n6,))).copy();dense,q5=convert(raw,n);dense=dense.reshape(rows,k)
  out=UOp.placeholder((rows,),dtypes.float32,0);words=UOp.placeholder((q5.size//4,),dtypes.uint32,1);xu=UOp.placeholder((k,),dtypes.float16,2);p=to_program(emit(rows,k)(out,words,xu),Device["NV"].renderer)
  rng=np.random.default_rng(20260825+rows);xnp=rng.normal(0,.2,k).astype(np.float16);w=Tensor(q5.view(np.uint32).copy(),dtype=dtypes.uint32,device="NV").realize();x=Tensor(xnp,device="NV").realize();o=Tensor.empty(rows,dtype=dtypes.float32,device="NV").realize();r=runner(p,o,w,x);r(True);got=o.numpy();ref=dense@xnp.astype(np.float32)
  q5d=ggml_data_to_tensor(Tensor(q5.copy(),dtype=dtypes.uint8),n,13).numpy().reshape(rows,k).astype(np.float32)
  q5ref=q5d@xnp.astype(np.float32);err=np.abs(got-q5ref);tol=max(2e-5,float(np.max(np.abs(q5ref)))*2e-6);times=[r(True)*1e6 for _ in range(reps)]
  return {"tensor":name,"shape":[rows,k],"q6_bytes":n6,"q5_bytes":q5.nbytes,"bytes_saved":n6-q5.nbytes,
    "quantization":_metrics(ref,q5ref),"correctness":{"max_abs_q5_ref":float(err.max()),"relative_l2_q5_ref":float(np.linalg.norm(err)/np.linalg.norm(q5ref)),"tolerance":tol,"pass":bool(err.max()<=tol)},"timing_us":{"median":statistics.median(times),"samples":times}}

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");ap.add_argument("--reps",type=int,default=31);ap.add_argument("--out",required=True);a=ap.parse_args();model=pathlib.Path(a.model);_,meta=gguf_load_metadata(model)
  rows=[one(model,meta,"blk.0.attn_v.weight",1024,4096,a.reps),one(model,meta,"blk.0.ffn_down.weight",4096,12288,a.reps)]
  ret={"schema":"tinygrad.nv_q5k_direct_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"rows":rows,"pass":all(r["correctness"]["pass"] for r in rows)};text=json.dumps(ret,indent=2,sort_keys=True);pathlib.Path(a.out).write_text(text+"\n");print(text);return 0 if ret["pass"] else 1
if __name__=="__main__":raise SystemExit(main())
