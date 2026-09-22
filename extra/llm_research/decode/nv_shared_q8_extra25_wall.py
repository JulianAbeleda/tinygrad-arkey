#!/usr/bin/env python3
"""Current composed reverse wall bracket for incremental shared-Q8 block 25."""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL,LOCK,_gpu_state

def child(candidate,depth,count,max_context,reps,out):
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context);model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  admission=getattr(model.blk[25],"_shared_q8_attention_admission",None)
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try:row=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally:gen.close()
  result={"schema":"tinygrad.nv_shared_q8_extra25_wall.v1","candidate":candidate,"block25_admitted":admission is not None,
    "block25_flags":vars(admission) if admission is not None else None,"gpu_state":_gpu_state(),**row}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

def semantic_child(candidate,depth,count,max_context,out):
  from tinygrad import Tensor,UOp
  from tinygrad.helpers import Context
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  model=_load(MODEL,max_context);model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  pre=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try:prelude_token=int(next(pre))
  finally:pre.close()
  token=Tensor([[1]],dtype="int32").contiguous();temp=Tensor([0.0]);pos=UOp.variable("start_pos",0,max_context-1)
  logits=[];tokens=[]
  with Context(JIT=0):model.forward_with_logits(token,pos.bind(depth),temp)
  for i in range(count):
    sampled,full=model.decode_with_logits(token,pos.bind(depth+1+i),temp,feedback_slot=i&1);arr=full.numpy();sid=int(sampled.item())
    if sid!=int(arr.reshape(-1).argmax()):raise RuntimeError("sample/logit binding mismatch")
    logits.append(arr);tokens.append(sid);token=sampled
  a=np.stack(logits);row={"schema":"tinygrad.nv_shared_q8_extra25_semantic.v1","candidate":candidate,"prelude_token":prelude_token,"tokens":tokens,
    "tokens_sha256":hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest(),"shape":list(a.shape),"finite":bool(np.isfinite(a).all())}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");np.savez_compressed(out.with_suffix(".npz"),logits=a)

def semantic(a):
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison
  out=pathlib.Path(a.out);root=pathlib.Path(str(out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True);rows=[];arrays=[]
  for candidate,label in ((False,"control"),(True,"candidate")):
    dst=root/f"{label}.json";env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"}
    if candidate:env["TINYGRAD_SHARED_Q8_EXTRA_BLOCK25"]="1"
    else:env.pop("TINYGRAD_SHARED_Q8_EXTRA_BLOCK25",None)
    cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,__file__,"--mode","semantic-child","--depth",str(a.depth),"--count",str(a.count),"--max-context",str(a.max_context),"--out",str(dst)]+(["--candidate"] if candidate else [])
    r=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode:raise RuntimeError(f"{label} rc={r.returncode}: {r.stderr[-5000:]}")
    rows.append(json.loads(dst.read_text()));arrays.append(np.load(dst.with_suffix(".npz"))["logits"])
  result={"schema":"tinygrad.nv_shared_q8_extra25_semantic.v1",**_semantic_comparison(arrays[0],arrays[1],rows[0],rows[1]),"arms":rows}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))

def bracket(a):
  out=pathlib.Path(a.out);root=pathlib.Path(str(out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True)
  def arm(candidate,label):
    dst=root/f"{label}.json";env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"}
    if candidate:env["TINYGRAD_SHARED_Q8_EXTRA_BLOCK25"]="1"
    else:env.pop("TINYGRAD_SHARED_Q8_EXTRA_BLOCK25",None)
    cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,__file__,"--mode","child","--depth",str(a.depth),"--count",str(a.count),"--max-context",str(a.max_context),"--reps",str(a.reps),"--out",str(dst)]+(["--candidate"] if candidate else [])
    r=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode:raise RuntimeError(f"{label} rc={r.returncode}: {r.stderr[-5000:]}")
    return json.loads(dst.read_text())
  ca,b,cc=arm(False,"control_a"),arm(True,"candidate"),arm(False,"control_c");mid=(ca["median_ms_per_token"]+cc["median_ms_per_token"])/2
  result={"schema":"tinygrad.nv_shared_q8_extra25_wall.v1","control_a_ms":ca["median_ms_per_token"],"candidate_ms":b["median_ms_per_token"],"control_c_ms":cc["median_ms_per_token"],"control_midpoint_ms":mid,"recovery_us":(mid-b["median_ms_per_token"])*1000,"all_hashes_equal":len({x["token_stream_hash"] for x in (ca,b,cc)})==1,"candidate_admitted":b["block25_admitted"],"arms":[ca,b,cc]}
  result["verdict"]="WALL_PASS" if result["all_hashes_equal"] and result["candidate_admitted"] and b["median_ms_per_token"]<min(ca["median_ms_per_token"],cc["median_ms_per_token"]) else "NO_GO_WALL"
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
  p=argparse.ArgumentParser();p.add_argument("--mode",choices=("timing","child","semantic","semantic-child"),default="timing");p.add_argument("--candidate",action="store_true");p.add_argument("--depth",type=int,default=512);p.add_argument("--count",type=int,default=16);p.add_argument("--max-context",type=int,default=1024);p.add_argument("--reps",type=int,default=9);p.add_argument("--out",required=True);a=p.parse_args()
  semantic_child(a.candidate,a.depth,a.count,a.max_context,pathlib.Path(a.out)) if a.mode=="semantic-child" else child(a.candidate,a.depth,a.count,a.max_context,a.reps,pathlib.Path(a.out)) if a.mode=="child" else semantic(a) if a.mode=="semantic" else bracket(a)
