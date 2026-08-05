#!/usr/bin/env python3
"""Default-off one-block full-logit lease for scale-only FFN RMSNorm fusion."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, time
import numpy as np
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

CONTRACT={"top_k":10,"relative_l2_max":1e-3,"historical_max_abs_atol":0.01}

def _digest(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def _lease(model,index:int):
  from tinygrad import Tensor,dtypes
  if not (0 <= index < len(model.blk)): raise ValueError("invalid block index")
  for b in model.blk:
    if hasattr(b,"_rms_affine_gateup_norm_weight"): delattr(b,"_rms_affine_gateup_norm_weight")
  b=model.blk[index]
  # An owned fp16 buffer makes the consumer's input physical and visible;
  # normal model loading never creates this attribute.
  b._rms_affine_gateup_norm_weight=Tensor.empty(4096,dtype=dtypes.float16,device=b.ffn_norm.weight.device).assign(
    b.ffn_norm.weight.cast(dtypes.float16).contiguous()).realize()

def run(model_path,depth,count,max_context,index,lease):
  from tinygrad import Tensor,UOp
  from tinygrad.helpers import Context
  m=_load(model_path,max_context)
  if lease: _lease(m,index)
  gen=m.generate(_prompt(model_path,depth),chunk_size=32,temperature=0.0)
  try: prelude=int(next(gen))
  finally: gen.close()
  token,temp=Tensor([[1]],dtype="int32").contiguous(),Tensor([0.0])
  pos=UOp.variable("start_pos",0,max_context-1)
  with Context(JIT=0): _, eager=m.forward_with_logits(token,pos.bind(depth),temp)
  if not np.isfinite(eager.numpy()).all(): raise RuntimeError("eager logits nonfinite")
  rows=[]; sampled=[]
  for i in range(count):
    s,full=m.decode_with_logits(token,pos.bind(depth+1+i),temp); arr=full.numpy()
    if int(s.item()) != int(arr.argmax(axis=-1).item()): raise RuntimeError("diagnostic sample/logit mismatch")
    if not np.isfinite(arr).all(): raise RuntimeError("nonfinite logits")
    rows.append(arr); sampled.append(int(s.item()))
  arr=np.stack(rows)
  return {"prelude":prelude,"tokens":sampled,"sha256":_digest(arr),"shape":list(arr.shape),"logits":arr}

def timing(model_path,depth,count,max_context,index,lease,reps):
  from tinygrad import Device
  m=_load(model_path,max_context)
  if lease: _lease(m,index)
  # This matches the settled composed decode contract used by the live parity
  # timing harness; the lease itself remains the only differing input.
  m._decode_direct_greedy_promoted=True; m._decode_feedback_pingpong_promoted=True
  gen=m.generate(_prompt(model_path,depth),chunk_size=32,temperature=0.0); dev=Device[Device.DEFAULT]
  try:
    prelude=int(next(gen)); warm=[int(next(gen)) for _ in range(6)]; dev.synchronize()
    samples=[]; streams=[]; hashes=[]
    for _ in range(reps):
      st=time.perf_counter_ns(); window=[int(next(gen)) for _ in range(count)]; dev.synchronize()
      samples.append((time.perf_counter_ns()-st)/count/1e6); streams.extend(window); hashes.append(hashlib.sha256(",".join(map(str,window)).encode()).hexdigest())
  finally: gen.close()
  return {"prelude":prelude,"warmup_tokens":warm,"samples_ms_per_token":samples,"median_ms_per_token":statistics.median(samples),
    "token_hashes":hashes,"token_stream_hash":hashlib.sha256(",".join(map(str,streams)).encode()).hexdigest(),"timed_token_count":len(streams)}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--out",required=True); ap.add_argument("--mode",choices=("logits","timing"),default="logits")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=4); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--block",type=int,default=0); ap.add_argument("--lease",action="store_true"); ap.add_argument("--reps",type=int,default=5); a=ap.parse_args()
  if a.max_context <= a.depth+a.count: raise ValueError("context too short")
  r=run(a.model,a.depth,a.count,a.max_context,a.block,a.lease) if a.mode == "logits" else timing(a.model,a.depth,a.count,a.max_context,a.block,a.lease,a.reps)
  logits=r.pop("logits",None)
  r.update({"schema":"tinygrad.nv.rmsnorm_scale_gateup_one_layer.v1","lease":a.lease,"block":a.block,"depth":a.depth,"count":a.count})
  p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
  if logits is not None: np.savez_compressed(p.with_suffix(".npz"),logits=logits)
  p.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n"); print(json.dumps(r,sort_keys=True))
if __name__ == "__main__": main()
