#!/usr/bin/env python3
"""Reverse token wall for one-time evict-last priming of real layer K/V caches."""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,statistics,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL,LOCK,PYTHON

def _prime(model,dev,active:int,max_context:int)->dict:
  from tinygrad.runtime.ops_nv import NVProgram
  from extra.llm_research.decode.nv_r_residual_cache_dispatch_probe import _make_queue
  from extra.llm_research.decode.nv_flash_l2_priority_turnability import _compile
  cache_bytes=2*1*8*max_context*128*2
  prg=NVProgram(dev,"kv_prime",_compile(dev,"kv_prime",cache_bytes//4,"evict_last"))
  sink=dev.allocator.alloc(512*4);q=_make_queue(dev);caches=[]
  for block in model.blk:
    cache=block.cache_kv.realize().uop.buffer._buf;caches.append(cache);q.exec(prg,prg.fill_kernargs((cache,sink)),(512,1,1),(256,1,1))
  q.signal(dev.timeline_signal,dev.next_timeline()).submit(dev);dev.synchronize()
  return {"active_tokens":active,"cache_count":len(caches),"active_bytes":len(caches)*2*8*active*128*2,"primed_bytes":len(caches)*cache_bytes}

def _measure(gen,dev,count,reps,prime_cb=None):
  prelude=int(next(gen));warm=[int(next(gen)) for _ in range(6)];dev.synchronize()
  prime=prime_cb() if prime_cb else None;dev.synchronize();samples,hashes,tokens=[],[],[]
  for _ in range(reps):
    st=time.perf_counter_ns();window=[int(next(gen)) for _ in range(count)];dev.synchronize();samples.append((time.perf_counter_ns()-st)/count/1e6)
    hashes.append(hashlib.sha256(",".join(map(str,window)).encode()).hexdigest());tokens+=window
  med=statistics.median(samples);mad=statistics.median(abs(x-med) for x in samples);limit=med+max(.25,6*mad);accepted=[x for x in samples if x<=limit]
  return {"prelude_token":prelude,"warmup_tokens":warm,"samples_ms_per_token":samples,"accepted_samples_ms_per_token":accepted,
    "median_ms_per_token":statistics.median(accepted),"token_stream_hash":hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest(),"prime":prime}

def child(arm,depth,count,max_context,reps,out):
  os.environ.update(DEV="NV",PROFILE="0",JIT_BATCH_SIZE="33")
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  model=_load(MODEL,max_context);model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True;dev=Device["NV"]
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0);active=depth+7
  try:settled=_measure(gen,dev,count,reps,(lambda:_prime(model,dev,active,max_context)) if arm.startswith("candidate") else None)
  finally:gen.close()
  result={"schema":"tinygrad.nv_flash_kv_retention_wall.v1","arm":arm,"settled":settled};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result

def launch(name,root,a):
  out=root/f"{name}.json";cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),"--arm",name,"--depth",str(a.depth),"--count",str(a.count),"--max-context",str(a.max_context),"--reps",str(a.reps),"--out",str(out)]
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,"PYTHONPATH":str(ROOT)})
  if run.returncode:raise RuntimeError(f"{name} failed: {run.stderr[-6000:]}")
  return json.loads(out.read_text())

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("control_a","candidate","control_c"));ap.add_argument("--depth",type=int,default=512);ap.add_argument("--count",type=int,default=16);ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--reps",type=int,default=7);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  if a.arm:r=child(a.arm,a.depth,a.count,a.max_context,a.reps,a.out)
  else:
    root=pathlib.Path(str(a.out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True);arms=[launch(x,root,a) for x in ("control_a","candidate","control_c")];walls=[x["settled"]["median_ms_per_token"]*1000 for x in arms];ctl=statistics.median((walls[0],walls[2]));cand=walls[1];hashes={x["settled"]["token_stream_hash"] for x in arms}
    r={"schema":"tinygrad.nv_flash_kv_retention_wall.v1","mode":"reverse-bracket","walls_us_per_token":{"control_a":walls[0],"candidate":cand,"control_c":walls[2],"control":ctl,"candidate_delta":cand-ctl},"tokens_per_second":{"control":1e6/ctl,"candidate":1e6/cand,"candidate_delta":1e6/cand-1e6/ctl},"all_token_hashes_equal":len(hashes)==1,"verdict":"WALL_PASS" if len(hashes)==1 and cand<min(walls[0],walls[2]) else "NO_GO_WALL","arms":arms};a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
  print(json.dumps(r,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
