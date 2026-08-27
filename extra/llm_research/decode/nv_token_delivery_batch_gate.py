#!/usr/bin/env python3
"""Bounded greedy run-ahead discriminator with exact retained token delivery."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'


def _windows(gen, count:int, reps:int):
  samples, hashes = [], []
  for _ in range(reps):
    vals=[]; begin=time.perf_counter_ns()
    for _ in range(count): vals.append(int(next(gen)))
    samples.append((time.perf_counter_ns()-begin)/1e6/count)
    hashes.append(hashlib.sha256(bytes().join(int(x).to_bytes(4,'little',signed=True) for x in vals)).hexdigest())
  return {'samples_ms_per_token':samples,'median_ms_per_token':statistics.median(samples),'token_hashes':hashes}


def _batched(gen, batch:int):
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.device import BufferSpec
  dev=Device['NV']; ring=dev.allocator._alloc(batch*4,BufferSpec()); original=Tensor.item; slot=0
  def intercept(t):
    nonlocal slot
    if t.dtype == dtypes.int32 and t.numel() == 1:
      src=t.uop.buffer.get_buf(t.device)
      dev.hw_copy_queue_t().wait(dev.timeline_signal,dev.timeline_value-1).copy(ring.offset(slot*4,4),src,4) \
        .signal(dev.timeline_signal,dev.next_timeline()).submit(dev)
      slot += 1
      return 0
    return original(t)
  try:
    while True:
      slot=0; Tensor.item=intercept
      for _ in range(batch): next(gen)
      Tensor.item=original
      raw=bytearray(batch*4);dev.allocator._copyout(memoryview(raw),ring)
      for i in range(batch): yield int.from_bytes(raw[i*4:(i+1)*4],'little',signed=True)
  finally: Tensor.item=original


def run(batch:int,count:int,reps:int,depth:int,max_context:int):
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  model=_load(MODEL,max_context)
  def make(): return model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  control=make(); control_warm=[next(control) for _ in range(6)]; Device['NV'].synchronize()
  a=_windows(control,count,reps); control.close()
  model.reset_generation_state()
  candidate_raw=make(); candidate_warm=[next(candidate_raw) for _ in range(6)]; Device['NV'].synchronize()
  candidate=_batched(candidate_raw,batch); b=_windows(candidate,count,reps); candidate.close(); candidate_raw.close()
  return {'schema':'tinygrad.nv_token_delivery_batch_gate.v1','batch':batch,'count':count,'reps':reps,'depth':depth,
    'control_warm':control_warm,'candidate_warm':candidate_warm,'control':a,'candidate':b,
    'walls_us_per_token':{'control':a['median_ms_per_token']*1000,'candidate':b['median_ms_per_token']*1000,
      'delta':(b['median_ms_per_token']-a['median_ms_per_token'])*1000},
    'tokens_per_second':{'control':1000/a['median_ms_per_token'],'candidate':1000/b['median_ms_per_token']},
    'all_windows_exact':a['token_hashes']==b['token_hashes']}


def main():
  ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,default=4);ap.add_argument('--count',type=int,default=24)
  ap.add_argument('--reps',type=int,default=7);ap.add_argument('--depth',type=int,default=512);ap.add_argument('--max-context',type=int,default=768)
  ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args();r=run(a.batch,a.count,a.reps,a.depth,a.max_context)
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2,sort_keys=True))
  return 0 if r['all_windows_exact'] else 1
if __name__=='__main__': raise SystemExit(main())
