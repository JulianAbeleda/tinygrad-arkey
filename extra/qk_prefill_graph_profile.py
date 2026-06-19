#!/usr/bin/env python3
"""BREAK THE MEASUREMENT WALL: warm per-kernel prefill split + GPU-busy-vs-overhead, via ProfileGraphEvent.

The HCQ graph replay records per-jit-item timestamps (graph/hcq.py collect_timestamps -> ProfileGraphEvent).
Each entry: dur = sigs[en_id]-sigs[st_id] (us, WARM). Sum = GPU-busy; span-busy = inter-kernel overhead/gaps.
Resolves: (a) matmuls tiny-warm-fraction vs (b) overhead-dominated, AND the 1.24x concrete attribution.

Run: DEV=AMD PREFILL_V2=1 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_graph_profile.py
"""
from __future__ import annotations
import os, collections, re
from tinygrad import Tensor, Device, UOp
from tinygrad.device import Compiled
from tinygrad.llm.model import Transformer, PREFILL_UBATCH

def cat(nm):
  nm=str(nm)
  if re.search(r'r_(16_192|8_64|8_16|16_64)', nm): return "FFN/proj matmul"
  if 'start_pos' in nm or re.search(r'(_512_|_128_4|2_8_16_4_4_128|512_16_256)', nm): return "attention/KV"
  if nm.startswith('E_'): return "elementwise glue"
  if '1187' in nm: return "lm_head"
  return "other"

def profile(model, chunk, temp, sp, label):
  for b in model.blk: b._use_flash, b._prefill_v2 = False, True
  from tinygrad import TinyJit
  model.prefill_v2_jit = TinyJit(model.forward)
  for _ in range(10): model(chunk, sp, temp).realize(); Device["AMD"].synchronize()
  import time, statistics
  walls=[]
  for _ in range(10):
    t0=time.perf_counter(); model(chunk, sp, temp).realize(); Device["AMD"].synchronize(); walls.append(time.perf_counter()-t0)
  wall_ms=statistics.median(walls)*1000
  base=len(Compiled.profile_events)
  model(chunk, sp, temp).realize(); Device["AMD"].synchronize()
  Device["AMD"]._at_profile_finalize()
  print(f"  [wall median {wall_ms:.0f}ms]")
  evs=[e for e in Compiled.profile_events[base:] if type(e).__name__=="ProfileGraphEvent"]
  if not evs: print(f"{label}: NO ProfileGraphEvent (wall not broken)"); return
  e=evs[-1]; sigs=[float(s) for s in e.sigs]
  durs=[]; per=collections.defaultdict(float); cnt=collections.defaultdict(int)
  for ent in e.ents:
    d=sigs[ent.en_id]-sigs[ent.st_id]; durs.append((sigs[ent.st_id],sigs[ent.en_id],d,ent.name))
    c=cat(ent.name); per[c]+=d; cnt[c]+=1
  busy=sum(d for *_,d,_ in durs); span=max(s[1] for s in durs)-min(s[0] for s in durs)
  print(f"\n=== {label}: {len(e.ents)} kernels | span {span:.0f}us | GPU-busy {busy:.0f}us | OVERHEAD/gaps {span-busy:.0f}us ({100*(span-busy)/span:.0f}%) ===")
  for c,d in sorted(per.items(),key=lambda x:-x[1]):
    print(f"  {c:<20}{d:8.0f}us ({100*d/busy:4.1f}% of busy, {100*d/span:4.1f}% of span, {cnt[c]} kernels)")

def main():
  assert os.environ.get("PREFILL_V2") and os.environ.get("PROFILE")
  Tensor.manual_seed(0)
  model,_=Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",2048)
  N=PREFILL_UBATCH
  t=Tensor([5,6,7,8,9,10]*200+[0]*(2048-1200),dtype="int32").reshape(1,2048); chunk=t[:,0:N].contiguous(); temp=Tensor([0.0])
  vsp=UOp.variable("start_pos",0,2047)
  profile(model, chunk, temp, 0, "CONCRETE start_pos=0")
  profile(model, chunk, temp, vsp.bind(0), "SYMBOLIC start_pos")

if __name__=="__main__": main()
