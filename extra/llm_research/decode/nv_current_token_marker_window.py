#!/usr/bin/env python3
"""Marker-light device-window accounting on the current continuous decode path."""
from __future__ import annotations
import argparse, json, os, pathlib, statistics, subprocess, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--model',default=MODEL); ap.add_argument('--depth',type=int,default=512)
  ap.add_argument('--count',type=int,default=24); ap.add_argument('--warmup',type=int,default=6)
  ap.add_argument('--max-context',type=int,default=768); ap.add_argument('--expected-groups',type=int,default=4)
  ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
  os.environ.update(DEV='NV',PROFILE='0')
  from tinygrad import Device
  from tinygrad.helpers import unwrap
  from tinygrad.runtime.graph.hcq import HCQGraph
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt

  dev=Device['NV']; state={'active':False,'idx':0,'start':None,'end':None,'first_enter_ns':None,'last_exit_ns':None}
  original=HCQGraph.__call__
  def marker(sig):
    unwrap(dev.hw_compute_queue_t)().wait(dev.timeline_signal,dev.timeline_value-1).timestamp(sig).signal(dev.timeline_signal,dev.next_timeline()).submit(dev)
  def wrapped(self,input_uops,var_vals,wait=False):
    if not state['active']: return original(self,input_uops,var_vals,wait=wait)
    idx=state['idx']
    if idx == 0: state['first_enter_ns']=time.perf_counter_ns()
    if idx == 0: state['start']=dev.new_signal(value=0); marker(state['start'])
    ret=original(self,input_uops,var_vals,wait=wait)
    if idx == a.expected_groups-1: state['last_exit_ns']=time.perf_counter_ns()
    if idx == a.expected_groups-1: state['end']=dev.new_signal(value=0); marker(state['end'])
    state['idx']=(idx+1)%a.expected_groups
    return ret
  HCQGraph.__call__=wrapped

  model=_load(a.model,a.max_context); gen=model.generate(_prompt(a.model,a.depth),chunk_size=32,temperature=0.0)
  tokens=[]; walls=[]; windows=[]; pre_first=[]; post_last=[]
  try:
    for _ in range(a.warmup): next(gen)
    dev.synchronize(); state['active']=True
    for _ in range(a.count):
      state.update(idx=0,start=None,end=None,first_enter_ns=None,last_exit_ns=None)
      t0=time.perf_counter_ns(); tok=int(next(gen)); dev.synchronize(); t1=time.perf_counter_ns()
      if state['idx'] != 0 or state['start'] is None or state['end'] is None: raise RuntimeError(f"group mismatch idx={state['idx']}")
      tokens.append(tok); walls.append((t1-t0)/1e3); windows.append(float(state['end'].timestamp-state['start'].timestamp))
      pre_first.append((state['first_enter_ns']-t0)/1e3); post_last.append((t1-state['last_exit_ns'])/1e3)
  finally: state['active']=False; gen.close(); HCQGraph.__call__=original
  row={'schema':'tinygrad.nv_current_token_marker_window.v1','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'depth':a.depth,'count':a.count,'warmup':a.warmup,'max_context':a.max_context,'expected_groups':a.expected_groups,
       'wall_us':walls,'device_window_us':windows,'outside_window_us':[w-d for w,d in zip(walls,windows)],
       'pre_first_graph_us':pre_first,'post_last_graph_us':post_last,'tokens':tokens,
       'median':{'wall_us':statistics.median(walls),'device_window_us':statistics.median(windows),'outside_window_us':statistics.median([w-d for w,d in zip(walls,windows)]),
                 'pre_first_graph_us':statistics.median(pre_first),'post_last_graph_us':statistics.median(post_last)}}
  a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n'); print(json.dumps(row['median'],indent=2))
if __name__=='__main__': main()
