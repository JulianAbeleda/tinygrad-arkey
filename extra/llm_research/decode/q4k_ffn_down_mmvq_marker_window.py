#!/usr/bin/env python3
"""Marker-light device/outside-window decomposition for one leased FFN layer."""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, statistics, time


def _median_rows(rows,key): return statistics.median(float(x[key]) for x in rows)


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--indices",default="");ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--reps",type=int,default=3);ap.add_argument("--out",required=True)
  a=ap.parse_args();indices=tuple(int(x) for x in a.indices.split(",") if x)
  from tinygrad import Device
  from tinygrad.helpers import unwrap
  from tinygrad.runtime.graph.hcq import HCQGraph
  from tinygrad.tensor import Tensor
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.q4k_ffn_down_mmvq_qualification import _install

  state={"active":False,"calls":[],"items":[],"start":None,"end":None};orig_graph=HCQGraph.__call__;orig_item=Tensor.item
  def marker(dev,sig):
    unwrap(dev.hw_compute_queue_t)().wait(dev.timeline_signal,dev.timeline_value-1).timestamp(sig) \
      .signal(dev.timeline_signal,dev.next_timeline()).submit(dev)
  def graph_call(self,input_uops,var_vals,wait=False):
    if not state["active"]: return orig_graph(self,input_uops,var_vals,wait=wait)
    idx=len(state["calls"]);dev=self.devices[0]
    if idx==0: state["start"]=dev.new_signal(value=0);marker(dev,state["start"])
    enter=time.perf_counter_ns();ret=orig_graph(self,input_uops,var_vals,wait=wait);exit=time.perf_counter_ns()
    names=[rt.name if rt is not None else "<non-program>" for rt in self.runtimes]
    state["calls"].append({"enter_ns":enter,"exit_ns":exit,"names":names})
    return ret
  def item_call(self):
    if not state["active"]: return orig_item(self)
    enter=time.perf_counter_ns();ret=orig_item(self);exit=time.perf_counter_ns();state["items"].append({"enter_ns":enter,"exit_ns":exit});return ret
  HCQGraph.__call__=graph_call;Tensor.item=item_call

  model=_load("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",a.max_context);_install(model,indices)
  model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",a.depth),chunk_size=32,temperature=0.0)
  rows=[]
  try:
    prelude=int(next(gen));warmup=[int(next(gen)) for _ in range(6)];Device[Device.DEFAULT].synchronize()
    for _ in range(a.reps):
      state.update({"active":True,"calls":[],"items":[],"start":None,"end":None});t0=time.perf_counter_ns();token=int(next(gen));next_return=time.perf_counter_ns()
      state["end"]=Device[Device.DEFAULT].new_signal(value=0);marker(Device[Device.DEFAULT],state["end"])
      Device[Device.DEFAULT].synchronize();t1=time.perf_counter_ns();state["active"]=False
      if not state["calls"] or len(state["items"])!=1 or state["start"] is None or state["end"] is None:
        raise RuntimeError(f"marker accounting calls/items={len(state['calls'])}/{len(state['items'])}")
      calls=state["calls"];item=state["items"][0];device=float(state["end"].timestamp-state["start"].timestamp);wall=(t1-t0)/1e3
      rows.append({"token":token,"wall_us":wall,"device_window_us":device,"outside_device_us":wall-device,
        "pre_first_graph_us":(calls[0]["enter_ns"]-t0)/1e3,
        "graph_call_cpu_sum_us":sum(x["exit_ns"]-x["enter_ns"] for x in calls)/1e3,
        "inter_graph_host_sum_us":sum(calls[i+1]["enter_ns"]-calls[i]["exit_ns"] for i in range(len(calls)-1))/1e3,
        "last_graph_return_to_item_us":(item["enter_ns"]-calls[-1]["exit_ns"])/1e3,
        "item_total_us":(item["exit_ns"]-item["enter_ns"])/1e3,
        "python_yield_tail_us":(next_return-item["exit_ns"])/1e3,"post_next_synchronize_us":(t1-next_return)/1e3,
        "group_members":[len(x["names"]) for x in calls],"program_names":[n for x in calls for n in x["names"]]})
  finally: gen.close();HCQGraph.__call__=orig_graph;Tensor.item=orig_item
  keys=("wall_us","device_window_us","outside_device_us","pre_first_graph_us","graph_call_cpu_sum_us","inter_graph_host_sum_us",
    "last_graph_return_to_item_us","item_total_us","python_yield_tail_us","post_next_synchronize_us")
  median={k:_median_rows(rows,k) for k in keys};hist=collections.Counter(rows[0]["program_names"])
  if any(collections.Counter(x["program_names"])!=hist for x in rows): raise RuntimeError("program histogram changed between reps")
  out={"schema":"tinygrad.q4k_ffn_down_mmvq_marker_window.v1","indices":list(indices),"depth":a.depth,"reps":a.reps,
    "prelude_token":prelude,"warmup_tokens":warmup,"token_stream_hash":hashlib.sha256(",".join(str(x["token"]) for x in rows).encode()).hexdigest(),
    "group_members":rows[0]["group_members"],"program_count":sum(hist.values()),"program_histogram":dict(sorted(hist.items())),"rows":rows,"median":median}
  pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"indices":list(indices),"tokens":[x["token"] for x in rows],**median},indent=2))


if __name__=="__main__": main()
