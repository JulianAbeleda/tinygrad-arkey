#!/usr/bin/env python3
"""Marker-light one-token HCQ decomposition for the leased FFN-down route."""
from __future__ import annotations

import argparse, json, os, pathlib, sys, time


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--indices",default="");ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--out",required=True)
  a=ap.parse_args();indices=tuple(int(x) for x in a.indices.split(",") if x)
  profile_path=pathlib.Path(a.out).with_suffix(".hcq.jsonl")
  os.environ["PROFILE"]="1";os.environ["HCQ_GRAPH_PROFILE_JSON"]=str(profile_path)

  from tinygrad import Device
  from tinygrad.llm.q4k_ffn_down_mmvq import Q4KFFNDownMMVQAdmission
  from tinygrad.runtime.graph.hcq import HCQGraph
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.q4k_ffn_down_mmvq_qualification import _install

  launches=[];capture={"on":False};orig=HCQGraph.__call__
  def traced(self,input_uops,var_vals,wait=False):
    names=tuple(rt.name if rt is not None else "<non-program>" for rt in self.runtimes)
    t0=time.perf_counter_ns();ret=orig(self,input_uops,var_vals,wait=wait);t1=time.perf_counter_ns()
    if capture["on"]: launches.append({"names":names,"host_call_us":(t1-t0)/1e3})
    return ret
  HCQGraph.__call__=traced

  model=_load("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",a.max_context);_install(model,indices)
  model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",a.depth),chunk_size=32,temperature=0.0)
  try:
    prelude=int(next(gen));warmup=[int(next(gen)) for _ in range(6)];Device[Device.DEFAULT].synchronize()
    profile_path.unlink(missing_ok=True);capture["on"]=True;t0=time.perf_counter_ns();token=int(next(gen));
    Device[Device.DEFAULT].synchronize();wall_sync_us=(time.perf_counter_ns()-t0)/1e3;capture["on"]=False
    # Ping-pong has two graph instances. Replay the opposite slot, then the
    # measured slot, so the latter collects the measured timestamp payload.
    flush_tokens=[int(next(gen)),int(next(gen))];Device[Device.DEFAULT].synchronize()
  finally: gen.close();HCQGraph.__call__=orig

  lines=[json.loads(x) for x in profile_path.read_text().splitlines() if x.strip()]
  if len(launches)!=5 or len(lines)<5: raise RuntimeError(f"profile accounting launches={len(launches)} lines={len(lines)}")
  measured=lines[-len(launches):]
  groups=[];all_entries=[]
  for group,(launch,line) in enumerate(zip(launches,measured)):
    entries=line["entries"]
    if len(entries)!=len(launch["names"]): raise RuntimeError(f"group {group} entry/name mismatch")
    for index,(name,entry) in enumerate(zip(launch["names"],entries)):
      if name != entry["name"]: raise RuntimeError(f"group {group} index {index} profile name mismatch")
      all_entries.append({"group":group,"index":index,**entry})
    starts=[float(x["start"]) for x in entries];ends=[float(x["end"]) for x in entries]
    durations=[float(x["duration"]) for x in entries]
    gaps=[max(0.,starts[i]-ends[i-1]) for i in range(1,len(entries))]
    groups.append({"group":group,"members":len(entries),"host_call_us":launch["host_call_us"],
      "node_sum_us":sum(durations),"span_us":max(ends)-min(starts),"positive_inter_node_gap_us":sum(gaps)})
  starts=[float(x["start"]) for x in all_entries];ends=[float(x["end"]) for x in all_entries]
  device_window_us=max(ends)-min(starts)
  out={"schema":"tinygrad.q4k_ffn_down_mmvq_profile.v1","indices":list(indices),"depth":a.depth,
    "callify_owned_precompiled_output_redirect":int(os.environ.get("CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT","0")),
    "composed":True,"prelude_token":prelude,"warmup_tokens":warmup,"measured_token":token,"flush_tokens":flush_tokens,
    "wall_sync_us":wall_sync_us,"device_window_us":device_window_us,"outside_device_us":wall_sync_us-device_window_us,
    "node_sum_us":sum(float(x["duration"]) for x in all_entries),"groups":groups,"entries":all_entries}
  pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  print(json.dumps({k:out[k] for k in ("indices","wall_sync_us","device_window_us","outside_device_us","node_sum_us")},indent=2))


if __name__=="__main__": main()
