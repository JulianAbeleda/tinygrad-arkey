#!/usr/bin/env python3
"""Replay the installed score->combine lifecycle and isolate combine conversion."""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.nv_flash_full_history_probe import _capture, _exec
from extra.llm_research.decode.nv_r_residual_cache_dispatch_probe import _make_queue

def summary(xs:list[float],warm:int)->dict:
  ys=xs[warm:]
  return {"all_us":xs,"warmup":warm,"n_settled":len(ys),"median_us":statistics.median(ys),
    "mean_us":statistics.mean(ys),"min_us":min(ys),"max_us":max(ys)}

def measure(dev,target,prefix,n,mode,score=None):
  out=[]
  for _ in range(n):
    q=_make_queue(dev);_exec(q,target)  # warm code and target input
    if mode in ("prefix_target","prefix_reheat_target"):
      for x in prefix:_exec(q,x)
      if mode=="prefix_reheat_target":_exec(q,target)
    elif mode in ("score_target","island"):
      assert score is not None
      if mode=="score_target":_exec(q,score)
    a,b=dev.new_signal(),dev.new_signal();q.timestamp(a)
    if mode=="island":_exec(q,score)
    _exec(q,target);q.timestamp(b);q.signal(dev.timeline_signal,dev.next_timeline()).submit(dev)
    dev.synchronize();out.append(float(b.timestamp-a.timestamp))
  return out

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--depth",type=int,default=512);ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--tokens",type=int,default=4);ap.add_argument("--n",type=int,default=35);ap.add_argument("--warmup",type=int,default=5)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  from tinygrad import Device
  trace,cap=_capture(a.depth,a.max_context,a.tokens,"flash_vec_llama_score_pv_32_128_6_widekv16*");dev=Device["NV"]
  combines=[i for i,x in enumerate(trace) if x["name"].startswith("flash_fused_gmax_combine_f16_32_128_s6")]
  if len(combines)<2:raise RuntimeError(f"need installed S6 combines, got {len(combines)}")
  ti=combines[-1];pi=combines[-2];target=trace[ti]
  scores=[i for i in range(pi+1,ti) if trace[i]["name"].startswith("flash_vec_llama_score_pv_32_128_6")]
  if not scores:raise RuntimeError("no S6 score immediately before selected combine")
  si=scores[-1];score=trace[si];prefix=trace[pi+1:ti]
  rows={
    "hot_repeat":summary(measure(dev,target,[],a.n,"hot",score),a.warmup),
    "score_then_combine":summary(measure(dev,target,[],a.n,"score_target",score),a.warmup),
    "complete_interval_then_combine":summary(measure(dev,target,prefix,a.n,"prefix_target",score),a.warmup),
    "complete_interval_reheat_then_combine":summary(measure(dev,target,prefix,a.n,"prefix_reheat_target",score),a.warmup),
    "score_to_combine_end_island":summary(measure(dev,target,[],a.n,"island",score),a.warmup),
  }
  result={"schema":"tinygrad.nv_flash_combine_conversion_probe.v1","capture":cap,
    "selected":{"combine_index":ti,"combine_name":target["name"],"score_index":si,"score_name":score["name"],
      "prefix_count":len(prefix),"prefix_names":[x["name"] for x in prefix]},"rows":rows}
  result["deltas_us"]={"score_conditioning":rows["score_then_combine"]["median_us"]-rows["hot_repeat"]["median_us"],
    "full_interval_conditioning":rows["complete_interval_then_combine"]["median_us"]-rows["hot_repeat"]["median_us"],
    "reheat_recovery":rows["complete_interval_then_combine"]["median_us"]-rows["complete_interval_reheat_then_combine"]["median_us"]}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
