#!/usr/bin/env python3
"""Test whether omitted production history explains Flash's replay-to-token gap.

Capture every native kernel launch from a real dense decode, select a late
wide-Flash score, and replay exact one- and two-score intervals before timing
that score.  Counter-arms reheat the target working set after the history, so
cache displacement can be separated from a residual launch/service effect.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_r_residual_cache_dispatch_probe import _make_queue
from extra.llm_research.decode.nv_flash_entry_hop_ledger import MODEL, TARGET, SELECTED, _buf_meta

PRODUCTION_PROFILE = ROOT / "docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-ledger/production.profile.jsonl"


def _production_layer_reference(path:pathlib.Path, layer:int=35) -> dict:
  chunks=[]
  for line in path.read_text().splitlines():
    try: obj=json.loads(line)
    except json.JSONDecodeError: continue
    vals=[float(x["duration"]) for x in obj.get("entries",[]) if x.get("name") == TARGET]
    if vals: chunks.append(vals)
  graphs=[]; current=[]
  for chunk in chunks:
    current.extend(chunk)
    if len(current) == 36: graphs.append(current); current=[]
    elif len(current) > 36: raise RuntimeError("production Flash chunks do not reconcile to 36 layers")
  if current or not graphs: raise RuntimeError("incomplete production Flash graph grouping")
  vals=[x[layer] for x in graphs]
  return {"profile":str(path.relative_to(ROOT)),"layer":layer,"n":len(vals),
    "median_us":round(statistics.median(vals),3),"mean_us":round(statistics.mean(vals),3),
    "min_us":round(min(vals),3),"max_us":round(max(vals),3)}


def _capture(depth:int, max_context:int, tokens:int, target_name:str=TARGET) -> tuple[list[dict], dict]:
  import tinygrad.runtime.ops_nv as ops_nv
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

  trace:list[dict] = []
  orig_call = ops_nv.NVProgram.__call__

  def patched_call(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, timeout=None):
    trace.append({"name":self.name, "program":self, "bufs":tuple(bufs), "vals":tuple(vals),
      "global_size":tuple(global_size), "local_size":tuple(local_size), "buf_meta":[_buf_meta(x) for x in bufs]})
    return orig_call(self, *bufs, global_size=global_size, local_size=local_size, vals=vals,
                     wait=wait, timeout=timeout)

  ops_nv.NVProgram.__call__ = patched_call
  try:
    model = _load(MODEL, max_context)
    model._decode_direct_greedy_promoted = True
    model._decode_feedback_pingpong_promoted = True
    gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
    try:
      produced = [int(next(gen)) for _ in range(tokens)]
      Device["NV"].synchronize()
    finally: gen.close()
  finally: ops_nv.NVProgram.__call__ = orig_call

  target_indices = [i for i,x in enumerate(trace) if x["name"].startswith(target_name[:-1])]
  if not target_name.endswith("*"): target_indices = [i for i,x in enumerate(trace) if x["name"] == target_name]
  if len(target_indices) < 3:
    observed=sorted({x["name"] for x in trace if "flash" in x["name"]})
    raise RuntimeError(f"need at least three {target_name} calls, found {len(target_indices)}; observed={observed}")
  # A late layer avoids graph-build prologue effects while retaining two complete
  # predecessor intervals. Calls are captured with their live production buffers.
  ti = target_indices[-1]
  pi, ppi = target_indices[-2], target_indices[-3]
  return trace, {"tokens":produced, "call_count":len(trace), "target_call_count":len(target_indices),
    "target_indices":{"two_back":ppi, "previous":pi, "target":ti}}


def _exec(queue, rec:dict) -> None:
  queue.exec(rec["program"], rec["program"].fill_kernargs(rec["bufs"], vals=rec["vals"]),
             rec["global_size"], rec["local_size"])


def _measure(dev, target:dict, prefix:list[dict], reheat:bool, n:int, timeout_s:float) -> list[float]:
  samples=[]
  for _ in range(n):
    q=_make_queue(dev)
    _exec(q,target)
    for rec in prefix: _exec(q,rec)
    if reheat: _exec(q,target)
    start,end=dev.new_signal(),dev.new_signal()
    q.timestamp(start); _exec(q,target); q.timestamp(end)
    q.signal(dev.timeline_signal,dev.next_timeline()).submit(dev)
    dev.synchronize(timeout=int(timeout_s*1000))
    samples.append(float(end.timestamp-start.timestamp))
  return samples


def _measure_fork_join(dev, target:dict, interval:list[dict], reheat:bool, n:int, timeout_s:float) -> list[float]:
  """Replay the installed Q versus K/V fork and join instead of serializing it."""
  from tinygrad.runtime.ops_nv import NVComputeQueue
  qpos=next(i for i,x in enumerate(interval) if x["name"].startswith("q4k_g3_lanemap_gemv_vec_4096_4096"))
  kpos=next(i for i,x in enumerate(interval) if x["name"].startswith("q4k_g3_lanemap_gemv_vec_1024_4096"))
  vpos=next(i for i,x in enumerate(interval) if x["name"].startswith("q6k_v_four_warp_fp16_direct_1024_4096"))
  qdone=next(i for i,x in enumerate(interval) if x["name"] == "reduce_output_rmsnorm_rope_32_128")
  kvdone=next(i for i,x in enumerate(interval) if x["name"] == "reduce_output_rmsnorm_rope_kv_cache_8_128")
  if not (qpos < kpos < vpos < qdone < kvdone): raise RuntimeError("unexpected installed Q/K/V interval order")
  pre=interval[:qpos]; qbranch=(interval[qpos],interval[qdone]); kvbranch=(interval[kpos],interval[vpos],interval[kvdone])
  samples=[]
  for _ in range(n):
    pre_ready,kv_ready=dev.new_signal(),dev.new_signal()
    q0=_make_queue(dev)
    _exec(q0,target)
    for rec in pre: _exec(q0,rec)
    q0.signal(pre_ready,1)
    for rec in qbranch: _exec(q0,rec)
    q0.wait(kv_ready,1)
    if reheat: _exec(q0,target)
    start,end=dev.new_signal(),dev.new_signal()
    q0.timestamp(start); _exec(q0,target); q0.timestamp(end)
    q0.signal(dev.timeline_signal,dev.next_timeline())

    q1=NVComputeQueue(queue_idx=1)
    q1.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
             shared_mem_window=dev.shared_mem_window)
    q1.wait(pre_ready,1)
    for rec in kvbranch: _exec(q1,rec)
    q1.signal(kv_ready,1)
    q0.submit(dev); q1.submit(dev)
    dev.synchronize(timeout=int(timeout_s*1000))
    samples.append(float(end.timestamp-start.timestamp))
  return samples


def _summary(xs:list[float], warmup:int) -> dict:
  ys=xs[warmup:]
  return {"median_us":round(statistics.median(ys),3), "mean_us":round(statistics.mean(ys),3),
    "min_us":round(min(ys),3), "max_us":round(max(ys),3), "samples_us":[round(x,3) for x in ys]}


def main() -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--capture-tokens",type=int,default=8); ap.add_argument("--n",type=int,default=48)
  ap.add_argument("--warmup",type=int,default=8); ap.add_argument("--timeout-s",type=float,default=180.0)
  ap.add_argument("--production-profile",type=pathlib.Path,default=PRODUCTION_PROFILE)
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  os.environ.update(DEV="NV",PROFILE="0")
  trace,capture=_capture(args.depth,args.max_context,args.capture_tokens)
  idx=capture["target_indices"]; target=trace[idx["target"]]
  one=trace[idx["previous"]+1:idx["target"]]
  two=trace[idx["two_back"]+1:idx["target"]]
  selected=[x for x in one if x["name"] in SELECTED and x["name"] != TARGET]
  omitted=[x for x in one if x not in selected]
  definitions={
    "hot":([],False),
    "selected_named_prefix":(selected,False),
    "exact_one_score_interval":(one,False),
    "exact_one_score_interval_reheat":(one,True),
    "exact_two_score_intervals":(two,False),
    "exact_two_score_intervals_reheat":(two,True),
  }
  from tinygrad import Device
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _gpu_state
  dev=Device["NV"]; rows={}
  for name,(prefix,reheat) in definitions.items():
    rows[name]=_summary(_measure(dev,target,prefix,reheat,args.n,args.timeout_s),args.warmup)
    print(json.dumps({name:rows[name]},sort_keys=True),flush=True)
  for name,reheat in (("exact_one_score_fork_join",False),("exact_one_score_fork_join_reheat",True)):
    rows[name]=_summary(_measure_fork_join(dev,target,one,reheat,args.n,args.timeout_s),args.warmup)
    print(json.dumps({name:rows[name]},sort_keys=True),flush=True)
  hot=rows["hot"]["median_us"]
  production_reference=_production_layer_reference(args.production_profile)
  production_us=production_reference["median_us"]
  for row in rows.values():
    row["minus_hot_us"]=round(row["median_us"]-hot,3)
    row["distance_to_production_us"]=round(production_us-row["median_us"],3)
  payload={"schema":"tinygrad.nv_flash_full_history_probe.v1",
    "commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
    "gpu_state":_gpu_state(), "shape":{"depth":args.depth,"max_context":args.max_context},
    "method":"real decode capture; exact native calls/live buffers between late production score launches; only final score timestamped",
    "capture":capture,
    "interval":{"one_score_call_count":len(one),"two_score_call_count":len(two),
      "selected_named_count":len(selected),"omitted_count":len(omitted),
      "one_score_names":[x["name"] for x in one],"omitted_names":[x["name"] for x in omitted]},
    "production_reference":production_reference,"n_per_arm":args.n-args.warmup,"rows":rows,
    "reconciliation":{
      "selected_penalty_us":rows["selected_named_prefix"]["minus_hot_us"],
      "exact_interval_penalty_us":rows["exact_one_score_interval"]["minus_hot_us"],
      "extra_history_penalty_us":round(rows["exact_one_score_interval"]["median_us"]-rows["selected_named_prefix"]["median_us"],3),
      "reheat_recovery_us":round(rows["exact_one_score_interval"]["median_us"]-rows["exact_one_score_interval_reheat"]["median_us"],3),
      "fork_join_minus_serial_us":round(rows["exact_one_score_fork_join"]["median_us"]-rows["exact_one_score_interval"]["median_us"],3),
      "fork_join_reheat_recovery_us":round(rows["exact_one_score_fork_join"]["median_us"]-rows["exact_one_score_fork_join_reheat"]["median_us"],3),
      "unexplained_after_exact_interval_us":rows["exact_one_score_interval"]["distance_to_production_us"],
      "fork_join_minus_production_reference_us":round(rows["exact_one_score_fork_join"]["median_us"]-production_us,3),
      "endpoint_translation_tok_s":{"endpoint_us":4094.502,"layers":36,
        "exact_interval_hot_ceiling":round(1e6/(4094.502-rows["exact_one_score_interval"]["minus_hot_us"]*36),3),
        "extra_history_only_ceiling":round(1e6/(4094.502-(rows["exact_one_score_interval"]["median_us"]-rows["selected_named_prefix"]["median_us"])*36),3)},
    }}
  args.out.parent.mkdir(parents=True,exist_ok=True)
  args.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps(payload["reconciliation"],indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
