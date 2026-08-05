#!/usr/bin/env python3
"""Observational CUDA d512 decode timing ledger.

Records CUDA events immediately before/after each production CUDAGraph replay.
It does not alter graph nodes, their dependencies, launch stream, or scheduling.
The marker-off arm is an end-to-end perturbation control; marker-on rows provide
per-group spans, device gaps, and a whole graph-window remainder.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def median(xs): return statistics.median(xs) if xs else None

def summarize(rows):
  """Pure ledger arithmetic: event positions are in one default-stream order."""
  spans = [r["span_us"] for r in rows]
  gaps = [max(0.0, rows[i+1]["start_us"]-rows[i]["end_us"]) for i in range(len(rows)-1)]
  window = rows[-1]["end_us"]-rows[0]["start_us"] if rows else 0.0
  return {"groups": len(rows), "group_span_sum_us": round(sum(spans), 3),
          "inter_group_gap_sum_us": round(sum(gaps), 3), "device_window_us": round(window, 3),
          "event_reconciliation_error_us": round(window-sum(spans)-sum(gaps), 6),
          "gaps_us": [round(x, 3) for x in gaps]}

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--out", required=True)
  a=ap.parse_args()
  from tinygrad.device import Device
  from tinygrad.helpers import Context
  from tinygrad.runtime.autogen import cuda
  from tinygrad.runtime.ops_cuda import check
  from tinygrad.runtime.graph.cuda import CUDAGraph
  from tinygrad.llm.model import Transformer

  active, calls = {"value": False}, []
  original = CUDAGraph.__call__
  def event():
    e=cuda.CUevent(); check(cuda.cuEventCreate(__import__('ctypes').byref(e), 0)); return e
  def elapsed(x,y):
    import ctypes
    ms=ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(ms), x, y)); return ms.value*1000
  def names_of(graph):
    return [c[1].arg.name if hasattr(c[1].arg, "name") else c[1].op.name for c in graph.calls]
  def wrapped(self, input_uops, var_vals, wait=False):
    if not active["value"]: return original(self, input_uops, var_vals, wait=wait)
    start,end=event(),event()
    check(cuda.cuEventRecord(start, None))
    ret=original(self,input_uops,var_vals,wait=wait)
    check(cuda.cuEventRecord(end, None))
    calls.append({"start":start,"end":end,"names":names_of(self),"call_count":len(self.calls)})
    return ret
  CUDAGraph.__call__=wrapped

  model,_=Transformer.from_gguf(a.model,4608); prompt=[1]*a.depth
  def run(markers:bool):
    model.reset_generation_state(); gen=model.generate(prompt.copy(),chunk_size=32,temperature=0.0)
    # prefill, capture, then settle a replay. The following token is measured.
    with Context(DEBUG=0): next(gen); next(gen); next(gen)
    Device["CUDA"].synchronize()
    active["value"]=markers; calls.clear(); t0=time.perf_counter()
    with Context(DEBUG=0): token=int(next(gen))
    Device["CUDA"].synchronize(); wall_us=(time.perf_counter()-t0)*1e6; active["value"]=False
    if not markers:
      gen.close(); return {"token":token,"wall_us":wall_us}
    if len(calls) != 6: raise RuntimeError(f"expected six CUDA graph groups, saw {len(calls)}")
    check(cuda.cuEventSynchronize(calls[-1]["end"]))
    origin=calls[0]["start"]
    rows=[]
    for i,c in enumerate(calls):
      rows.append({"group":i,"kernels":c["call_count"],"name_sha256":hashlib.sha256("\n".join(c["names"]).encode()).hexdigest(),
                   "start_us":elapsed(origin,c["start"]),"end_us":elapsed(origin,c["end"]),"span_us":elapsed(c["start"],c["end"])})
    for c in calls:
      check(cuda.cuEventDestroy_v2(c["start"])); check(cuda.cuEventDestroy_v2(c["end"]))
    gen.close(); led=summarize(rows); led.update({"token":token,"wall_us":wall_us,"outside_device_window_us":wall_us-led["device_window_us"],"rows":rows}); return led

  # Alternating order makes marker density/order effects visible without assuming stationarity.
  controls=[]; measured=[]
  for _ in range(a.reps): controls.append(run(False)); measured.append(run(True))
  out={"schema":"tinygrad.cuda_decode_group_span_ledger.v1","evidence":"OBSERVATIONAL_EVENT_MARKERS",
       "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"route":"DEV=CUDA CUDA_GRAPH_STREAMS=1",
       "depth":a.depth,"reps":a.reps,"marker_off":controls,"marker_on":measured,
       "median": {"marker_off_wall_us":median([x["wall_us"] for x in controls]),"marker_on_wall_us":median([x["wall_us"] for x in measured]),
                  "device_window_us":median([x["device_window_us"] for x in measured]),"group_span_sum_us":median([x["group_span_sum_us"] for x in measured]),
                  "gap_sum_us":median([x["inter_group_gap_sum_us"] for x in measured]),"outside_window_us":median([x["outside_device_window_us"] for x in measured])}}
  out["median"]["marker_overhead_us"]=out["median"]["marker_on_wall_us"]-out["median"]["marker_off_wall_us"]
  pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["median"],indent=2))
if __name__ == "__main__": main()
