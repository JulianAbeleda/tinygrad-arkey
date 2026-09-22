#!/usr/bin/env python3
"""Measure the incremental capture cost of the adaptive S48/S64 decode policy."""
from __future__ import annotations

import argparse, json, pathlib, time

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def _snapshot(model, label:str) -> dict:
  from tinygrad import Device
  from tinygrad.helpers import GlobalCounters
  Device[Device.DEFAULT].synchronize()
  pairs = {
    "s48": model.rollout_greedy_pingpong_jits_flash,
    "s64": model.rollout_greedy_pingpong_jits_flash_s64,
  }
  return {"label":label, "global_mem_bytes":GlobalCounters.mem_used,
          "device_mem_bytes":GlobalCounters.mem_used_per_device[Device.DEFAULT],
          "captures":{name:[None if j.captured is None else len(j.captured.linear.src) for j in pair]
                      for name,pair in pairs.items()}}

def _capture_pair(model, split_count:int, pos:int) -> float:
  from tinygrad import Device, Tensor, UOp
  tok, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, model.max_context-1)
  Device[Device.DEFAULT].synchronize()
  started = time.perf_counter()
  for slot in (0, 1):
    for _ in range(3):
      model(tok, start_pos.bind(pos), temp, use_flash=True, greedy=True,
            feedback_slot=slot, flash_split_count=split_count).realize()
  Device[Device.DEFAULT].synchronize()
  return time.perf_counter() - started

def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--out",default="/tmp/nv-flash-adaptive-deployment.json")
  args=ap.parse_args()
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load
  model=_load(MODEL,args.max_context)
  model._flash_decode_adaptive_s64_lease=True
  model._decode_direct_greedy_promoted=True
  model._decode_feedback_pingpong_promoted=True
  rows=[_snapshot(model,"loaded")]
  s48_s=_capture_pair(model,48,min(700,args.max_context-1)); rows.append(_snapshot(model,"s48_captured"))
  s64_s=_capture_pair(model,64,min(800,args.max_context-1)); rows.append(_snapshot(model,"s64_captured"))
  result={"schema":"tinygrad.nv_flash_adaptive_deployment.v1","max_context":args.max_context,
          "s48_capture_seconds":s48_s,"s64_incremental_capture_seconds":s64_s,"snapshots":rows,
          "s48_incremental_bytes":rows[1]["device_mem_bytes"]-rows[0]["device_mem_bytes"],
          "s64_incremental_bytes":rows[2]["device_mem_bytes"]-rows[1]["device_mem_bytes"]}
  pathlib.Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True))
  return 0

if __name__ == "__main__": raise SystemExit(main())
