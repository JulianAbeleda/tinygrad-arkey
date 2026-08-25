#!/usr/bin/env python3
"""A/B/A production-token sensitivity to SM clock with memory clock unchanged."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[3]
HARNESS=ROOT/"extra/llm_research/decode/qk_norm_rope_wall_bracket.py"


def gpu_state()->dict:
  fields=("name","pstate","clocks.sm","clocks.mem","temperature.gpu","power.draw")
  line=subprocess.check_output(["nvidia-smi",f"--query-gpu={','.join(fields)}","--format=csv,noheader,nounits"],text=True).strip()
  return dict(zip(fields,(x.strip() for x in line.split(","))))


def arm(clock:int,label:str,args,root:pathlib.Path)->dict:
  subprocess.run(["sudo","nvidia-smi","--lock-gpu-clocks",f"{clock},{clock}"],check=True,stdout=subprocess.PIPE,text=True)
  out=root/f"{label}.json";cmd=[sys.executable,str(HARNESS),"--mode","timing-child","--production-default","--composed",
    "--depth",str(args.depth),"--count",str(args.count),"--reps",str(args.reps),"--max-context",str(args.max_context),"--out",str(out)]
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,"DEV":"NV","PYTHONPATH":str(ROOT)})
  if run.returncode:raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
  row=json.loads(out.read_text());row["requested_sm_clock_mhz"]=clock;row["post_arm_gpu_state"]=gpu_state();out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");return row


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--high",type=int,default=2895);ap.add_argument("--low",type=int,default=2100)
  ap.add_argument("--depth",type=int,default=512);ap.add_argument("--count",type=int,default=32);ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args()
  root=pathlib.Path(str(args.out).removesuffix(".json"));root.mkdir(parents=True,exist_ok=True)
  try:rows=[arm(args.high,"high_a",args,root),arm(args.low,"low",args,root),arm(args.high,"high_c",args,root)]
  finally:subprocess.run(["sudo","nvidia-smi","--reset-gpu-clocks"],check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  high=statistics.median((rows[0]["median_ms_per_token"],rows[2]["median_ms_per_token"]));low=rows[1]["median_ms_per_token"]
  hashes={x["token_stream_hash"] for x in rows};clock_drop=1-args.low/args.high;latency_growth=low/high-1
  measured_high=statistics.median((int(rows[0]["gpu_state"]["clocks.sm"]),int(rows[2]["gpu_state"]["clocks.sm"])))
  measured_low=int(rows[1]["gpu_state"]["clocks.sm"]);inverse_clock_growth=measured_high/measured_low-1
  result={"schema":"tinygrad.nv_sm_clock_sensitivity.v1","high_clock_mhz":args.high,"low_clock_mhz":args.low,
    "memory_clock_mhz":rows[1]["post_arm_gpu_state"]["clocks.mem"],"high_midpoint_ms_per_token":high,"low_ms_per_token":low,
    "latency_growth_pct":100*latency_growth,"sm_clock_drop_pct":100*clock_drop,
    "normalized_latency_sensitivity":latency_growth/clock_drop,"measured_loaded_high_sm_clock_mhz":measured_high,
    "measured_loaded_low_sm_clock_mhz":measured_low,
    "inverse_clock_scaled_wall_fraction":latency_growth/inverse_clock_growth,
    "all_token_hashes_equal":len(hashes)==1,"arms":rows}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
