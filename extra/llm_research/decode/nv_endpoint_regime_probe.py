#!/usr/bin/env python3
"""Continuous decode windows joined to per-window GPU power/thermal state."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import Device
from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import MODEL
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt

FIELDS=("pstate","clocks.sm","clocks.mem","temperature.gpu","temperature.gpu.tlimit","temperature.memory","power.draw.average","power.draw.instant","clocks_event_reasons.sw_power_cap","clocks_event_reasons.hw_slowdown","clocks_event_reasons.hw_thermal_slowdown","clocks_event_reasons.sw_thermal_slowdown")
def state():
  raw=subprocess.check_output(["nvidia-smi",f"--query-gpu={','.join(FIELDS)}","--format=csv,noheader,nounits"],text=True).strip()
  return dict(zip(FIELDS,[x.strip() for x in raw.split(",")]))
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--depth",type=int,default=512);ap.add_argument("--count",type=int,default=32);ap.add_argument("--reps",type=int,default=15);ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  model=_load(MODEL,a.max_context);model._decode_direct_greedy_promoted=False;model._decode_feedback_pingpong_promoted=False;gen=model.generate(_prompt(MODEL,a.depth),chunk_size=32,temperature=0.0)
  try:
    prelude=int(next(gen));warmup=[int(next(gen)) for _ in range(6)];Device["NV"].synchronize();rows=[];tokens=[]
    for i in range(a.reps):
      before=state();start=time.perf_counter_ns();window=[int(next(gen)) for _ in range(a.count)];Device["NV"].synchronize();elapsed=(time.perf_counter_ns()-start)/a.count/1e6;after=state();tokens.extend(window);rows.append({"index":i,"ms_per_token":elapsed,"before":before,"after":after,"token_hash":hashlib.sha256(",".join(map(str,window)).encode()).hexdigest()})
  finally:gen.close()
  ret={"schema":"tinygrad.nv_endpoint_regime_probe.v1","depth":a.depth,"count":a.count,"reps":a.reps,"prelude_token":prelude,"warmup_tokens":warmup,"windows":rows,"token_stream_hash":hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest()};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(ret,indent=2,sort_keys=True)+"\n");print(json.dumps(ret,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
