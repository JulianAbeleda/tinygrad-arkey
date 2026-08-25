#!/usr/bin/env python3
"""Assemble the final machine-readable dense FFN composition ledger."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, subprocess

ROOT=pathlib.Path(__file__).resolve().parents[3]


def load(path:pathlib.Path)->dict:return json.loads(path.read_text())
def sha(path:pathlib.Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path:pathlib.Path,data:dict)->None:path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
def metric(data:dict,name:str)->float:
  row=next(x for x in data["ncu"]["rows"] if x["metric"]==name);return float(row["value"].replace(",",""))


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=pathlib.Path,required=True);args=ap.parse_args();root=args.root
  edge=load(root/"ffn-edge-ledger.json");control=load(root/"control-profile.json");candidate=load(root/"candidate-profile.json")
  wall=load(root/"q6-down-u4-wall-r9.json");installed=load(root/"installed-wall.json")
  gate=load(root/"gateup-cold-counters.json");q4=load(ROOT/"docs/task_workflow/evidence/nv-q4k-down-vector-load-20260824/microgate.json")
  q6=load(root/"q6-down-cold-counters.json");q6u4=load(root/"q6-down-u4-cold-counters.json");exact=load(root/"q6-down-unroll-microgate.json")
  commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
  status=subprocess.check_output(["git","-C",str(ROOT),"status","--short"],text=True).splitlines()
  model=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,pci.bus_id,clocks.current.graphics,clocks.current.memory,pstate",
    "--format=csv,noheader"],text=True).strip()
  write(root/"provenance.json",{"schema":"tinygrad.nv_dense_ffn_composition_provenance.v1","commit":commit,
    "model":str(model),"model_sha256":sha(model),"gpu":gpu,"local_status":status,
    "route_environment":{k:v for k,v in __import__("os").environ.items() if k.startswith(("TINYGRAD_","NV_","HCQ_"))}})
  write(root/"ffn-cold-counters.json",{"schema":"tinygrad.nv_dense_ffn_cold_counters.v1","sources":{
    "gateup":{"path":str(root/"gateup-cold-counters.json"),"sha256":sha(root/"gateup-cold-counters.json")},
    "q4_down":{"path":"docs/task_workflow/evidence/nv-q4k-down-vector-load-20260824/microgate.json",
      "sha256":sha(ROOT/"docs/task_workflow/evidence/nv-q4k-down-vector-load-20260824/microgate.json")},
    "q6_down":{"path":str(root/"q6-down-cold-counters.json"),"sha256":sha(root/"q6-down-cold-counters.json")},
    "q6_down_u4":{"path":str(root/"q6-down-u4-cold-counters.json"),"sha256":sha(root/"q6-down-u4-cold-counters.json")}},
    "attribution":{"gateup_dram_pct":float(next(x["value"] for x in gate["ncu"]["current"]["rows"] if x["metric"].startswith("dram__throughput"))),
      "q4_down_vector_dram_pct":float(next(x["value"] for x in q4["ncu"]["vector"]["rows"] if x["metric"].startswith("dram__throughput"))),
      "q6_control_dram_pct":metric(q6,"dram__throughput.avg.pct_of_peak_sustained_elapsed"),
      "q6_u4_dram_pct":metric(q6u4,"dram__throughput.avg.pct_of_peak_sustained_elapsed"),
      "q6_control_long_scoreboard_pct":metric(q6,"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
      "q6_u4_long_scoreboard_pct":metric(q6u4,"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
      "q6_control_duration_ns":metric(q6,"gpu__time_duration.sum"),"q6_u4_duration_ns":metric(q6u4,"gpu__time_duration.sum")}})
  write(root/"candidate-contract.json",{"schema":"tinygrad.nv_dense_ffn_candidate_contract.v1","candidate":"Q6 packed-lane four-block unroll",
    "population":18,"input_contract":"unchanged Q6_K weights, fp16 activation, fp32 residual","output_contract":"unchanged fp32[4096] residual-added output",
    "arithmetic_order":"left-to-right block accumulation preserved","compulsory_bytes":"unchanged","topology":"unchanged",
    "rollback":"TINYGRAD_Q6K_FFN_DOWN_UNROLL_DISABLE=1"})
  write(root/"candidate-exactness.json",{"schema":"tinygrad.nv_dense_ffn_candidate_exactness.v1","microgate":exact["correctness"],
    "wall_all_token_hashes_equal":wall["all_token_hashes_equal"],"token_stream_hash":wall["token_stream_hash"]})
  control_q6=next(x for x in control["rows"] if x["name"]=="q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd")
  candidate_q6=next(x for x in candidate["rows"] if x["name"]=="q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd")
  installed_us=float(installed["median_ms_per_token"])*1000;llama_us=4048.3246
  conservative=min(float(wall["control_a_ms_per_token"]),float(wall["control_c_ms_per_token"]))*1000-float(wall["candidate_ms_per_token"])*1000
  final={"schema":"tinygrad.nv_dense_ffn_composition_final_ledger.v1","commit":commit,"verdict":"PROMOTE_FFN_COLD_RATE_CONSTRUCTION",
    "H1":{"verdict":edge["hypothesis"],"edge_count":edge["edge_count"],"edge_wait_us":edge["edge_wait_us"]},
    "H2":{"verdict":"PROMOTE_Q6_PACKED_LANEMAP_U4","control_q6_us_per_token":control_q6["wall_us_per_replay"],
      "candidate_q6_us_per_token":candidate_q6["wall_us_per_replay"],"q6_row_recovery_us":control_q6["wall_us_per_replay"]-candidate_q6["wall_us_per_replay"]},
    "wall":{"control_a_us":wall["control_a_ms_per_token"]*1000,"candidate_us":wall["candidate_ms_per_token"]*1000,
      "control_c_us":wall["control_c_ms_per_token"]*1000,"midpoint_recovery_us":wall["recovery_us_per_token"],
      "conservative_booked_recovery_us":conservative,"installed_us":installed_us,"installed_tok_s":1e6/installed_us,
      "remaining_to_240_us":installed_us-1e6/240,"remaining_to_llama_us":installed_us-llama_us},
    "device":{"control_node_sum_us":control["closure"]["node_sum_us"],"candidate_node_sum_us":candidate["closure"]["node_sum_us"],
      "node_sum_recovery_us":control["closure"]["node_sum_us"]-candidate["closure"]["node_sum_us"],
      "control_union_us":control["closure"]["union_us"],"candidate_union_us":candidate["closure"]["union_us"],
      "union_recovery_us":control["closure"]["union_us"]-candidate["closure"]["union_us"]},
    "next":"attention lifecycle; FFN composition H1 is closed and the admitted H2 construction is booked"}
  write(root/"final-ledger.json",final);print(json.dumps(final,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
