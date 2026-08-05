#!/usr/bin/env python3
"""Exact local/matched and device/outside equations for the FFN-down route."""
from __future__ import annotations

import argparse, collections, difflib, hashlib, json, pathlib

CONTROL="q4k_g3_lanemap_gemv_4096_12288"
PROVIDER="q8_1_llama_provider_12288"
CONSUMER="q4k_q8_mmvq_direct_4096_12288"


def analyze(control:dict,candidate:dict,wall:dict,marker_control:dict|None=None,marker_candidate:dict|None=None) -> dict:
  ce,be=control["entries"],candidate["entries"];cn=[x["name"] for x in ce];bn=[x["name"] for x in be]
  sm=difflib.SequenceMatcher(a=cn,b=bn,autojunk=False);matched=[];cu=[];bu=[]
  for tag,i1,i2,j1,j2 in sm.get_opcodes():
    if tag=="equal": matched.extend((ce[i1+k],be[j1+k]) for k in range(i2-i1))
    else: cu.extend(ce[i1:i2]);bu.extend(be[j1:j2])
  if [x["name"] for x in cu] != [CONTROL] or [x["name"] for x in bu] != [PROVIDER,CONSUMER]:
    raise ValueError("profile LCS is not the exact one-to-two replacement")
  ch,bh=collections.Counter(cn),collections.Counter(bn);hist_delta={k:bh[k]-ch[k] for k in ch.keys()|bh.keys() if bh[k]!=ch[k]}
  expected={CONTROL:-1,PROVIDER:1,CONSUMER:1}
  if hist_delta != expected: raise ValueError(f"unexpected histogram delta {hist_delta}")
  local_control=sum(float(x["duration"]) for x in cu);local_candidate=sum(float(x["duration"]) for x in bu)
  matched_control=sum(float(x["duration"]) for x,y in matched);matched_candidate=sum(float(y["duration"]) for x,y in matched)
  local_delta=local_candidate-local_control;matched_delta=matched_candidate-matched_control;node_delta=matched_delta+local_delta
  if abs(node_delta-(candidate["node_sum_us"]-control["node_sum_us"])) > 1e-6: raise ValueError("node equation does not close")
  gaps_control=control["device_window_us"]-sum(x["span_us"] for x in control["groups"])
  gaps_candidate=candidate["device_window_us"]-sum(x["span_us"] for x in candidate["groups"])
  span_rounding_delta=(sum(x["span_us"] for x in candidate["groups"])-candidate["node_sum_us"])-(
    sum(x["span_us"] for x in control["groups"])-control["node_sum_us"])
  gap_delta=gaps_candidate-gaps_control;profile_device_delta=candidate["device_window_us"]-control["device_window_us"]
  if abs((node_delta+span_rounding_delta+gap_delta)-profile_device_delta) > 1e-6: raise ValueError("profile device equation does not close")
  profile_outside_delta=candidate["outside_device_us"]-control["outside_device_us"]
  profile_wall_delta=candidate["wall_sync_us"]-control["wall_sync_us"]
  authority_wall_delta_us=float(wall["candidate_minus_control_ms"])*1000
  out={"schema":"tinygrad.q4k_ffn_down_mmvq_profile_analysis.v1",
    "profile_hashes":{"control":hashlib.sha256(json.dumps(control,sort_keys=True).encode()).hexdigest(),
      "candidate":hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest()},
    "topology":{"control_programs":len(ce),"candidate_programs":len(be),"matched_programs":len(matched),
      "control_group_members":[x["members"] for x in control["groups"]],
      "candidate_group_members":[x["members"] for x in candidate["groups"]],"histogram_delta":hist_delta,
      "copy_or_other_program_delta":False},
    "profile_node_equation_us":{"installed":local_control,"provider":float(bu[0]["duration"]),
      "consumer":float(bu[1]["duration"]),"local_delta":local_delta,"matched_control":matched_control,
      "matched_candidate":matched_candidate,"matched_delta":matched_delta,"node_sum_delta":node_delta},
    "profile_window_equation_us":{"node_sum_delta":node_delta,"span_minus_node_rounding_delta":span_rounding_delta,
      "inter_group_gap_delta":gap_delta,"device_window_delta":profile_device_delta,"outside_device_delta":profile_outside_delta,
      "wall_sync_delta":profile_wall_delta},
    "authority":{"settled_wall_delta_us":authority_wall_delta_us,"isolated_device_delta_us":-6.368,
      "isolated_to_wall_inversion_us":authority_wall_delta_us+6.368,
      "profile_local_to_wall_residual_us":authority_wall_delta_us-local_delta},
    "fixed_extra_launch_tax_supported":False,
    "fixed_extra_launch_tax_reason":"the exact unmatched one-to-two region is faster; the single PROFILE replicate drifts in matched nodes",
    "rmsnorm_comparison":{"isolated_delta_us":-6.42915,"wall_delta_us":8.3694375,"inversion_us":14.7985875,
      "same_cause_supported":False,"reason":"the RMSNorm lease constructs ordinary normed_h before recomputing scale in the leased gate/up path; FFN-down has no duplicate/copy delta"},
    "cheapest_generic_fix_seam":{"name":"producer-owned typed boundary without an extra graph member",
      "ffn_construction":"replace the existing silu*mul E_128 producer with a silu*mul+Q8 provider consumed by the direct Q4 kernel",
      "rmsnorm_construction":"select the raw-scale gate/up branch before materializing ordinary normed_h",
      "falsifier":"the one-to-one construction must restore the control program count/partition and win a one-layer direct reverse wall bracket"}}
  if (marker_control is None) != (marker_candidate is None): raise ValueError("marker arms must be supplied together")
  if marker_control is not None and marker_candidate is not None:
    if marker_control["token_stream_hash"] != marker_candidate["token_stream_hash"]: raise ValueError("marker token streams differ")
    md={k:float(marker_candidate["median"][k])-float(marker_control["median"][k]) for k in
      ("wall_us","device_window_us","outside_device_us","pre_first_graph_us","graph_call_cpu_sum_us","inter_graph_host_sum_us",
       "last_graph_return_to_item_us","item_total_us","python_yield_tail_us","post_next_synchronize_us")}
    if abs(md["device_window_us"]+md["outside_device_us"]-md["wall_us"]) > 1e-6: raise ValueError("marker outer equation does not close")
    out["marker_window_equation_us"]={"control":marker_control["median"],"candidate":marker_candidate["median"],"delta":md}
  return out


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--control",type=pathlib.Path,required=True);ap.add_argument("--candidate",type=pathlib.Path,required=True)
  ap.add_argument("--wall",type=pathlib.Path,required=True);ap.add_argument("--marker-control",type=pathlib.Path);ap.add_argument("--marker-candidate",type=pathlib.Path)
  ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  load=lambda p:json.loads(p.read_text())
  out=analyze(load(a.control),load(a.candidate),load(a.wall),load(a.marker_control) if a.marker_control else None,
    load(a.marker_candidate) if a.marker_candidate else None)
  a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__": main()
