#!/usr/bin/env python3
"""Disjoint semantic accounting for native and llama decode support work.

Native attribution is deliberately fail-closed: it recognizes the exact d512
36-layer call template and exact unhashed kernel signatures.  Llama attribution
uses symmetric Shapley ownership of every elementary timeline interval, so role
rows add to the aggregate union even when non-MMQ classes overlap each other.
"""
from __future__ import annotations

import argparse, json, pathlib, sqlite3, statistics
from collections import Counter

from extra.llm_research.decode.cuda_graph_timeline_ledger import _load_rows, _split_replays
from extra.llm_research.decode.native_semantic_profile_ledger import clean, sha256

SCHEMA = "tinygrad.nv_decode.nonquant_role_partition.v1"
PREFIX = ["E", "E_2", "E_16_4_2_8_16_2_4_4", "E_1187_32_4", "r_32_32_4_32_4", "r_16_256", "E_32_32_4"]
LAYER = {
  3:("q_norm", "E_4_2_8_16_4"), 4:("k_norm", "E_2_8_16_4"),
  5:("q_norm", "r_2_8_4_4_16"), 6:("k_norm", "r_8_16_8"),
  7:("q_norm", "E_2_8_16_4_4"), 8:("k_norm", "E_8_2_16_4"),
  9:("rope_q", "E_16_32_4_2"),
  11:("flash_score", "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"),
  12:("flash_combine", "flash_fused_gmax_combine_32_128"),
  13:("attention_cast", "E_32_32_4"), 15:("attention_residual_add", "E_32_32_4"),
  16:("ffn_rmsnorm", "r_16_256"), 17:("ffn_rmsnorm", "E_32_32_4"),
  19:("ffn_activation_cast", "E_128_32_3"), 21:("ffn_down_cast", "E_32_32_4"),
  22:("ffn_residual_add", "E_32_32_4"), 23:("block_output_contiguous", "E_32_32_4"),
  24:("next_or_final_rmsnorm", "r_16_256"), 25:("next_or_final_rmsnorm", "E_32_32_4"),
}


def _starts(nodes:list[dict]) -> list[int]:
  out=[]
  for i,node in enumerate(nodes):
    sem=(node.get("metadata") or {}).get("semantic") or []
    if sem and sem[0].get("tensor_name", "").endswith("attn_q.weight"): out.append(i)
  return out


def native_roles(dag_path:pathlib.Path, profile_path:pathlib.Path, scale:float) -> dict:
  nodes=json.loads(dag_path.read_text())["nodes"]
  records=[json.loads(line) for line in profile_path.read_text().splitlines() if line.strip()]
  entries=[entry for record in records for entry in record.get("entries", [])]
  if len(nodes) != 948 or len(entries) != 948: raise ValueError("expected exact 948-node native token")
  for i,(node,entry) in enumerate(zip(nodes,entries)):
    if node["name"] != entry["name"]: raise ValueError(f"profile/DAG mismatch at {i}")
  starts=_starts(nodes)
  if starts != [7+26*i for i in range(36)]: raise ValueError(f"unexpected layer starts {starts}")
  rows=[]
  def add(i:int, role:str): rows.append({"node":i,"role":role,"name":nodes[i]["name"],"us":float(entries[i]["duration"])*scale})
  for i,(want,role) in enumerate(zip(PREFIX, ("token_feedback","token_feedback","vocab_sampler","vocab_sampler",
                                               "vocab_sampler","initial_rmsnorm","initial_rmsnorm"))):
    if clean(nodes[i]["name"]) != want: raise ValueError(f"prefix mismatch at {i}: {nodes[i]['name']}")
    add(i,role)
  for layer,start in enumerate(starts):
    quant={0:"attn_q",1:"attn_k",2:"attn_v",14:"attn_o",18:"ffn_gate_up",20:"ffn_down"}
    for rel,role in quant.items():
      sem=(nodes[start+rel].get("metadata") or {}).get("semantic") or []
      if not sem: raise ValueError(f"missing quant identity layer {layer} rel {rel}")
    for rel,(role,prefix) in LAYER.items():
      if clean(nodes[start+rel]["name"]) != prefix:
        raise ValueError(f"layer {layer} rel {rel} mismatch: {nodes[start+rel]['name']}")
      add(start+rel,role)
    # The promoted KV-store kernel owns K RoPE, K/V casts and store.  On Q6 V
    # layers it also owns the four-partial reduction; preserve that compound ABI.
    vsem=(nodes[start+2].get("metadata") or {})["semantic"][0]
    kvrole="kv_store_k_rope_cast_with_q6_partial_reduce" if vsem["source_quant_storage"] == "Q6_K" else "kv_store_k_rope_cast"
    want="r_8_8_16_2_4" if vsem["source_quant_storage"] == "Q6_K" else "E_8_8_16_2"
    if clean(nodes[start+10]["name"]) != want: raise ValueError(f"KV variant mismatch layer {layer}")
    add(start+10,kvrole)
  # Final vocab head is quantized node 943; only its disjoint sampler tail is here.
  for i,want in zip(range(944,948),("E_1187_32_4","r_32_4_1187","r_128_16_8_1187","r_16_8")):
    if clean(nodes[i]["name"]) != want: raise ValueError(f"suffix mismatch at {i}")
    add(i,"vocab_sampler")
  if len(rows) != 731 or len({r["node"] for r in rows}) != 731: raise ValueError("native partition is not exactly 731 disjoint nodes")
  totals=Counter()
  for row in rows: totals[row["role"]] += row["us"]
  return {"nodes":rows,"roles_us":{k:round(v,3) for k,v in sorted(totals.items())},
          "node_count":len(rows),"total_us":round(sum(totals.values()),3)}


def _shapley_partition(rows:list[dict], anchor="mmq") -> tuple[dict,dict,dict]:
  """Partition exposed/hidden non-anchor unions over elementary intervals.

  If n roles cover an interval, symmetric Shapley ownership is dt/n each.
  The order-marginal lower/upper bounds are unique coverage/full coverage.
  """
  points=sorted({int(r[k]) for r in rows for k in ("start","end")})
  exposed,hidden,upper=Counter(),Counter(),Counter()
  unique=Counter()
  for a,b in zip(points,points[1:]):
    if b <= a: continue
    active={r["class"] for r in rows if int(r["start"]) < b and int(r["end"]) > a}
    non=active-{anchor}; dt=(b-a)/1000.0
    if not non: continue
    target=hidden if anchor in active else exposed
    for role in non: target[role] += dt/len(non); upper[("hidden" if anchor in active else "exposed",role)] += dt
    if len(non) == 1:
      unique[("hidden" if anchor in active else "exposed",next(iter(non)))] += dt
  bounds={role:{"exposed_lower_us":unique[("exposed",role)],"exposed_upper_us":upper[("exposed",role)],
                "hidden_lower_us":unique[("hidden",role)],"hidden_upper_us":upper[("hidden",role)]}
          for role in set(exposed)|set(hidden)}
  return dict(exposed),dict(hidden),bounds


def llama_roles(trace:pathlib.Path, exposed_authority:float, hidden_authority:float, graph_id=2, warmup=2) -> dict:
  con=sqlite3.connect(str(trace))
  try: replays,_=_split_replays(_load_rows(con,graph_id))
  finally: con.close()
  parts=[_shapley_partition(rows) for rows in replays[warmup:]]
  roles=sorted({k for p in parts for side in p[:2] for k in side})
  emed={r:statistics.median(p[0].get(r,0.0) for p in parts) for r in roles}
  hmed={r:statistics.median(p[1].get(r,0.0) for p in parts) for r in roles}
  # Independently medianed Shapley rows are normalized to the already-settled
  # aggregate union authorities; this preserves shares and exact additivity.
  es=exposed_authority/sum(emed.values()); hs=hidden_authority/sum(hmed.values())
  rows=[]
  for role in roles:
    bmed={key:statistics.median(p[2].get(role,{}).get(key,0.0) for p in parts)
          for key in ("exposed_lower_us","exposed_upper_us","hidden_lower_us","hidden_upper_us")}
    rows.append({"role":role,"exposed_shapley_us":emed[role]*es,"hidden_shapley_us":hmed[role]*hs,
                 "median_order_bounds_us":bmed})
  # Correct binary rounding in a deterministic last row.
  rows[-1]["exposed_shapley_us"] += exposed_authority-sum(r["exposed_shapley_us"] for r in rows)
  rows[-1]["hidden_shapley_us"] += hidden_authority-sum(r["hidden_shapley_us"] for r in rows)
  return {"method":"symmetric Shapley interval ownership; median replay shares normalized to aggregate union authorities",
          "replays":len(parts),"roles":rows,"exposed_total_us":sum(r["exposed_shapley_us"] for r in rows),
          "hidden_total_us":sum(r["hidden_shapley_us"] for r in rows)}


def build(dag:pathlib.Path,profile:pathlib.Path,scale:float,trace:pathlib.Path,exposed:float,hidden:float) -> dict:
  native=native_roles(dag,profile,scale); llama=llama_roles(trace,exposed,hidden)
  n=native["roles_us"]; l={r["role"]:r["exposed_shapley_us"] for r in llama["roles"]}
  lh={r["role"]:r["hidden_shapley_us"] for r in llama["roles"]}
  families=[
    ("norms",sum(n.get(k,0) for k in ("initial_rmsnorm","q_norm","k_norm","ffn_rmsnorm","next_or_final_rmsnorm")),
      l.get("rms_norm",0),lh.get("rms_norm",0)),
    ("flash",n.get("flash_score",0)+n.get("flash_combine",0),l.get("flash_score",0)+l.get("flash_combine",0),
      lh.get("flash_score",0)+lh.get("flash_combine",0)),
    ("rope_and_kv_store",n.get("rope_q",0)+n.get("kv_store_k_rope_cast",0)+n.get("kv_store_k_rope_cast_with_q6_partial_reduce",0),
      l.get("rope",0)+l.get("kv_set_rows",0),lh.get("rope",0)+lh.get("kv_set_rows",0)),
    ("residuals_casts_and_contiguous",sum(n.get(k,0) for k in ("attention_cast","attention_residual_add","ffn_activation_cast",
      "ffn_down_cast","ffn_residual_add","block_output_contiguous")),l.get("elementwise",0),lh.get("elementwise",0)),
    ("vocab_sampler_and_token_feedback",n.get("vocab_sampler",0)+n.get("token_feedback",0),l.get("get_rows",0),lh.get("get_rows",0)),
    ("llama_projection_input_quantization",0.0,l.get("quantize_q8_1",0),lh.get("quantize_q8_1",0)),
  ]
  comparison=sorted(({"family":name,"native_us":round(nu,3),"llama_exposed_shapley_us":round(le,3),
                      "llama_hidden_shapley_us":round(lh,3),"llama_total_shapley_us":round(le+lh,3),
                      "fusion_dataflow_and_body_attribution_us":round(nu-le-lh,3),"hidden_overlap_delta_us":round(lh,3),
                      "delta_us":round(nu-le,3)} for name,nu,le,lh in families),key=lambda r:r["delta_us"],reverse=True)
  raw_delta=sum(r["fusion_dataflow_and_body_attribution_us"] for r in comparison)
  hidden_delta=sum(r["hidden_overlap_delta_us"] for r in comparison)
  return {"schema":SCHEMA,"status":"PASS","native":native,"llama":llama,
          "ranked_disjoint_family_deltas":comparison,
          "support_gap_mechanism_split":{"fusion_dataflow_and_body_attribution_us":round(raw_delta,3),
                                         "llama_hidden_overlap_us":round(hidden_delta,3),
                                         "total_nonquantized_delta_us":round(raw_delta+hidden_delta,3)},
          "reconciliation":{"native_authority_us":1408.818,"native_residual_us":round(native["total_us"]-1408.818,3),
                            "llama_exposed_authority_us":exposed,"llama_exposed_residual_us":llama["exposed_total_us"]-exposed},
          "provenance":{"native_dag":{"path":str(dag),"sha256":sha256(dag)},
                        "native_profile":{"path":str(profile),"sha256":sha256(profile)},
                        "llama_trace":{"path":str(trace),"sha256":sha256(trace)}}}


def main():
  p=argparse.ArgumentParser(); p.add_argument("--dag",type=pathlib.Path,required=True);p.add_argument("--profile",type=pathlib.Path,required=True)
  p.add_argument("--scale",type=float,required=True);p.add_argument("--trace",type=pathlib.Path,required=True)
  p.add_argument("--exposed-us",type=float,default=300.736);p.add_argument("--hidden-us",type=float,default=445.954)
  p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args()
  out=build(a.dag,a.profile,a.scale,a.trace,a.exposed_us,a.hidden_us);a.out.parent.mkdir(parents=True,exist_ok=True)
  a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out["reconciliation"],indent=2))

if __name__ == "__main__": main()
