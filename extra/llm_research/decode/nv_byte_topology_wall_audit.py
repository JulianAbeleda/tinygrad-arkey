#!/usr/bin/env python3
"""Join exact GGUF weight bytes, the current device ledger, and a dependency DAG."""
from __future__ import annotations

import argparse, collections, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad.llm.gguf import gguf_load_metadata
from tinygrad.llm.gguf_memory_scan import gguf_tensor_spans


def weight_role(name:str)->str:
  if name=="output.weight":return "vocab_output"
  if name=="token_embd.weight":return "token_embedding_storage"
  if name=="output_norm.weight":return "output_norm"
  for needle,role in ((".attn_q.weight","attn_q"),(".attn_k.weight","attn_k"),(".attn_v.weight","attn_v"),
                      (".attn_output.weight","attn_o"),(".ffn_gate.weight","ffn_gate"),
                      (".ffn_up.weight","ffn_up"),(".ffn_down.weight","ffn_down")):
    if needle in name:return role
  if name.endswith("_norm.weight"):return "norms"
  return "other"


def node_role(name:str)->str:
  if name.startswith("q4k_gate_up"):return "gate_up"
  if "4096_12288" in name:return "ffn_down"
  if "151936_4096" in name:return "vocab_main"
  if "gemv_vec_epi_resadd_4096_4096" in name:return "attn_o"
  if "4096_4096" in name and "q4k_" in name:return "attn_q"
  if "1024_4096" in name:return "attn_kv"
  if "flash_block" in name:return "flash_score"
  if "flash_fused" in name:return "flash_combine"
  if "rope" in name:return "qk_norm_rope_cache"
  if "rmsnorm_q8" in name:return "q8_provider"
  if name.startswith("reduce_output_rmsnorm"):return "norm"
  if name.startswith("r_"):return "reduction_norm"
  if name.startswith("E"):return "elementwise"
  if "argmax" in name:return "argmax"
  return "other"


def dag_accounting(dag:dict)->dict:
  nodes,edges=dag["nodes"],dag["edges"];n=len(nodes)
  preds=[set() for _ in range(n)];succs=[set() for _ in range(n)]
  for edge in edges:preds[edge["to"]].add(edge["from"]);succs[edge["from"]].add(edge["to"])
  indegree=[len(x) for x in preds];ready=collections.deque(i for i,x in enumerate(indegree) if x==0);order=[]
  while ready:
    i=ready.popleft();order.append(i)
    for child in succs[i]:
      indegree[child]-=1
      if indegree[child]==0:ready.append(child)
  if len(order)!=n:raise RuntimeError("DAG contains a cycle")
  duration=[float(x["duration_us"]) for x in nodes];earliest=[0.0]*n
  for i in order:earliest[i]=max((earliest[p]+duration[p] for p in preds[i]),default=0.0)
  critical=max((earliest[i]+duration[i] for i in range(n)),default=0.0);latest_finish=[critical]*n
  for i in reversed(order):
    if succs[i]:latest_finish[i]=min(latest_finish[c]-duration[c] for c in succs[i])
  slack=[latest_finish[i]-duration[i]-earliest[i] for i in range(n)]
  families=collections.defaultdict(lambda:{"nodes":0,"node_us":0.0,"zero_slack_nodes":0,"zero_slack_us":0.0,"off_path_us":0.0})
  for i,node in enumerate(nodes):
    row=families[node_role(node["name"])];row["nodes"]+=1;row["node_us"]+=duration[i]
    if slack[i]<=1e-6:row["zero_slack_nodes"]+=1;row["zero_slack_us"]+=duration[i]
    else:row["off_path_us"]+=duration[i]
  return {"node_count":n,"edge_count":len(edges),"serialized_us":round(sum(duration),3),
    "critical_path_us":round(critical,3),"dependency_overlap_ceiling_us":round(sum(duration)-critical,3),
    "families":{k:{kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items()}
                for k,v in sorted(families.items(),key=lambda x:-x[1]["node_us"])}}


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--model",type=pathlib.Path,required=True);ap.add_argument("--profile",type=pathlib.Path,required=True)
  ap.add_argument("--dag",type=pathlib.Path,required=True);ap.add_argument("--queue-sweep",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args()
  kv,meta=gguf_load_metadata(args.model);spans=gguf_tensor_spans(meta,args.model.stat().st_size)
  roles=collections.defaultdict(lambda:{"tensors":0,"payload_bytes":0,"ggml_types":collections.Counter()})
  for span in spans:
    row=roles[weight_role(span.name)];row["tensors"]+=1;row["payload_bytes"]+=span.payload_bytes or 0;row["ggml_types"][str(span.ggml_type)]+=1
  projection_roles=("vocab_output","attn_q","attn_k","attn_v","attn_o","ffn_gate","ffn_up","ffn_down")
  projection_bytes=sum(roles[x]["payload_bytes"] for x in projection_roles)
  profile=json.loads(args.profile.read_text());rows={x["name"]:x for x in profile["rows"]}
  weight_names=[name for name in rows if node_role(name) in ("gate_up","ffn_down","vocab_main","attn_o","attn_q","attn_kv")]
  weight_kernel_us=sum(rows[name]["wall_us_per_replay"] for name in weight_names)
  union=float(profile["closure"]["union_us"]);observed_rate=projection_bytes/weight_kernel_us/1e3
  q6_bytes=sum(span.payload_bytes or 0 for span in spans if span.ggml_type==14 and weight_role(span.name) in projection_roles)
  payload={"schema":"tinygrad.nv_byte_topology_wall_audit.v1","model":str(args.model),"model_architecture":kv.get("general.architecture"),
    "weight_bytes":{"roles":{k:{"tensors":v["tensors"],"payload_bytes":v["payload_bytes"],"ggml_types":dict(v["ggml_types"])} for k,v in sorted(roles.items())},
      "projection_and_vocab_payload_bytes_per_token":projection_bytes,"q6_projection_payload_bytes":q6_bytes,
      "token_embedding_full_storage_bytes_not_streamed_per_decode":roles["token_embedding_storage"]["payload_bytes"]},
    "device":{"union_us":union,"node_sum_us":profile["closure"]["node_sum_us"],"measured_overlap_us":profile["closure"]["overlap_us"],
      "weight_kernel_us":round(weight_kernel_us,3),"weight_kernel_fraction_of_union":round(weight_kernel_us/union,6),
      "aggregate_weight_rate_GBps":round(observed_rate,3),"one_percent_weight_bytes_us_at_observed_rate":round(0.01*weight_kernel_us,3),
      "one_hundred_MB_us_at_observed_rate":round(100e6/(observed_rate*1e9)*1e6,3)},
    "topology":dag_accounting(json.loads(args.dag.read_text())),"queue_sweep":json.loads(args.queue_sweep.read_text())}
  args.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
