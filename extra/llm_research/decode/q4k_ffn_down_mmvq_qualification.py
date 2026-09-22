#!/usr/bin/env python3
"""Full-token qualification for the closed-default Q4_K FFN-down MMVQ route."""
from __future__ import annotations

import argparse, collections, hashlib, json, os, pathlib, statistics, subprocess, sys, time
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  _semantic_comparison, _settled_continuous_windows, _validate_run_extent)
from tinygrad.llm.q4k_ffn_down_mmvq import Q4KFFNDownMMVQAdmission

Q4_FFN_DOWN_INDICES=(4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29)
ACCEPTED_ATTENTION_MAX17_INDICES=(1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18)


def _normalize(spec:str) -> tuple[int,...]:
  if not spec.strip(): return ()
  values=tuple(int(x) for x in spec.split(","))
  if values != tuple(sorted(set(values))) or any(x not in Q4_FFN_DOWN_INDICES for x in values):
    raise ValueError(f"indices must be an increasing subset of {Q4_FFN_DOWN_INDICES}")
  return values


def _install(model,indices:tuple[int,...],owned_input_boundary:bool=False,fp16_fma:bool=False,
             scalar_q8_packet:bool=False) -> list[int]:
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  for block in model.blk:
    linear=getattr(block,"ffn_down",None)
    if hasattr(linear,"_q4k_ffn_down_mmvq_admission"): delattr(linear,"_q4k_ffn_down_mmvq_admission")
  for index in indices:
    linear=model.blk[index].ffn_down
    if not isinstance(linear,Q4KPrimitiveLinear) or linear.route_role != "ffn_down" or \
       (linear.out_features,linear.in_features) != (4096,12288):
      raise RuntimeError(f"block {index} is not the exact Q4_K FFN-down production role")
    linear._q4k_ffn_down_mmvq_admission=Q4KFFNDownMMVQAdmission(index,
      owned_input_boundary=owned_input_boundary,fp16_fma=fp16_fma,scalar_q8_packet=scalar_q8_packet)
  return list(indices)


def _digest(a:np.ndarray) -> str: return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _exact_topology_delta(control:dict,candidate:dict,lease_count:int,owned:bool=False,fp16:bool=False,
                          scalar:bool=False) -> dict:
  factor=2 if candidate["composed"] else 1
  before,after=collections.Counter(control["program_names"]),collections.Counter(candidate["program_names"])
  changed={name:after[name]-before[name] for name in before.keys()|after.keys() if after[name]!=before[name]}
  expected=({"q4k_g3_lanemap_gemv_w1w3fused16_12288_4096":-lease_count*factor,
    "ffn_w1w3_q8_scalar_packet_12288_4096":lease_count*factor,
    "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288":-lease_count*factor,
    "q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd":lease_count*factor} if scalar else
    {"q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288":-lease_count*factor,
    "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd":lease_count*factor} if fp16 else
    {"q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288":-lease_count*factor,
    "q8_1_llama_provider_12288":lease_count*factor,
    "q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd":lease_count*factor})
  removed_materialize={name:delta for name,delta in changed.items() if name not in expected}
  base_ok=all(changed.get(name)==delta for name,delta in expected.items()) and len(changed)==len(expected)+len(removed_materialize)
  owned_ok=owned and len(removed_materialize)==1 and next(iter(removed_materialize.values()))==-lease_count*factor
  # Both the closed Q8 route and the fp16 geometry route must add zero extra
  # transport/materialize nodes; any entry in removed_materialize fails closed.
  closed_ok=not owned and not removed_materialize
  pass_=base_ok and (owned_ok or closed_ok)
  return {"changed_program_counts":changed,"expected_changed_program_counts":expected,
    "removed_materialize_program":removed_materialize,"no_adapter_or_other_program_delta":pass_,"pass":pass_,
    "fp16_fma":fp16,"scalar_q8_packet":scalar}


def _install_accepted_attention_max17(model,enabled:bool) -> list[int]:
  if not enabled: return []
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _install_fused_norm_lease_indices
  if not CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT:
    raise RuntimeError("accepted attention max17 composition requires CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1")
  return _install_fused_norm_lease_indices(model,ACCEPTED_ATTENTION_MAX17_INDICES,cooperative_q4=True,q6_direct_output=False)


def child(model_path:str,depth:int,count:int,max_context:int,indices:tuple[int,...],composed:bool=False,compact:bool=False,
          accepted_attention_max17:bool=False,owned_input_boundary:bool=False,fp16_fma:bool=False,
          scalar_q8_packet:bool=False):
  from tinygrad import Tensor,UOp
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  from tinygrad.engine.jit import GraphAdmissionCensus,observe_graph_admissions
  from tinygrad.helpers import Context
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  model=_load(model_path,max_context); leases=_install(model,indices,owned_input_boundary,fp16_fma,scalar_q8_packet)
  attention_leases=_install_accepted_attention_max17(model,accepted_attention_max17)
  model._decode_direct_greedy_promoted=composed; model._decode_feedback_pingpong_promoted=composed
  gen=model.generate(_prompt(model_path,depth),chunk_size=32,temperature=0.0)
  try: prelude=int(next(gen))
  finally: gen.close()
  token,temp=Tensor([[1]],dtype="int32").contiguous(),Tensor([0.0])
  start_pos=UOp.variable("start_pos",0,max_context-1)
  with Context(JIT=0): _,eager=model.forward_with_logits(token,start_pos.bind(depth),temp)
  if not np.isfinite(eager.numpy()).all(): raise RuntimeError("eager full logits non-finite")
  censuses=[GraphAdmissionCensus() for _ in range(2 if composed else 1)]
  logits,tokens=[],[]
  for i in range(count):
    with observe_graph_admissions(censuses[i&1 if composed else 0]):
      sampled,full=model.decode_with_logits(token,start_pos.bind(depth+1+i),temp,feedback_slot=(i&1) if composed else None)
      sampled_id,full_np=int(sampled.item()),full.numpy()
      if sampled_id != int(full_np.reshape(-1).argmax()): raise RuntimeError(f"sample/logit binding mismatch at {i}")
    tokens.append(sampled_id);logits.append(full_np);token=sampled
  arr=np.stack(logits)
  if not np.isfinite(arr).all(): raise RuntimeError("decode full logits non-finite")
  programs=[r.program_name for census in censuses for r in census.records if r.program_name]
  factor=2 if composed else 1
  if scalar_q8_packet:
    # The folded producer replaces the promoted fused16 W1/W3 kernel on every
    # leased Q4-down block, but fused16 also feeds the Q6-down blocks' gate/up
    # projection, so its baseline population is wider than Q4_FFN_DOWN_INDICES.
    fused_gateup_blocks=sum(1 for block in model.blk
      if isinstance(getattr(block,"ffn_gate",None),Q4KPrimitiveLinear)
      and isinstance(getattr(block,"ffn_up",None),Q4KPrimitiveLinear))
    provider_count=programs.count("q8_1_llama_provider_12288")
    producer_count=programs.count("ffn_w1w3_q8_scalar_packet_12288_4096")
    consumer_count=programs.count("q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd")
    installed_producer_count=programs.count("q4k_g3_lanemap_gemv_w1w3fused16_12288_4096")
    installed_count=programs.count("q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288")
    topology={"provider_count":provider_count,"producer_count":producer_count,"consumer_count":consumer_count,
      "installed_w1w3_fused16_count":installed_producer_count,"installed_q4_ffn_down_count":installed_count,
      "expected_provider_count":0,"expected_producer_count":len(indices)*factor,"expected_consumer_count":len(indices)*factor,
      "expected_installed_w1w3_fused16_count":(fused_gateup_blocks-len(indices))*factor,
      "expected_installed_count":(len(Q4_FFN_DOWN_INDICES)-len(indices))*factor,
      "adapter_kernel_count":0,"pass":provider_count==0 and producer_count==consumer_count==len(indices)*factor and
        installed_producer_count==(fused_gateup_blocks-len(indices))*factor and
        installed_count==(len(Q4_FFN_DOWN_INDICES)-len(indices))*factor}
  elif fp16_fma:
    provider_count=programs.count("q8_1_llama_provider_12288")
    consumer_count=programs.count("q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd")
    installed_count=programs.count("q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288")
    topology={"provider_count":provider_count,"consumer_count":consumer_count,"installed_q4_ffn_down_count":installed_count,
      "expected_provider_count":0,"expected_consumer_count":len(indices)*factor,
      "expected_installed_count":(len(Q4_FFN_DOWN_INDICES)-len(indices))*factor,
      "adapter_kernel_count":0,"pass":provider_count==0 and consumer_count==len(indices)*factor and
        installed_count==(len(Q4_FFN_DOWN_INDICES)-len(indices))*factor}
  else:
    provider_count=programs.count("q8_1_llama_provider_12288")
    consumer_count=programs.count("q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd")
    installed_count=programs.count("q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288")
    topology={"provider_count":provider_count,"consumer_count":consumer_count,"installed_q4_ffn_down_count":installed_count,
      "expected_provider_count":len(indices)*factor,"expected_consumer_count":len(indices)*factor,
      "expected_installed_count":(len(Q4_FFN_DOWN_INDICES)-len(indices))*factor,
      "adapter_kernel_count":0,"pass":provider_count==consumer_count==len(indices)*factor and
        installed_count==(len(Q4_FFN_DOWN_INDICES)-len(indices))*factor}
  if not topology["pass"]: raise RuntimeError(f"production topology failed: {topology}")
  attention_topology={"enabled":accepted_attention_max17,"indices":attention_leases,
    "fused_provider_count":programs.count("rmsnorm_q8_1_llama_provider_4096"),
    "cooperative_q4_consumer_count":sum(p.startswith("q4k_warp_coop_q8_dp4a_partial_") for p in programs),
    "legacy_q4_consumer_count":sum(p.startswith("q4k_q8_dp4a_") for p in programs)}
  attention_topology["pass"] = not accepted_attention_max17 or (
    attention_topology["fused_provider_count"]==34 and attention_topology["cooperative_q4_consumer_count"]==86 and
    attention_topology["legacy_q4_consumer_count"]==0)
  if not attention_topology["pass"]: raise RuntimeError(f"accepted attention max17 topology failed: {attention_topology}")
  row={"schema":"tinygrad.q4k_ffn_down_mmvq_qualification.v1","indices":leases,"composed":composed,
    "owned_input_boundary":owned_input_boundary,
    "fp16_fma":fp16_fma,
    "scalar_q8_packet":scalar_q8_packet,
    "callify_owned_precompiled_output_redirect":int(bool(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT)),
    "accepted_attention_max17":accepted_attention_max17,"attention_topology":attention_topology,
    "depth":depth,"count":count,"prelude_token":prelude,"tokens":tokens,
    "tokens_sha256":hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest(),"logits_sha256":_digest(arr),
    "shape":list(arr.shape),"finite":True,"program_count":len(programs),"topology":topology,
    "program_names":programs,"graph_census":([{"counts":x.to_dict()["counts"],"constructor_failures":x.to_dict()["constructor_failures"]}
      for x in censuses] if compact else [x.to_dict() for x in censuses])}
  return row,arr


def timing_child(model_path:str,depth:int,count:int,max_context:int,reps:int,indices:tuple[int,...],composed:bool,
                 accepted_attention_max17:bool=False,owned_input_boundary:bool=False,fp16_fma:bool=False,
                 scalar_q8_packet:bool=False):
  from tinygrad import Device
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  model=_load(model_path,max_context);leases=_install(model,indices,owned_input_boundary,fp16_fma,scalar_q8_packet)
  attention_leases=_install_accepted_attention_max17(model,accepted_attention_max17)
  model._decode_direct_greedy_promoted=composed;model._decode_feedback_pingpong_promoted=composed
  gen=model.generate(_prompt(model_path,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally: gen.close()
  return {"schema":"tinygrad.q4k_ffn_down_mmvq_timing.v1","indices":leases,"composed":composed,
    "owned_input_boundary":owned_input_boundary,
    "fp16_fma":fp16_fma,
    "scalar_q8_packet":scalar_q8_packet,
    "callify_owned_precompiled_output_redirect":int(bool(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT)),
    "accepted_attention_max17":accepted_attention_max17,"attention_indices":attention_leases,
    "included_cost":True,"settled_continuous":True,"warmup_decode_calls":6,"reps":reps,"tokens_per_rep":count,**settled}


def _cmd(args,mode,indices,out,accepted_attention_max17:bool|None=None,owned:bool=False,fp16:bool=False,
         scalar:bool=False):
  cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),"--mode",mode,"--model",args.model,"--depth",str(args.depth),
    "--count",str(args.count),"--max-context",str(args.max_context),"--reps",str(args.reps),"--indices",",".join(map(str,indices)),"--out",str(out)]
  if args.composed: cmd.append("--composed")
  if args.accepted_attention_max17 if accepted_attention_max17 is None else accepted_attention_max17:
    cmd.append("--accepted-attention-max17")
  if owned or args.owned_input_boundary: cmd.append("--owned-input-boundary")
  if fp16 or args.fp16_fma: cmd.append("--fp16-fma")
  if scalar or args.scalar_q8_packet: cmd.append("--scalar-q8-packet")
  cmd.append("--compact")
  return cmd


def qualify(args):
  root=pathlib.Path(args.out).with_suffix("");root.mkdir(parents=True,exist_ok=True)
  rows,arrays={},{}
  arms=(("baseline",(),False),("control",(),True),("candidate",args.indices,True)) if args.accepted_attention_max17 else \
    (("control",(),False),("candidate",args.indices,False))
  for label,indices,attention in arms:
    out=root/f"{label}.json"
    run=subprocess.run(["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
      *_cmd(args,"child",indices,out,attention,scalar=args.scalar_q8_packet)],
      text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
    rows[label]=json.loads(out.read_text());arrays[label]=np.load(out.with_suffix(".npz"))["logits"]
  comparison=_semantic_comparison(arrays["control"],arrays["candidate"],rows["control"],rows["candidate"])
  comparison["exact_topology_delta"]=_exact_topology_delta(rows["control"],rows["candidate"],len(args.indices),
    owned=args.owned_input_boundary,fp16=args.fp16_fma,scalar=args.scalar_q8_packet)
  comparison["topology_pass"]=rows["candidate"]["topology"]["pass"] and comparison["exact_topology_delta"]["pass"]
  global_comparison=_semantic_comparison(arrays["baseline"],arrays["candidate"],rows["baseline"],rows["candidate"]) \
    if args.accepted_attention_max17 else comparison
  attention_comparison=_semantic_comparison(arrays["baseline"],arrays["control"],rows["baseline"],rows["control"]) \
    if args.accepted_attention_max17 else None
  comparison["gate_pass"]=global_comparison["semantic_pass"] and comparison["topology_pass"]
  return {"schema":"tinygrad.q4k_ffn_down_mmvq_qualification.v1","mode":"qualify","indices":list(args.indices),
    "accepted_attention_max17":args.accepted_attention_max17,"children":rows,"comparison":comparison,
    "fp16_fma":args.fp16_fma,
    "scalar_q8_packet":args.scalar_q8_packet,
    "incremental_comparison":comparison,
    "global_candidate_vs_unleased":global_comparison,"attention_vs_unleased":attention_comparison,
    "verdict":"PASS" if comparison["gate_pass"] else "FAIL_CLOSED"}


def timing(args):
  root=pathlib.Path(args.out).with_suffix("");root.mkdir(parents=True,exist_ok=True);rows=[]
  for seq,indices in enumerate(((),args.indices,())):
    label=f"{'control' if not indices else 'candidate'}-{seq}";out=root/f"{label}.json"
    run=subprocess.run(["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
      *_cmd(args,"timing-child",indices,out,fp16=args.fp16_fma,scalar=args.scalar_q8_packet)],
      text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
    rows.append(json.loads(out.read_text()))
  control=statistics.median((rows[0]["median_ms_per_token"],rows[2]["median_ms_per_token"]));candidate=rows[1]["median_ms_per_token"]
  hashes={x["token_stream_hash"] for x in rows}
  return {"schema":"tinygrad.q4k_ffn_down_mmvq_timing.v1","mode":"reverse-bracket","indices":list(args.indices),
    "fp16_fma":args.fp16_fma,
    "scalar_q8_packet":args.scalar_q8_packet,
    "arms":rows,"all_token_hashes_equal":len(hashes)==1,"control_bracket_median_ms":control,"candidate_ms":candidate,
    "candidate_minus_control_ms":candidate-control,"candidate_speedup_pct":(control/candidate-1)*100,
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate<control else "NO_GO_WALL"}


def sweep(args):
  """One shared control, 18 one-layer children, then signed additive prediction."""
  root=pathlib.Path(args.out).with_suffix("");root.mkdir(parents=True,exist_ok=True)
  control_out=root/"control.json"
  def valid_existing(out:pathlib.Path,indices:tuple[int,...]) -> bool:
    if not args.resume_existing or not out.is_file() or not out.with_suffix(".npz").is_file(): return False
    try: row=json.loads(out.read_text())
    except (OSError,json.JSONDecodeError): return False
    return row.get("schema")=="tinygrad.q4k_ffn_down_mmvq_qualification.v1" and tuple(row.get("indices",()))==indices and \
      row.get("depth")==args.depth and row.get("count")==args.count and row.get("composed")==args.composed and row.get("finite") is True
  if not valid_existing(control_out,()):
    run=subprocess.run(["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
      *_cmd(args,"child",(),control_out,fp16=args.fp16_fma,scalar=args.scalar_q8_packet)],
      text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"control failed rc={run.returncode}: {run.stderr[-5000:]}")
  control_row=json.loads(control_out.read_text());control=np.load(control_out.with_suffix(".npz"))["logits"]
  layer_rows={};deltas=[]
  for index in Q4_FFN_DOWN_INDICES:
    out=root/f"layer-{index}.json"
    if not valid_existing(out,(index,)):
      run=subprocess.run(["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
        *_cmd(args,"child",(index,),out,fp16=args.fp16_fma,scalar=args.scalar_q8_packet)],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      if run.returncode: raise RuntimeError(f"layer {index} failed rc={run.returncode}: {run.stderr[-5000:]}")
    row=json.loads(out.read_text());candidate=np.load(out.with_suffix(".npz"))["logits"]
    comparison=_semantic_comparison(control,candidate,control_row,row)
    exact_topology=_exact_topology_delta(control_row,row,1,owned=args.owned_input_boundary,fp16=args.fp16_fma,
      scalar=args.scalar_q8_packet)
    comparison.update({"index":index,"exact_topology_delta":exact_topology,
      "topology_pass":row["topology"]["pass"] and exact_topology["pass"]})
    layer_rows[str(index)]=comparison;deltas.append((candidate-control).astype(np.float32))
  # The Gram matrix preserves signed cancellation between layers. Enumerate
  # every 2^18 subset cheaply in scalar space, then materialize only the best
  # predicted subset at each cardinality for the full semantic contract.
  flat=np.stack([d.reshape(-1) for d in deltas]);gram=(flat@flat.T).astype(np.float64)
  base_norm2=float(np.square(control.astype(np.float64)).sum());size=1<<len(Q4_FFN_DOWN_INDICES)
  norm2=np.zeros(size,dtype=np.float64);best=[None]*(len(Q4_FFN_DOWN_INDICES)+1)
  for mask in range(1,size):
    bit=mask&-mask;i=bit.bit_length()-1;prev=mask^bit
    js=[j for j in range(i) if prev&(1<<j)] + [j for j in range(i+1,len(Q4_FFN_DOWN_INDICES)) if prev&(1<<j)]
    norm2[mask]=norm2[prev]+gram[i,i]+2*sum(gram[i,j] for j in js)
    count=mask.bit_count()
    if best[count] is None or norm2[mask] < norm2[best[count]]: best[count]=mask
  predictions=[]
  for count in range(1,len(best)):
    mask=best[count];selected=[Q4_FFN_DOWN_INDICES[i] for i in range(len(Q4_FFN_DOWN_INDICES)) if mask&(1<<i)]
    predicted=control+sum((deltas[i] for i in range(len(deltas)) if mask&(1<<i)),np.zeros_like(control))
    comparison=_semantic_comparison(control,predicted,control_row,control_row)
    comparison.update({"indices":selected,"cardinality":count,"mask":mask,
      "gram_relative_l2":float(np.sqrt(max(norm2[mask],0.)/max(base_norm2,1e-30)))})
    predictions.append(comparison)
  ranked=sorted((dict(v) for v in layer_rows.values()),key=lambda x:(x["relative_l2"],x["max_abs"]))
  promising=sorted((x for x in predictions if x["semantic_pass"]),key=lambda x:(-x["cardinality"],x["relative_l2"]))
  return {"schema":"tinygrad.q4k_ffn_down_mmvq_subset_search.v1","mode":"sweep-and-additive-predict",
    "indices":list(Q4_FFN_DOWN_INDICES),"control":{"tokens":control_row["tokens"],"logits_sha256":control_row["logits_sha256"]},
    "per_layer":layer_rows,"ranked_per_layer":ranked,"best_by_cardinality":predictions,
    "promising_predicted_subsets":promising,"next_real_validation":[x["indices"] for x in promising[:3]],
    "non_claim":"additive predictions are direction only; only a fresh full-token subset child can qualify"}


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("child","qualify","timing-child","timing","sweep"),required=True)
  ap.add_argument("--model",default=os.environ.get("QK_MODEL","/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"));ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--count",type=int,default=8);ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--reps",type=int,default=3)
  ap.add_argument("--indices",default="");ap.add_argument("--composed",action="store_true");ap.add_argument("--compact",action="store_true")
  ap.add_argument("--accepted-attention-max17",action="store_true",
    help="compose on the accepted blocks 1-12,14-18 cooperative shared-Q8 route; requires redirect=1")
  ap.add_argument("--owned-input-boundary",action="store_true",
    help="lease the Q4 FFN-down MMVQ with the owned fp32 input boundary (provider consumes fp32; no fp16 materialize)")
  ap.add_argument("--fp16-fma",action="store_true",
    help="lease the Q4 FFN-down four-warp fp16-FMA geometry consumer (no Q8 provider node)")
  ap.add_argument("--scalar-q8-packet",action="store_true",
    help="lease the folded scalar-packet W1/W3 Q8 producer plus four-warp Q4/Q8 resadd consumer")
  ap.add_argument("--resume-existing",action="store_true",help="sweep only: reuse exact-matching complete control/layer artifacts")
  ap.add_argument("--out",required=True)
  ap.add_argument("--timeout",type=int,default=900);ap.add_argument("--lock-wait",type=int,default=120);ap.add_argument("--lock",default="/tmp/gpu-bench.lock")
  args=ap.parse_args();args.indices=_normalize(args.indices);_validate_run_extent(args.depth,args.count,args.max_context,args.reps,args.mode in ("timing","timing-child"))
  if args.mode=="child":
    row,arr=child(args.model,args.depth,args.count,args.max_context,args.indices,args.composed,args.compact,
      args.accepted_attention_max17,args.owned_input_boundary,args.fp16_fma,args.scalar_q8_packet);out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix(".npz").open("wb") as f: np.savez_compressed(f,logits=arr)
    out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":"complete","out":str(out)}));os._exit(0)
  if args.mode=="timing-child":
    row=timing_child(args.model,args.depth,args.count,args.max_context,args.reps,args.indices,args.composed,
      args.accepted_attention_max17,args.owned_input_boundary,args.fp16_fma,args.scalar_q8_packet);out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");print(json.dumps(row));os._exit(0)
  result=qualify(args) if args.mode=="qualify" else sweep(args) if args.mode=="sweep" else timing(args)
  pathlib.Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__": main()
