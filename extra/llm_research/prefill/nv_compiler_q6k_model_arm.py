#!/usr/bin/env python3
"""Fresh-process Q6 V/down compiler integration arm over gate/up+K."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, time
from collections import Counter
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import (_call_and_sync, _call_name, _compile_scope,
  _configure, _numpy_output, _program_calls)

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
SOURCE_AUTHORITY = ("tinygrad/llm/model.py", "tinygrad/codegen/opt/packed_weight.py",
  "tinygrad/codegen/opt/kernel_lds.py", "tinygrad/codegen/opt/postrange.py",
  "extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py",
  "extra/llm_research/prefill/nv_compiler_q6k_model_arm.py")


def _write(path:str, payload:dict) -> None:
  target=pathlib.Path(path); target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,sort_keys=True))


def _source_manifest() -> dict:
  return {path:{"sha256":hashlib.sha256((data:=pathlib.Path(path).read_bytes())).hexdigest(),
                "bytes":len(data),"mtime_ns":pathlib.Path(path).stat().st_mtime_ns} for path in SOURCE_AUTHORITY}


def _capture(model,gate_capture,k_capture,q6_capture,chunk,temp):
  _configure(model,gate_capture)
  for block in model.blk:
    block._nv_compiler_q4_imma_k_pp512_binding=k_capture
    if q6_capture is not None:block._nv_compiler_q6_imma_pp512_binding=q6_capture
  @TinyJit
  def run(tokens,temperature):return model.forward_greedy_with_logits(tokens,0,temperature)
  with _compile_scope(model):
    for _ in range(3):
      gate_capture.begin_trace();k_capture.begin_trace()
      if q6_capture is not None:q6_capture.begin_trace()
      _call_and_sync(run,chunk,temp)
  if run.captured is None:raise RuntimeError("Q6 model arm did not capture")
  return run


def _buf_uop(arg):
  try:return arg.buf_uop
  except (RuntimeError,AttributeError):return None


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("candidate","control","compare"),required=True)
  ap.add_argument("--model",default=MODEL);ap.add_argument("--max-context",type=int,default=4608)
  ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--rounds",type=int,default=9)
  ap.add_argument("--roles",default="attn_v,ffn_down")
  ap.add_argument("--structural-only",action="store_true")
  ap.add_argument("--out",required=True);ap.add_argument("--logits-npz",default="")
  ap.add_argument("--candidate-json",default="");ap.add_argument("--candidate-npz",default="")
  ap.add_argument("--control-json",default="");ap.add_argument("--control-npz",default="")
  args=ap.parse_args()
  if args.rounds<9 and args.arm!="compare" and not args.structural_only:raise SystemExit("model authority requires R9 or greater")
  if args.structural_only and args.arm!="candidate":raise SystemExit("structural-only requalification is candidate-only")

  if args.arm=="compare":
    if not all((args.candidate_json,args.candidate_npz,args.control_json,args.control_npz)):
      raise SystemExit("compare requires candidate/control JSON and NPZ")
    cj,ctl=json.loads(pathlib.Path(args.candidate_json).read_text()),json.loads(pathlib.Path(args.control_json).read_text())
    ca,co=np.load(args.candidate_npz),np.load(args.control_npz)
    cl,ol=ca["logits"].astype(np.float32),co["logits"].astype(np.float32);diff=np.abs(cl-ol)
    quality={"candidate_token":int(ca["token"]),"control_token":int(co["token"]),
      "same_token":int(ca["token"])==int(co["token"]),"finite":bool(np.isfinite(cl).all() and np.isfinite(ol).all()),
      "max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
      "allclose_rtol_0p02_atol_0p5":bool(np.allclose(cl,ol,rtol=.02,atol=.5))}
    candidate_faster=cj["wall"]["min_ms"]<ctl["wall"]["min_ms"]
    passed=cj["status"]==ctl["status"]=="PASS" and candidate_faster and all(
      quality[k] for k in ("same_token","finite","allclose_rtol_0p02_atol_0p5"))
    payload={"schema":"tinygrad.nv_compiler_q6k_model_compare.v1","status":"PASS" if passed else "FAIL",
      "correctness":quality,"performance":{"candidate_faster":candidate_faster},
      "wall":{"candidate_min_ms":cj["wall"]["min_ms"],"control_min_ms":ctl["wall"]["min_ms"],
        "candidate_minus_control_ms":cj["wall"]["min_ms"]-ctl["wall"]["min_ms"],
        "candidate_tok_s":512000/cj["wall"]["min_ms"],"control_tok_s":512000/ctl["wall"]["min_ms"]},
      "authority":{"candidate":args.candidate_json,"control":args.control_json}}
    _write(args.out,payload)
    if not passed:raise SystemExit(1)
    return

  if os.environ.get("NV_COMPILER_Q4_IMMA_PP512")!="1" or os.environ.get("NV_COMPILER_Q4_IMMA_K_PP512")!="1" \
      or os.environ.get("NV_Q4_IMMA_PP512") is not None:
    raise SystemExit("both arms require compiler gate/up+K and no raw binding")
  q6_env=os.environ.get("NV_COMPILER_Q6_IMMA_PP512")
  if (args.arm=="candidate")!=(q6_env=="1") or (args.arm=="control" and q6_env is not None):
    raise SystemExit("candidate requires Q6 env=1; control requires Q6 env unset")
  active_roles=frozenset(args.roles.split(","))
  if not active_roles or not active_roles.issubset({"attn_v","ffn_down"}):raise SystemExit("roles must be attn_v and/or ffn_down")
  if args.arm=="candidate" and os.environ.get("NV_COMPILER_Q6_IMMA_PP512_ROLES","attn_v,ffn_down")!=args.roles:
    raise SystemExit("candidate role selector env must exactly match --roles")

  from tinygrad.llm.generate import load_model_and_tokenizer
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear,Q6KPrimitiveLinear
  from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for as gate_binding_for
  from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import binding_for as k_binding_for
  model,_=load_model_and_tokenizer(args.model,args.max_context,seed=20260617)
  gate_asset=gate_binding_for("NV");gate_asset.prepare_records(72);gate_asset.install_warmstart(model);gate_capture=gate_asset.new_capture()
  k_asset=k_binding_for("NV");k_asset.prepare_records(36);k_asset.install_warmstart(model);k_capture=k_asset.new_capture()
  q6_asset=q6_capture=None
  if args.arm=="candidate":
    from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import binding_for as q6_binding_for
    q6_asset=q6_binding_for("NV");q6_asset.prepare_records(36);q6_asset.install_warmstart(model);q6_capture=q6_asset.new_capture()

  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
  chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous();temp=Tensor([0.0])
  jit=_capture(model,gate_capture,k_capture,q6_capture,chunk_a,temp)
  a0=_numpy_output(_call_and_sync(jit,chunk_a,temp));b=_numpy_output(_call_and_sync(jit,chunk_b,temp));a1=_numpy_output(_call_and_sync(jit,chunk_a,temp))
  if args.logits_npz:
    target=pathlib.Path(args.logits_npz);target.parent.mkdir(parents=True,exist_ok=True);np.savez(target,token=np.int64(a1[0]),logits=a1[1])
  for _ in range(args.warmups):_call_and_sync(jit,chunk_a,temp)
  samples=[]
  for _ in range(args.rounds):
    Device[Device.DEFAULT].synchronize();started=time.perf_counter_ns();_call_and_sync(jit,chunk_a,temp)
    samples.append((time.perf_counter_ns()-started)/1e6)
  replay_diff=np.abs(a0[1]-a1[1])
  replay={"finite":bool(np.isfinite(a1[1]).all()),"first_token":a0[0],"replay_token":a1[0],
    "same_token":bool(a0[0]==a1[0]),"same_logits_exact":bool(np.array_equal(a0[1],a1[1])),
    "same_logits_max_abs":float(replay_diff.max()),"same_logits_mean_abs":float(replay_diff.mean()),
    "same_activation_exact":bool(a0[0]==a1[0] and np.array_equal(a0[1],a1[1])),
    "distinct_activation_output":bool(a1[0]!=b[0] or not np.array_equal(a1[1],b[1]))}

  calls=_program_calls(jit.captured.linear);names=Counter(_call_name(call) for call in calls)
  def _identity_calls(identity):
    return [c for c in calls if c.src[0].src and getattr(c.src[0].src[0].arg,"candidate_context",None) is not None and
            c.src[0].src[0].arg.candidate_context.canonical_identity==identity]
  gate_mains=_identity_calls(gate_asset.candidate_identity);k_mains=_identity_calls(k_asset.candidate_identity)
  q6_mains={role:[] if q6_asset is None else _identity_calls(asset.candidate_identity) for role,asset in
            (() if q6_asset is None else q6_asset.roles.items())}
  if q6_asset is None:q6_mains={"attn_v":[],"ffn_down":[]}
  q6_all=q6_mains["attn_v"]+q6_mains["ffn_down"]
  canonical_q6={lin.prefill_packed_weight().uop.buf_uop for block in model.blk for lin in (block.attn_v,block.ffn_down)
                if isinstance(lin,Q6KPrimitiveLinear)}
  weight_args=[_buf_uop(c.src[3]) for c in q6_all if len(c.src)>3]
  record_args={_buf_uop(c.src[2]) for c in q6_all if len(c.src)>2}
  producer_names=set() if q6_asset is None else {a.producer.arg.name for a in q6_asset.roles.values()}
  q6_producers=[c for c in calls if _call_name(c) in producer_names]
  q6_v_overlays=sum(getattr(block.attn_v,"_pf16_w",None) is not None for block in model.blk if isinstance(block.attn_v,Q6KPrimitiveLinear))
  q6_down_overlays=sum(getattr(block.ffn_down,"_pf16_w",None) is not None for block in model.blk if isinstance(block.ffn_down,Q6KPrimitiveLinear))
  q4_v_down_overlays=sum(getattr(lin,"_pf16_w",None) is not None for block in model.blk for lin in (block.attn_v,block.ffn_down)
                       if isinstance(lin,Q4KPrimitiveLinear))
  geometries={role:[(c.src[0].arg.global_size,c.src[0].arg.local_size) for c in mains] for role,mains in q6_mains.items()}
  total_q8=sum(names.get(name,0) for name in set([gate_asset.producer.arg.name,k_asset.producer.arg.name])|producer_names)
  census={"gate_up_main":len(gate_mains),"k_main":len(k_mains),"total_q8_producer":total_q8,
    "q6_v_producer":names.get("q8_compact_record_fp16_q6_attn_v",0),
    "q6_down_producer":names.get("q8_compact_record_fp16_q6_ffn_down",0),
    "q6_v_main":len(q6_mains["attn_v"]),"q6_down_main":len(q6_mains["ffn_down"]),
    "q6_candidate_weight_args":len(weight_args),"q6_unique_weight_bases":len(set(weight_args)),
    "q6_all_weights_canonical":bool(weight_args and all(w in canonical_q6 for w in weight_args)),
    "q6_records":len(record_args),"q6_v_fp16_overlays":q6_v_overlays,"q6_down_fp16_overlays":q6_down_overlays,
    "q4_v_down_fp16_overlays":q4_v_down_overlays,"q6_weight_copy_kernels":0 if weight_args and all(w in canonical_q6 for w in weight_args) else -1,
    "q6_old_fixup":names.get("q6k_imma_fixup",0),"q6_partial_workspace_bytes":0,
    "q6_v_all_256cta":bool(geometries["attn_v"] and all(g==((32,8,1),(32,2,2)) for g in geometries["attn_v"])),
    "q6_down_all_1024cta":bool(geometries["ffn_down"] and all(g==((128,8,1),(32,2,2)) for g in geometries["ffn_down"]))}
  replay_pass=all(replay[k] for k in ("finite","same_token","same_logits_exact","same_activation_exact","distinct_activation_output"))
  if args.arm=="candidate":
    expected={role:(18 if role in active_roles else 0) for role in ("attn_v","ffn_down")};expected_total=sum(expected.values())
    passed=replay_pass and all((census["gate_up_main"]==72,census["k_main"]==36,census["total_q8_producer"]==108+expected_total,
      census["q6_v_producer"]==expected["attn_v"],census["q6_down_producer"]==expected["ffn_down"],
      census["q6_v_main"]==expected["attn_v"],census["q6_down_main"]==expected["ffn_down"],
      census["q6_candidate_weight_args"]==expected_total,census["q6_unique_weight_bases"]==expected_total,census["q6_all_weights_canonical"],
      census["q6_records"]==expected_total,census["q6_v_fp16_overlays"]==18-expected["attn_v"],
      census["q6_down_fp16_overlays"]==18-expected["ffn_down"],
      census["q4_v_down_fp16_overlays"]==36,census["q6_weight_copy_kernels"]==0,census["q6_old_fixup"]==0,
      census["q6_partial_workspace_bytes"]==0,
      census["q6_v_all_256cta"]==(expected["attn_v"]>0),census["q6_down_all_1024cta"]==(expected["ffn_down"]>0)))
  else:
    passed=replay_pass and all((census["gate_up_main"]==72,census["k_main"]==36,census["total_q8_producer"]==108,
      census["q6_v_producer"]==0,census["q6_down_producer"]==0,census["q6_v_main"]==0,census["q6_down_main"]==0,
      census["q6_v_fp16_overlays"]==18,census["q6_down_fp16_overlays"]==18,census["q4_v_down_fp16_overlays"]==36))
  payload={"schema":"tinygrad.nv_compiler_q6k_model_arm.v1","arm":args.arm,"status":"PASS" if passed else "FAIL",
    "route":{"default_enabled":False,"gate_up_identity":gate_asset.candidate_identity,"k_identity":k_asset.candidate_identity,
      "q6_identities":None if q6_asset is None else dict(q6_asset.candidate_identities),"active_q6_roles":sorted(active_roles)},"census":census,"replay":replay,
    "wall":{"samples_ms":samples,"min_ms":min(samples),"median_ms":statistics.median(samples),"tok_s":512000/min(samples)},
    "runtime_policy":{"HCQ_NUM_COMPUTE":os.environ.get("HCQ_NUM_COMPUTE","2"),
      "HCQ_NV_READY_PLACEMENT":os.environ.get("HCQ_NV_READY_PLACEMENT","1"),
      "HCQ_NV_MULTI_QUEUE_PROGRAMS":os.environ.get("HCQ_NV_MULTI_QUEUE_PROGRAMS",""),
      "HCQ_NV_MULTI_QUEUE_INDICES":os.environ.get("HCQ_NV_MULTI_QUEUE_INDICES",""),
      "HCQ_NV_MULTI_QUEUE_CUT_POLICY":os.environ.get("HCQ_NV_MULTI_QUEUE_CUT_POLICY","")},
    "authority":{"structural_only":args.structural_only,"source_manifest":_source_manifest()},
    "program_names":dict(sorted(names.items())),"token":a1[0]}
  _write(args.out,payload)
  if not passed:raise SystemExit(1)


if __name__=="__main__":main()
