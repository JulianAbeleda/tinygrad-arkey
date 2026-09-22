#!/usr/bin/env python3
"""Fresh-process K-only compiler integration arm over the gate/up baseline."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, time
from collections import Counter
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import (_call_and_sync, _call_name, _compile_scope,
  _configure, _numpy_output, _program_calls)

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _write(path:str,payload:dict) -> None:
  target=pathlib.Path(path);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps(payload,sort_keys=True))


def _capture(model,gate_capture,k_capture,chunk,temp):
  _configure(model,gate_capture)
  if k_capture is not None:
    for block in model.blk: block._nv_compiler_q4_imma_k_pp512_binding = k_capture
  @TinyJit
  def run(tokens,temperature): return model.forward_greedy_with_logits(tokens,0,temperature)
  with _compile_scope(model):
    for _ in range(3):
      gate_capture.begin_trace()
      if k_capture is not None:k_capture.begin_trace()
      _call_and_sync(run,chunk,temp)
  if run.captured is None:raise RuntimeError("K model arm did not capture")
  return run


def _buf_uop(arg):
  try:return arg.buf_uop
  except (RuntimeError,AttributeError):return None


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("candidate","control","compare"),required=True)
  ap.add_argument("--model",default=MODEL);ap.add_argument("--max-context",type=int,default=4608)
  ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--rounds",type=int,default=9)
  ap.add_argument("--out",required=True);ap.add_argument("--logits-npz",default="")
  ap.add_argument("--candidate-json",default="");ap.add_argument("--candidate-npz",default="")
  ap.add_argument("--control-json",default="");ap.add_argument("--control-npz",default="")
  args=ap.parse_args()
  if args.rounds < 9 and args.arm != "compare":raise SystemExit("model authority requires R9 or greater")

  if args.arm == "compare":
    if not all((args.candidate_json,args.candidate_npz,args.control_json,args.control_npz)):
      raise SystemExit("compare requires candidate/control JSON and NPZ")
    cj,ctl=json.loads(pathlib.Path(args.candidate_json).read_text()),json.loads(pathlib.Path(args.control_json).read_text())
    ca,co=np.load(args.candidate_npz),np.load(args.control_npz)
    cl,ol=ca["logits"].astype(np.float32),co["logits"].astype(np.float32);diff=np.abs(cl-ol)
    quality={"candidate_token":int(ca["token"]),"control_token":int(co["token"]),
      "same_token":int(ca["token"])==int(co["token"]),"finite":bool(np.isfinite(cl).all() and np.isfinite(ol).all()),
      "max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
      "allclose_rtol_0p02_atol_0p5":bool(np.allclose(cl,ol,rtol=.02,atol=.5))}
    passed=cj["status"]==ctl["status"]=="PASS" and all(quality[k] for k in ("same_token","finite","allclose_rtol_0p02_atol_0p5"))
    payload={"schema":"tinygrad.nv_compiler_q4k_k_model_compare.v1","status":"PASS" if passed else "FAIL",
      "correctness":quality,"wall":{"candidate_min_ms":cj["wall"]["min_ms"],"control_min_ms":ctl["wall"]["min_ms"],
        "candidate_minus_control_ms":cj["wall"]["min_ms"]-ctl["wall"]["min_ms"],
        "candidate_tok_s":512000/cj["wall"]["min_ms"],"control_tok_s":512000/ctl["wall"]["min_ms"]},
      "authority":{"candidate":args.candidate_json,"control":args.control_json}}
    _write(args.out,payload)
    if not passed:raise SystemExit(1)
    return

  if os.environ.get("NV_COMPILER_Q4_IMMA_PP512") != "1" or os.environ.get("NV_Q4_IMMA_PP512") is not None:
    raise SystemExit("both arms require only the compiler gate/up baseline")
  k_env=os.environ.get("NV_COMPILER_Q4_IMMA_K_PP512")
  if (args.arm=="candidate") != (k_env=="1") or (args.arm=="control" and k_env is not None):
    raise SystemExit("candidate requires K env=1; control requires K env unset")
  from tinygrad.llm.generate import load_model_and_tokenizer
  from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for as gate_binding_for
  model,_=load_model_and_tokenizer(args.model,args.max_context,seed=20260617)
  gate_asset=gate_binding_for("NV");gate_asset.prepare_records(72);gate_asset.install_warmstart(model);gate_capture=gate_asset.new_capture()
  k_asset=k_capture=None
  if args.arm=="candidate":
    from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import binding_for as k_binding_for
    k_asset=k_binding_for("NV");k_asset.prepare_records(36);k_asset.install_warmstart(model);k_capture=k_asset.new_capture()

  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
  chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous();temp=Tensor([0.0])
  jit=_capture(model,gate_capture,k_capture,chunk_a,temp)
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
  gate_mains=[c for c in calls if c.src[0].src and getattr(c.src[0].src[0].arg,"candidate_context",None) is not None and
              c.src[0].src[0].arg.candidate_context.canonical_identity==gate_asset.candidate_identity]
  k_mains=[] if k_asset is None else [c for c in calls if c.src[0].src and
    getattr(c.src[0].src[0].arg,"candidate_context",None) is not None and
    c.src[0].src[0].arg.candidate_context.canonical_identity==k_asset.candidate_identity]
  canonical_k={block.attn_k.prefill_packed_weight().uop.buf_uop for block in model.blk}
  k_weight_args=[_buf_uop(c.src[3]) for c in k_mains if len(c.src)>3]
  k_record_args={_buf_uop(c.src[2]) for c in k_mains if len(c.src)>2}
  q8_calls=[c for c in calls if _call_name(c)==gate_asset.producer.arg.name]
  k_q8_calls=[c for c in q8_calls if len(c.src)>2 and _buf_uop(c.src[2]) in k_record_args]
  k_overlays=sum(getattr(block.attn_k,"_pf16_w",None) is not None for block in model.blk)
  k_geometries=[(c.src[0].arg.global_size,c.src[0].arg.local_size) for c in k_mains]
  census={"gate_up_main":len(gate_mains),"total_q8_producer":len(q8_calls),"k_q8_producer":len(k_q8_calls),
    "k_compiler_main":len(k_mains),"k_candidate_weight_args":len(k_weight_args),"k_unique_weight_bases":len(set(k_weight_args)),
    "k_all_weights_canonical":bool(k_weight_args and all(w in canonical_k for w in k_weight_args)),
    "k_fp16_overlays":k_overlays,"k_weight_copy_kernels":0 if k_weight_args and all(w in canonical_k for w in k_weight_args) else -1,
    "k_old_fixup":names.get("q4k_imma_fixup",0),"k_records":len(k_record_args),
    "k_partial_workspace_bytes":0,"k_all_256cta":bool(k_geometries and all(g==((32,8,1),(32,2,2)) for g in k_geometries))}
  replay_pass=all(replay[k] for k in ("finite","same_token","same_logits_exact","same_activation_exact","distinct_activation_output"))
  if args.arm=="candidate":
    passed=replay_pass and all((census["gate_up_main"]==72,census["total_q8_producer"]==108,
      census["k_q8_producer"]==36,census["k_compiler_main"]==36,census["k_candidate_weight_args"]==36,
      census["k_unique_weight_bases"]==36,census["k_all_weights_canonical"],census["k_fp16_overlays"]==0,
      census["k_weight_copy_kernels"]==0,census["k_old_fixup"]==0,census["k_records"]==36,
      census["k_partial_workspace_bytes"]==0,census["k_all_256cta"]))
  else:
    passed=replay_pass and all((census["gate_up_main"]==72,census["total_q8_producer"]==72,
      census["k_q8_producer"]==0,census["k_compiler_main"]==0,census["k_fp16_overlays"]==36))
  payload={"schema":"tinygrad.nv_compiler_q4k_k_model_arm.v1","arm":args.arm,"status":"PASS" if passed else "FAIL",
    "route":{"default_enabled":False,"gate_up_identity":gate_asset.candidate_identity,
             "k_identity":None if k_asset is None else k_asset.candidate_identity},"census":census,"replay":replay,
    "wall":{"samples_ms":samples,"min_ms":min(samples),"median_ms":statistics.median(samples),"tok_s":512000/min(samples)},
    "program_names":dict(sorted(names.items())),"token":a1[0]}
  _write(args.out,payload)
  if not passed:raise SystemExit(1)


if __name__=="__main__":main()
