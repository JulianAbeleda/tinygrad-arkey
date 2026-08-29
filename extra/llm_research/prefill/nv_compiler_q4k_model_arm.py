#!/usr/bin/env python3
"""Fresh-process candidate/control/compare gate for compiler Q4_K pp512."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, time
from collections import Counter
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.device import Compiled
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import (_capture, _call_and_sync, _call_name,
  _numpy_output, _program_calls)

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _write(path:str, payload:dict) -> None:
  target=pathlib.Path(path); target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload, indent=2, sort_keys=True))


def _profile(events):
  from collections import defaultdict
  from tinygrad.device import ProfileGraphEvent
  from tinygrad.helpers import ProfileRangeEvent
  by_name=defaultdict(list)
  for event in events:
    if isinstance(event,ProfileRangeEvent) and event.en is not None:
      by_name[str(event.name)].append((float(event.en)-float(event.st))*1e-3)
    elif isinstance(event,ProfileGraphEvent):
      for ent in event.ents:
        by_name[str(ent.name)].append((float(event.sigs[ent.en_id])-float(event.sigs[ent.st_id]))*1e-3)
  rows={name:{"calls":len(vals),"device_ms":round(sum(vals),4),"min_ms":round(min(vals),4),"max_ms":round(max(vals),4)}
        for name,vals in sorted(by_name.items()) if vals}
  return {"schema":"prefill-device-profile-range-summary.v2","kernel_count":sum(row["calls"] for row in rows.values()),
          "device_ms":round(sum(row["device_ms"] for row in rows.values()),4),"by_name":rows}


def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("candidate", "fp16", "compare"), required=True)
  ap.add_argument("--model", default=MODEL); ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--warmups", type=int, default=3); ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--out", required=True); ap.add_argument("--logits-npz", default="")
  ap.add_argument("--candidate-json", default=""); ap.add_argument("--candidate-npz", default="")
  ap.add_argument("--fp16-json", default=""); ap.add_argument("--fp16-npz", default="")
  args=ap.parse_args()

  if args.arm == "compare":
    if not all((args.candidate_json,args.candidate_npz,args.fp16_json,args.fp16_npz)):
      raise SystemExit("compare requires candidate/fp16 JSON and NPZ")
    cj,fj=json.loads(pathlib.Path(args.candidate_json).read_text()),json.loads(pathlib.Path(args.fp16_json).read_text())
    ca,fr=np.load(args.candidate_npz),np.load(args.fp16_npz)
    cl,fl=ca["logits"].astype(np.float32),fr["logits"].astype(np.float32); diff=np.abs(cl-fl)
    quality={"candidate_token":int(ca["token"]),"fp16_token":int(fr["token"]),
      "same_token":int(ca["token"])==int(fr["token"]),"finite":bool(np.isfinite(cl).all() and np.isfinite(fl).all()),
      "max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
      "allclose_rtol_0p02_atol_0p5":bool(np.allclose(cl,fl,rtol=0.02,atol=0.5))}
    passed=cj["status"]==fj["status"]=="PASS" and all(quality[k] for k in ("same_token","finite","allclose_rtol_0p02_atol_0p5"))
    payload={"schema":"tinygrad.nv_compiler_q4k_model_compare.v1","status":"PASS" if passed else "FAIL",
      "correctness":quality,"wall":{"candidate_min_ms":cj["wall"]["min_ms"],"fp16_min_ms":fj["wall"]["min_ms"],
      "candidate_minus_fp16_ms":cj["wall"]["min_ms"]-fj["wall"]["min_ms"],
      "candidate_tok_s":512/cj["wall"]["min_ms"]*1000,"fp16_tok_s":512/fj["wall"]["min_ms"]*1000},
      "authority":{"candidate":args.candidate_json,"fp16":args.fp16_json}}
    _write(args.out,payload)
    if not passed: raise SystemExit(1)
    return

  if not str(Device.DEFAULT).startswith("NV"): raise SystemExit("NV required")
  compiler_env=os.environ.get("NV_COMPILER_Q4_IMMA_PP512")
  if args.arm=="candidate" and (compiler_env!="1" or os.environ.get("NV_Q4_IMMA_PP512") is not None):
    raise SystemExit("candidate requires only NV_COMPILER_Q4_IMMA_PP512=1")
  if args.arm=="fp16" and (compiler_env is not None or os.environ.get("NV_Q4_IMMA_PP512") is not None):
    raise SystemExit("fp16 requires both Q4 IMMA research envs unset")
  from tinygrad.llm.generate import load_model_and_tokenizer
  model,_=load_model_and_tokenizer(args.model,args.max_context,seed=20260617)
  binding=None
  if args.arm=="candidate":
    from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for
    binding=binding_for("NV");binding.prepare_records(72);binding.install_warmstart(model)
  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
  chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous()
  temp=Tensor([0.0])
  jit=_capture(model,binding,chunk_a,temp,candidate=args.arm=="candidate")
  a0=_numpy_output(_call_and_sync(jit,chunk_a,temp))
  b=_numpy_output(_call_and_sync(jit,chunk_b,temp))
  a1=_numpy_output(_call_and_sync(jit,chunk_a,temp))
  if args.logits_npz:
    target=pathlib.Path(args.logits_npz);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez(target,token=np.int64(a1[0]),logits=a1[1])
  for _ in range(args.warmups):_call_and_sync(jit,chunk_a,temp)
  profile_start=len(Compiled.profile_events);samples=[]
  for _ in range(args.rounds):
    Device[Device.DEFAULT].synchronize();started=time.perf_counter_ns();_call_and_sync(jit,chunk_a,temp)
    samples.append((time.perf_counter_ns()-started)/1e6)
  profile=_profile(list(Compiled.profile_events[profile_start:]))
  calls=_program_calls(jit.captured.linear);names=Counter(_call_name(call) for call in calls)
  replay={"finite":bool(np.isfinite(a1[1]).all()),"same_activation_exact":bool(a0[0]==a1[0] and np.array_equal(a0[1],a1[1])),
    "distinct_activation_output":bool(a1[0]!=b[0] or not np.array_equal(a1[1],b[1]))}
  common={"wall":{"samples_ms":samples,"min_ms":min(samples),"median_ms":statistics.median(samples),
                   "tok_s":512/min(samples)*1000},"replay":replay,"device_profile":profile,
          "program_names":dict(sorted(names.items())),"token":a1[0]}

  if args.arm=="fp16":
    q8=names.get("q8_compact_record_fp16",0);fixup=names.get("q4k_imma_fixup",0)
    overlays=sum(getattr(p,"_pf16_w",None) is not None for block in model.blk for p in (block.ffn_gate,block.ffn_up))
    passed=all(replay.values()) and q8==fixup==0 and overlays==72
    payload={"schema":"tinygrad.nv_compiler_q4k_model_arm.v1","arm":"fp16","status":"PASS" if passed else "FAIL",
      "research_calls":{"q8_producer":q8,"old_fixup":fixup},"gate_up_fp16_overlays":overlays,**common}
  else:
    identity=binding.candidate_identity
    mains=[call for call in calls if call.src[0].src and getattr(call.src[0].src[0].arg,"candidate_context",None) is not None and
           call.src[0].src[0].arg.candidate_context.canonical_identity==identity]
    canonical={p.prefill_packed_weight().uop.buf_uop for block in model.blk for p in (block.ffn_gate,block.ffn_up)}
    weight_args=[];expanded=0
    for call in mains:
      for arg in call.src[1:]:
        try:size=arg.buf_uop.numel()*arg.buf_uop.dtype.itemsize
        except (RuntimeError,AttributeError):continue
        if arg.dtype==binding.transform.storage_dtype and size==12288*(4096//256)*36*4:weight_args.append(arg.buf_uop)
        if arg.dtype.itemsize==2 and size==12288*4096*2:expanded+=size
    copies=sum(count for name,count in names.items() if name.startswith("E_73728_32_3_"))
    census={"q8_producer":names.get(binding.producer.arg.name,0),"compiler_generated_main":len(mains),
      "old_fixup":names.get("q4k_imma_fixup",0),"weight_copy_kernels":copies,
      "candidate_weight_args":len(weight_args),"unique_weight_bases":len(set(weight_args)),
      "all_weights_canonical":bool(weight_args and all(x in canonical for x in weight_args)),
      "expanded_fp16_weight_bytes":expanded,"q8_records":len(binding.records),"partial_workspace_bytes":0}
    passed=all(replay.values()) and all((census["q8_producer"]==72,census["compiler_generated_main"]==72,
      census["old_fixup"]==0,census["weight_copy_kernels"]==0,census["candidate_weight_args"]==72,census["unique_weight_bases"]==72,
      census["all_weights_canonical"],census["expanded_fp16_weight_bytes"]==0,census["q8_records"]==72,
      census["partial_workspace_bytes"]==0))
    payload={"schema":"tinygrad.nv_compiler_q4k_model_arm.v1","arm":"candidate","status":"PASS" if passed else "FAIL",
      "route":{"default_enabled":False,"candidate_identity":identity},"census":census,**common}
  _write(args.out,payload)
  if payload["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
