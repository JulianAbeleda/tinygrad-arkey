#!/usr/bin/env python3
"""Isolated compiler gate/up + K + Q/O pp512 composition authority."""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, pathlib, statistics, time
from collections import Counter
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.prefill_routes import prefill_route_override
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import (_call_and_sync, _call_name, _compile_scope, _configure,
  _numpy_output, _program_calls)

MODEL="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


class _GraphOwnedQOCapture:
  """Capture-local Q/O storage using the already-qualified compiler PROGRAM.

  The isolated Q/O gate deliberately retained 72 device-global buffers to
  diagnose replay.  Composition must instead give those lifetimes to the
  captured graph, exactly like the passing gate/up binding does.
  """
  def __init__(self,asset,record_u32):
    self.asset,self.record_u32=asset,record_u32
    self.records,self.outputs,self.cursor=[],[],0

  @property
  def candidate_identity(self):return self.asset.candidate_identity

  @property
  def transform(self):return self.asset.transform

  @property
  def producer(self):return self.asset.producer

  def begin_trace(self):self.cursor=0

  def project(self,x,words,*,model_family,role,weight_type="Q4_K"):
    if model_family!="qwen3_8b" or role not in ("attn_q","attn_output") or weight_type!="Q4_K" or \
        tuple(x.shape)!=(512,4096) or x.device!="NV":raise ValueError("unsupported graph-owned Q/O route")
    if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32:raise ValueError("Q/O route requires fp16 x and canonical uint32 Q4_K")
    if self.cursor>=72:raise RuntimeError("graph-owned Q/O trace exceeded exact 72-projection census")
    if self.cursor==0:self.records,self.outputs=[],[]
    record=Tensor.empty(self.record_u32,dtype=dtypes.uint32,device=x.device)
    out=Tensor.empty(512*4096,dtype=dtypes.float32,device=x.device)
    self.records.append(record);self.outputs.append(out);self.cursor+=1
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    out,record,words=out.uop_program(record,words,fxn=lambda *_:self.asset.main_program)
    # Retain the produced Tensor identities, not the pre-program placeholders.
    self.records[-1],self.outputs[-1]=record,out
    return out.reshape(512,4096)


class _GraphOwnedKCapture:
  """Use K's frozen compiler PROGRAM when composition changes scheduler grouping."""
  def __init__(self,asset,record_u32):
    self.asset,self.record_u32=asset,record_u32
    self.records,self.outputs,self.cursor=[],[],0

  @property
  def candidate_identity(self):return self.asset.candidate_identity

  @property
  def transform(self):return self.asset.transform

  @property
  def producer(self):return self.asset.producer

  def begin_trace(self):self.cursor=0

  def project(self,x,words,*,model_family,role,weight_type="Q4_K",wait=False):
    del wait
    if model_family!="qwen3_8b" or role!="attn_k" or weight_type!="Q4_K" or tuple(x.shape)!=(512,4096) or x.device!="NV":
      raise ValueError("unsupported graph-owned K route")
    if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32:raise ValueError("K route requires fp16 x and canonical uint32 Q4_K")
    if self.cursor>=36:raise RuntimeError("graph-owned K trace exceeded exact 36-projection census")
    if self.cursor==0:self.records,self.outputs=[],[]
    record=Tensor.empty(self.record_u32,dtype=dtypes.uint32,device=x.device)
    out=Tensor.empty(512*1024,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    out,record,words=out.uop_program(record,words,fxn=lambda *_:self.asset.main_program)
    self.records.append(record);self.outputs.append(out);self.cursor+=1
    return out.reshape(512,1024)


def _write(path,payload):
  target=pathlib.Path(path);target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,sort_keys=True))


def _hash_tensor(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _hash_buffer(buf):return hashlib.sha256(buf.numpy().tobytes()).hexdigest()


def _graph_stage_buffers(jit,identities):
  """Return the allocations actually rebound into the captured HCQ graphs."""
  from tinygrad.engine.realize import graph_cache
  by_identity={value:key for key,value in identities.items() if value is not None}
  stages={f"{role}_{kind}":[] for role in by_identity.values() for kind in ("records","outputs")}
  for outer in jit.captured.linear.src:
    if outer.src[0].op is not Ops.CUSTOM_FUNCTION or outer.src[0].arg!="graph":continue
    graph=graph_cache.get(outer.src[0])
    if graph is None:raise RuntimeError("captured graph missing from graph cache")
    for _,call,bufs,_ in graph.calls:
      if call.op is not Ops.PROGRAM:continue
      ctx=getattr(call.arg,"candidate_context",None);role=by_identity.get(getattr(ctx,"canonical_identity",None))
      if role is None:continue
      if call.arg.outs!=(0,) or call.arg.ins!=(1,2):raise RuntimeError(f"unexpected {role} captured ABI")
      stages[f"{role}_outputs"].append(bufs[0]);stages[f"{role}_records"].append(bufs[1])
  return stages


def _snapshot(model,stages):
  return {**{key:[_hash_buffer(buf) for buf in bufs] for key,bufs in stages.items()},
    "kv":[_hash_tensor(block.cache_kv[:,:,:,:512,:]) for block in model.blk]}


def _compare_snapshot(reference,current):
  return {key:{"exact_by_index":[a==b for a,b in zip(reference[key],current[key])],
    "first_mismatch":next((i for i,(a,b) in enumerate(zip(reference[key],current[key])) if a!=b),None),
    "same_length":len(reference[key])==len(current[key])} for key in reference}


def _capture(model,qo,chunk,temp,candidate):
  gate,kval=model._nv_gkqo_gate_capture,model._nv_gkqo_k_capture
  _configure(model,gate)
  for block in model.blk:block._nv_compiler_q4_imma_k_pp512_binding=kval
  role_by_id={}
  if candidate:
    for block in model.blk:role_by_id[id(block.attn_q)]="attn_q";role_by_id[id(block.attn_output)]="attn_output"
    def route(lin,x):
      if (role:=role_by_id.get(id(lin))) is None:return None
      if tuple(x.shape)!=(1,512,4096):raise RuntimeError(f"unexpected combined Q/O activation {x.shape}")
      x16=x.cast(dtypes.float16).contiguous()
      return qo.project(x16.reshape(512,4096),lin.prefill_packed_weight(),model_family="qwen3_8b",role=role).reshape(1,512,4096)
    override=prefill_route_override(route)
  else:override=contextlib.nullcontext()
  @TinyJit
  def run(tokens,temperature):return model.forward_greedy_with_logits(tokens,0,temperature)
  with override,_compile_scope(model):
    for _ in range(3):
      gate.begin_trace();kval.begin_trace()
      if candidate:qo.begin_trace()
      _call_and_sync(run,chunk,temp)
  if run.captured is None:raise RuntimeError("combined gate/up+K+Q/O arm did not capture")
  return run


def _buf_uop(arg):
  try:return arg.buf_uop
  except (RuntimeError,AttributeError):return None


def _identity_calls(calls,identity):
  return [c for c in calls if c.src[0].src and getattr(c.src[0].src[0].arg,"candidate_context",None) is not None and
          c.src[0].src[0].arg.candidate_context.canonical_identity==identity]


def _weight_arg(call,transform):
  for arg in call.src[1:]:
    u=_buf_uop(arg)
    if u is None:continue
    try:size=u.numel()*u.dtype.itemsize
    except (RuntimeError,AttributeError):continue
    if u.dtype==transform.storage_dtype and size==transform.packed_bytes:return u
  return None


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("candidate","control","compare"),required=True)
  ap.add_argument("--model",default=MODEL);ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--rounds",type=int,default=9)
  ap.add_argument("--replay-cycles",type=int,default=20);ap.add_argument("--deep-replay",action="store_true")
  ap.add_argument("--out",required=True);ap.add_argument("--logits-npz",default="")
  ap.add_argument("--candidate-json",default="");ap.add_argument("--candidate-npz",default="")
  ap.add_argument("--control-json",default="");ap.add_argument("--control-npz",default="");args=ap.parse_args()
  if args.arm=="compare":
    cj,ctl=json.loads(pathlib.Path(args.candidate_json).read_text()),json.loads(pathlib.Path(args.control_json).read_text())
    ca,co=np.load(args.candidate_npz),np.load(args.control_npz);cl,ol=ca["logits"].astype(np.float32),co["logits"].astype(np.float32)
    diff=np.abs(cl-ol);quality={"candidate_token":int(ca["token"]),"control_token":int(co["token"]),
      "same_token":int(ca["token"])==int(co["token"]),"finite":bool(np.isfinite(cl).all() and np.isfinite(ol).all()),
      "max_abs":float(diff.max()),"mean_abs":float(diff.mean()),"allclose_rtol_0p02_atol_0p5":bool(np.allclose(cl,ol,rtol=.02,atol=.5))}
    performance=cj["wall"]["min_ms"]<ctl["wall"]["min_ms"]
    passed=cj["status"]==ctl["status"]=="PASS" and performance and all(quality[k] for k in ("same_token","finite","allclose_rtol_0p02_atol_0p5"))
    _write(args.out,{"schema":"tinygrad.nv_compiler_q4k_gkqo_compare.v1","status":"PASS" if passed else "NO_GO",
      "correctness":quality,"performance_pass":performance,"wall":{"candidate_min_ms":cj["wall"]["min_ms"],
      "control_min_ms":ctl["wall"]["min_ms"],"candidate_minus_control_ms":cj["wall"]["min_ms"]-ctl["wall"]["min_ms"],
      "candidate_tok_s":512000/cj["wall"]["min_ms"],"control_tok_s":512000/ctl["wall"]["min_ms"]}})
    if not passed:raise SystemExit(1)
    return

  if os.environ.get("NV_COMPILER_Q4_IMMA_PP512")!="1" or os.environ.get("NV_COMPILER_Q4_IMMA_K_PP512")!="1" or \
      os.environ.get("NV_Q4_IMMA_PP512") is not None or os.environ.get("NV_COMPILER_Q6_IMMA_PP512") is not None:
    raise SystemExit("combined arm requires compiler gate/up+K and excludes raw/Q6 routes")
  qo_env=os.environ.get("NV_COMPILER_Q4_IMMA_QO_PP512")
  if (args.arm=="candidate")!=(qo_env=="1") or (args.arm=="control" and qo_env is not None):
    raise SystemExit("candidate requires Q/O env=1; control requires Q/O env unset")

  from tinygrad.llm.generate import load_model_and_tokenizer
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for as gate_binding_for
  from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import binding_for as k_binding_for, RECORD_U32 as K_RECORD_U32
  model,_=load_model_and_tokenizer(args.model,4608,seed=20260617)
  gate_asset=gate_binding_for("NV");gate_asset.prepare_records(72);gate_asset.install_warmstart(model);gate=gate_asset.new_capture()
  k_asset=k_binding_for("NV");k_asset.prepare_records(36);k_asset.install_warmstart(model)
  # The matched control retains K's independently-qualified ordinary carrier.
  # Q/O composition changes its scheduling boundary, so the combined candidate
  # invokes K's exact same frozen compiler PROGRAM with graph-owned storage.
  kval=_GraphOwnedKCapture(k_asset,K_RECORD_U32) if args.arm=="candidate" else k_asset.new_capture()
  model._nv_gkqo_gate_capture,model._nv_gkqo_k_capture=gate,kval
  qo=None;qo_linears=[p for block in model.blk for p in (block.attn_q,block.attn_output)]
  if args.arm=="candidate":
    from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import binding_for as qo_binding_for, RECORD_U32
    # binding_for() has already compiled and frozen Q/O's ordinary compiler PROGRAM under
    # its typed contract.  The composed graph invokes only that opaque PROGRAM, so adding
    # Q/O's shape key to the ambient model warmstart table would incorrectly claim unrelated
    # 512x4096x4096 ordinary matmuls (the exact composition collision this arm must avoid).
    qo=_GraphOwnedQOCapture(qo_binding_for("NV"),RECORD_U32)
    for lin in qo_linears:
      if hasattr(lin,"_pf16_w"):delattr(lin,"_pf16_w")
  identities={"gate_up":gate.candidate_identity,"k":kval.candidate_identity,"qo":None if qo is None else qo.candidate_identity}

  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
  chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous();temp=Tensor([0.0])
  jit=_capture(model,qo,chunk_a,temp,args.arm=="candidate")
  stage_buffers=_graph_stage_buffers(jit,identities) if args.deep_replay else {}
  stage_census={key:{"calls":len(bufs),"unique_allocations":len({id(buf) for buf in bufs})} for key,bufs in stage_buffers.items()}
  a0=_numpy_output(_call_and_sync(jit,chunk_a,temp));deep0=_snapshot(model,stage_buffers) if args.deep_replay else None
  cycles=[];a1=b=None
  deep_cycles=[]
  for cycle in range(args.replay_cycles):
    b=_numpy_output(_call_and_sync(jit,chunk_b,temp));a1=_numpy_output(_call_and_sync(jit,chunk_a,temp));diff=np.abs(a0[1]-a1[1])
    cycle_row={"cycle":cycle,"same_token":a0[0]==a1[0],"same_logits_exact":bool(np.array_equal(a0[1],a1[1])),
      "max_abs":float(diff.max()),"mean_abs":float(diff.mean())}
    if deep0 is not None:
      deep_row=_compare_snapshot(deep0,_snapshot(model,stage_buffers));deep_cycles.append(deep_row)
      cycle_row.update({f"{key}_exact":value["same_length"] and value["first_mismatch"] is None
                        for key,value in deep_row.items()})
    cycles.append(cycle_row)
  assert a1 is not None and b is not None
  if args.logits_npz:
    target=pathlib.Path(args.logits_npz);target.parent.mkdir(parents=True,exist_ok=True);np.savez(target,token=np.int64(a1[0]),logits=a1[1])
  for _ in range(args.warmups):_call_and_sync(jit,chunk_a,temp)
  samples=[]
  for _ in range(args.rounds):
    Device[Device.DEFAULT].synchronize();st=time.perf_counter_ns();_call_and_sync(jit,chunk_a,temp);samples.append((time.perf_counter_ns()-st)/1e6)

  calls=_program_calls(jit.captured.linear);names=Counter(_call_name(c) for c in calls)
  mains={role:([] if ident is None else _identity_calls(calls,ident)) for role,ident in identities.items()}
  transforms={"gate_up":gate.transform,"k":kval.transform,"qo":None if qo is None else qo.transform}
  weights=[]
  for role in ("gate_up","k","qo"):
    if transforms[role] is not None:weights += [_weight_arg(c,transforms[role]) for c in mains[role]]
  weights=[x for x in weights if x is not None]
  canonical={lin.prefill_packed_weight().uop.buf_uop for block in model.blk for lin in
    (block.ffn_gate,block.ffn_up,block.attn_k,block.attn_q,block.attn_output) if isinstance(lin,Q4KPrimitiveLinear)}
  admitted=[lin for block in model.blk for lin in (block.ffn_gate,block.ffn_up,block.attn_k,block.attn_q,block.attn_output)]
  remaining=[lin for block in model.blk for lin in (block.attn_v,block.ffn_down)]
  total_mains=sum(len(x) for x in mains.values());q8=names.get("q8_compact_record_fp16",0)
  census={"gate_up_main":len(mains["gate_up"]),"k_main":len(mains["k"]),"qo_main":len(mains["qo"]),
    "compiler_main_total":total_mains,"q8_producer_total":q8,"candidate_weight_args":len(weights),
    "unique_weight_bases":len(set(weights)),"all_weights_canonical":bool(weights and all(x in canonical for x in weights)),
    "admitted_fp16_overlays":sum(getattr(x,"_pf16_w",None) is not None for x in admitted),
    "remaining_v_down_fp16_overlays":sum(getattr(x,"_pf16_w",None) is not None for x in remaining),
    "weight_copy_kernels":0 if weights and all(x in canonical for x in weights) else -1,
    "old_fixups":names.get("q4k_imma_fixup",0),"partial_workspace_bytes":0}
  replay={"finite":bool(np.isfinite(a1[1]).all()),"same_token":a0[0]==a1[0],
    "distinct_activation_output":bool(a1[0]!=b[0] or not np.array_equal(a1[1],b[1])),"cycles":cycles}
  replay_pass=replay["finite"] and replay["same_token"] and replay["distinct_activation_output"] and \
    all(x["same_token"] and x["same_logits_exact"] for x in cycles)
  deep=None
  if deep0 is not None:
    deep={"cycles":deep_cycles,"all_cycles_exact":all(value["same_length"] and value["first_mismatch"] is None
      for cycle in deep_cycles for value in cycle.values())}
    replay_pass &= deep["all_cycles_exact"]
  expected_stage={"gate_up_records":72,"gate_up_outputs":72,"k_records":36,"k_outputs":36}
  if args.arm=="candidate":expected_stage.update({"qo_records":72,"qo_outputs":72})
  stage_census_pass=not args.deep_replay or all(stage_census.get(key,{}).get("calls")==count for key,count in expected_stage.items())
  if args.arm=="candidate":
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["k_main"]==36,census["qo_main"]==72,
      census["compiler_main_total"]==180,census["q8_producer_total"]==180,census["candidate_weight_args"]==180,
      census["unique_weight_bases"]==180,census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,
      census["remaining_v_down_fp16_overlays"]==72,census["weight_copy_kernels"]==0,census["old_fixups"]==0))
  else:
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["k_main"]==36,census["qo_main"]==0,
      census["compiler_main_total"]==108,census["q8_producer_total"]==108,census["candidate_weight_args"]==108,
      census["unique_weight_bases"]==108,census["all_weights_canonical"],census["admitted_fp16_overlays"]==72,
      census["remaining_v_down_fp16_overlays"]==72,census["weight_copy_kernels"]==0,census["old_fixups"]==0))
  payload={"schema":"tinygrad.nv_compiler_q4k_gkqo_model_arm.v1","arm":args.arm,"status":"PASS" if replay_pass and structural else "FAIL",
    "route":{"default_enabled":False,"identities":identities},"stage_buffer_census":stage_census,
    "environment":{"HCQ_NUM_COMPUTE":os.environ.get("HCQ_NUM_COMPUTE"),
    "HCQ_NV_READY_PLACEMENT":os.environ.get("HCQ_NV_READY_PLACEMENT"),"HCQ_NV_MULTI_QUEUE_CUT_POLICY":os.environ.get("HCQ_NV_MULTI_QUEUE_CUT_POLICY")},
    "census":census,"replay":replay,"deep_replay":deep,"wall":{"samples_ms":samples,"min_ms":min(samples),
    "median_ms":statistics.median(samples),"tok_s":512000/min(samples)},"program_names":dict(sorted(names.items())),"token":a1[0]}
  _write(args.out,payload)
  if payload["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
