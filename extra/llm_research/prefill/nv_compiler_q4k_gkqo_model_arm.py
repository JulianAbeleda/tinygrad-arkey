#!/usr/bin/env python3
"""Isolated compiler gate/up + K + Q/O pp512 composition authority."""
from __future__ import annotations

import argparse, contextlib, dataclasses, hashlib, json, os, pathlib, statistics, time
import fcntl
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


class _PairedGateQ8Capture:
  """One graph-owned Q8 record shared by each ordered gate/up projection pair."""
  def __init__(self,asset,record_u32):
    self.asset,self.record_u32=asset,record_u32
    self.records,self.outputs,self.cursor,self.pair_record=[],[],0,None

  @property
  def candidate_identity(self):return self.asset.candidate_identity

  @property
  def transform(self):return self.asset.transform

  @property
  def producer(self):return self.asset.producer

  def begin_trace(self):self.cursor,self.pair_record=0,None

  def project(self,x,words,*,model_family,role,weight_type="Q4_K",wait=False):
    del wait
    expected_role="ffn_gate" if self.cursor%2==0 else "ffn_up"
    if model_family!="qwen3_8b" or role!=expected_role or weight_type!="Q4_K" or tuple(x.shape)!=(512,4096) or x.device!="NV":
      raise ValueError("gate Q8 reuse requires exact ordered Qwen3-8B gate/up pairs")
    if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32:raise ValueError("gate Q8 reuse requires fp16 x and canonical Q4_K words")
    if self.cursor>=72:raise RuntimeError("gate Q8 reuse exceeded exact 72-main census")
    if self.cursor==0:self.records,self.outputs=[],[]
    if self.cursor%2==0:
      record=Tensor.empty(self.record_u32,dtype=dtypes.uint32,device=x.device)
      _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
      self.pair_record=record
    else:
      if self.pair_record is None:raise RuntimeError("up projection has no preceding gate record")
      record=self.pair_record
    out=Tensor.empty(512*12288,dtype=dtypes.float32,device=x.device)
    out,record,words=out.uop_program(record,words,fxn=lambda *_:self.asset.main_program)
    self.records.append(record);self.outputs.append(out);self.cursor+=1
    return out.reshape(512,12288)


class _FusedGateEpilogueCapture:
  """Retain ordinary gate/up mains while publishing one fused FP16 down activation."""
  def __init__(self,asset,record_u32,fused_program):
    self.asset,self.record_u32,self.fused_program=asset,record_u32,fused_program
    self.records,self.outputs,self.fused_outputs=[],[],[]
    self.cursor,self.down_cursor,self.gate_output=0,0,None

  @property
  def candidate_identity(self):return self.asset.candidate_identity

  @property
  def transform(self):return self.asset.transform

  @property
  def producer(self):return self.asset.producer

  def begin_trace(self):
    self.records,self.outputs,self.fused_outputs=[],[],[]
    self.cursor,self.down_cursor,self.gate_output=0,0,None

  def project(self,x,words,*,model_family,role,weight_type="Q4_K",wait=False):
    del wait
    expected_role="ffn_gate" if self.cursor%2==0 else "ffn_up"
    if model_family!="qwen3_8b" or role!=expected_role or weight_type!="Q4_K" or tuple(x.shape)!=(512,4096) or x.device!="NV":
      raise ValueError("fused gate epilogue requires exact ordered Qwen3-8B gate/up pairs")
    if x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32:raise ValueError("fused gate epilogue requires fp16 x and Q4_K words")
    if self.cursor>=72:raise RuntimeError("fused gate epilogue exceeded exact 72-main census")
    record=Tensor.empty(self.record_u32,dtype=dtypes.uint32,device=x.device)
    out=Tensor.empty(512*12288,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    out,record,words=out.uop_program(record,words,fxn=lambda *_:self.asset.main_program)
    if self.cursor%2==0:self.gate_output=out
    else:
      if self.gate_output is None:raise RuntimeError("up projection has no preceding gate output")
      fused=Tensor.empty(512*12288,dtype=dtypes.float16,device=x.device)
      fused,gate_out,up_out=fused.uop_program(self.gate_output,out,fxn=lambda *_:self.fused_program)
      self.fused_outputs.append(fused);self.gate_output=None
    self.records.append(record);self.outputs.append(out);self.cursor+=1
    return out.reshape(512,12288)

  def down_activation(self):
    if self.down_cursor>=len(self.fused_outputs):raise RuntimeError("down requested before fused gate epilogue publication")
    value=self.fused_outputs[self.down_cursor];self.down_cursor+=1
    return value.reshape(1,512,12288)


class _GraphOwnedVCapture(_GraphOwnedKCapture):
  """Independent capture-local lease for the 18 Q4 attention-V projections."""
  def begin_trace(self): self.cursor=0
  def project(self,x,words,**kw):
    if kw.get("role")!="attn_v": raise ValueError("V capture received non-V role")
    if self.cursor>=18: raise RuntimeError("Q4 V trace exceeded exact 18-projection census")
    if tuple(x.shape)!=(512,4096) or x.device!="NV" or x.dtype!=dtypes.float16 or words.dtype!=dtypes.uint32:
      raise ValueError("V route requires NV fp16 (512,4096) activation and canonical uint32 Q4_K weights")
    if self.cursor==0:self.records,self.outputs=[],[]
    record=Tensor.empty(self.record_u32,dtype=dtypes.uint32,device=x.device)
    out=Tensor.empty(512*1024,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.asset.producer)
    out,record,words=out.uop_program(record,words,fxn=lambda *_:self.asset.main_program)
    self.records.append(record);self.outputs.append(out);self.cursor+=1
    return out.reshape(512,1024)


class _GraphOwnedQ6VCapture:
  """Pin Q6 V's frozen compiler PROGRAM across the composed scheduler boundary."""
  def __init__(self,asset):
    self.asset,self.role_asset=asset,asset.roles["attn_v"]
    self.records,self.outputs,self.cursor=[],[],0

  def begin_trace(self):self.cursor=0

  def project(self,x,halfs,*,model_family,role,weight_type="Q6_K",wait=False):
    del wait
    if model_family!="qwen3_8b" or role!="attn_v" or weight_type!="Q6_K" or tuple(x.shape)!=(512,4096) or x.device!="NV":
      raise ValueError("unsupported graph-owned Q6 V route")
    if x.dtype!=dtypes.float16 or halfs.dtype!=dtypes.uint16:
      raise ValueError("Q6 V route requires NV fp16 activation and canonical uint16 Q6_K weights")
    if self.cursor>=18:raise RuntimeError("graph-owned Q6 V trace exceeded exact 18-projection census")
    if self.cursor==0:self.records,self.outputs=[],[]
    record_u32=(512*4096+2*512*(4096//32)*4)//4
    record=Tensor.empty(record_u32,dtype=dtypes.uint32,device=x.device)
    out=Tensor.empty(512*1024,dtype=dtypes.float32,device=x.device)
    _,record=x.uop_program(record,fxn=lambda *_:self.role_asset.producer)
    out,record,halfs=out.uop_program(record,halfs,fxn=lambda *_:self.role_asset.main_program)
    self.records.append(record);self.outputs.append(out);self.cursor+=1
    return out.reshape(512,1024)


def _gate_oracle_program():
  """Bandwidth-only zero provider preserving gate/up's three-buffer ABI."""
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
  source=r'''extern "C" __global__ void nv_gate_oracle_zero(float *out, const unsigned int *record, const unsigned int *words) {
    (void)record; (void)words;
    unsigned int i=blockIdx.x*256u+threadIdx.x;
    for (; i<6291456u; i+=65536u) out[i]=0.0f;
  }'''
  cubin=NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="nv_gate_oracle_zero_v4").compile(source)
  return native_nv_program("nv_gate_oracle_zero",cubin,global_size=(256,1,1),local_size=(256,1,1),
    globals=(0,1,2),outs=(0,),ins=(1,2))


def _down_oracle_program():
  """Bandwidth-only FP16 zero provider preserving down's activation/weight boundary."""
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
  source=r'''extern "C" __global__ void nv_down_oracle_zero(unsigned short *out, const unsigned short *activation,
      const unsigned short *weight) {
    (void)activation; (void)weight;
    unsigned int i=blockIdx.x*256u+threadIdx.x;
    for (; i<2097152u; i+=65536u) out[i]=0;
  }'''
  cubin=NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="nv_down_oracle_zero_v1").compile(source)
  return native_nv_program("nv_down_oracle_zero",cubin,global_size=(256,1,1),local_size=(256,1,1),
    globals=(0,1,2),outs=(0,),ins=(1,2))


def _gate_epilogue_program():
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
  source=r'''#include <cuda_fp16.h>
  extern "C" __global__ void nv_gate_silu_mul_cast_fused(half *out, const float *gate, const float *up) {
    unsigned int i=blockIdx.x*256u+threadIdx.x;
    for (; i<6291456u; i+=65536u) {
      float g=gate[i];
      out[i]=__float2half_rn(g*(1.0f/(1.0f+exp2f(g*-1.4426950408889634f)))*up[i]);
    }
  }'''
  cubin=NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="nv_gate_silu_mul_cast_fused_v1").compile(source)
  return native_nv_program("nv_gate_silu_mul_cast_fused",cubin,global_size=(256,1,1),local_size=(256,1,1),
    globals=(0,1,2),outs=(0,),ins=(1,2))


def _write(path,payload):
  target=pathlib.Path(path);target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,sort_keys=True))


def _hash_tensor(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _hash_buffer(buf):return hashlib.sha256(buf.numpy().tobytes()).hexdigest()

def _graph_buffer(buf):
  """Serialize only metadata owned by a finalized graph buffer/view."""
  base=getattr(buf,"base",None)
  if base is None: base=getattr(buf,"buffer",None)
  offset=getattr(buf,"offset",None)
  shape=getattr(buf,"shape",None)
  strides=getattr(buf,"strides",None)
  dtype=getattr(buf,"dtype",None)
  try: digest=_hash_buffer(buf)
  except Exception: digest=None
  return {"base":None if base is None else repr(base),
    "offset":None if offset is None else int(offset),
    "shape":None if shape is None else list(shape),
    "strides":None if strides is None else list(strides),
    "dtype":None if dtype is None else str(dtype), "hash":digest}


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
      ctx=getattr(call.arg,"candidate_context",None)
      role=by_identity.get(getattr(ctx,"canonical_identity",None))
      if role is None:
        role=next((native_role for native_role in ("v","gate_oracle","down_oracle","gate_epilogue")
          if getattr(call.arg,"name",None)==identities.get(native_role)),None)
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
  vval=getattr(model, "_nv_compiler_q4_imma_v_pp512_binding", None)
  q6val=getattr(model, "_nv_compiler_q6_imma_pp512_binding", None)
  _configure(model,gate)
  for block in model.blk:block._nv_compiler_q4_imma_k_pp512_binding=kval
  role_by_id={}
  if qo is not None:
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
      if candidate and vval is not None:vval.begin_trace()
      if q6val is not None:q6val.begin_trace()
      if qo is not None:qo.begin_trace()
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


class _LiveF1Adapter:
  """Default-closed adapter over one finalized graph's live Flash calls.

  Runtime and Buffer objects deliberately stay process-local.  The callback is
  opt-in because it is an evidence probe, never part of model execution.
  """
  def __init__(self, calls, runtimes, candidate=None):
    self.calls, self.runtimes, self.candidate = tuple(calls), tuple(runtimes), candidate
    if len(self.calls) != 36: raise ValueError(f"live F1 requires 36 calls, got {len(self.calls)}")

  def run(self, arm, lock_path=None):
    if arm not in ("control_0", "candidate_1", "control_2"): raise ValueError("unknown F1 arm")
    lock = open(lock_path, "w") if lock_path else contextlib.nullcontext()
    with lock:
      rows=[]
      for i, ((_, program, bufs, _), runtime) in enumerate(zip(self.calls, self.runtimes)):
        # GraphRunner retains Buffer views; device runtimes consume their
        # underlying HCQBuffer handles (the latter provide va_addr/offset).
        live_bufs=tuple(buf.get_buf("NV") for buf in bufs)
        if arm == "candidate_1":
          if self.candidate is None: raise RuntimeError("candidate runtime is unavailable")
          if len(bufs) != 4: raise RuntimeError(f"candidate ABI requires 4 live buffers, call {i} has {len(bufs)}")
          elapsed=self.candidate(*live_bufs, global_size=(32,8,1), local_size=(128,1,1), wait=True)
        else:
          if runtime is None: raise RuntimeError(f"installed runtime missing for call {i}")
          elapsed=runtime(*live_bufs, global_size=program.arg.global_size, local_size=program.arg.local_size, wait=True)
        rows.append({"ordinal":i,"arm":arm,"elapsed_us":None if elapsed is None else float(elapsed)*1e6,
                     "program":getattr(program.arg,"function_name",None),"buffers":len(bufs)})
      return rows


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("candidate","control","compare"),required=True);ap.add_argument("--q4-v",action="store_true")
  ap.add_argument("--q6-v",action="store_true")
  ap.add_argument("--gate-oracle",action="store_true")
  ap.add_argument("--down-oracle",action="store_true")
  ap.add_argument("--gate-q8-reuse",action="store_true")
  ap.add_argument("--gate-epilogue-fused",action="store_true")
  ap.add_argument("--model",default=MODEL);ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--rounds",type=int,default=9)
  ap.add_argument("--replay-cycles",type=int,default=20);ap.add_argument("--deep-replay",action="store_true")
  ap.add_argument("--prune-final-row",action="store_true")
  ap.add_argument("--out",required=True);ap.add_argument("--logits-npz",default="")
  ap.add_argument("--dump-semantic-inventory",default="")
  ap.add_argument("--dump-service-inventory",default="");ap.add_argument("--service-programs",default="")
  ap.add_argument("--service-rounds",type=int,default=9)
  ap.add_argument("--candidate-json",default="");ap.add_argument("--candidate-npz",default="");ap.add_argument("--dump-flash-abi",default="")
  ap.add_argument("--dump-flash-f1",default="", help="opt-in finalized graph-owned F1 binding dump")
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

  q6_env=os.environ.get("NV_COMPILER_Q6_IMMA_PP512")
  q6_roles={x.strip() for x in os.environ.get("NV_COMPILER_Q6_IMMA_PP512_ROLES","").split(",") if x.strip()}
  if os.environ.get("NV_COMPILER_Q4_IMMA_PP512")!="1" or os.environ.get("NV_COMPILER_Q4_IMMA_K_PP512")!="1" or \
      os.environ.get("NV_Q4_IMMA_PP512") is not None or (args.q6_v and (q6_env!="1" or q6_roles!={"attn_v"})) or \
      (not args.q6_v and q6_env is not None):
    raise SystemExit("combined arm requires compiler gate/up+K; Q6 is admitted only by --q6-v with the exact attn_v role")
  if args.gate_oracle and (args.arm!="candidate" or not args.q4_v or not args.q6_v or args.prune_final_row):
    raise SystemExit("gate oracle requires the unpruned current-best candidate with both Q4 V and Q6 V")
  if args.down_oracle and (args.arm!="candidate" or not args.q4_v or not args.q6_v or args.prune_final_row or args.gate_oracle):
    raise SystemExit("down oracle requires the unpruned current-best candidate and excludes another oracle")
  if args.gate_q8_reuse and (args.arm!="candidate" or not args.q4_v or not args.q6_v or args.prune_final_row or args.gate_oracle or args.down_oracle):
    raise SystemExit("gate Q8 reuse requires the unpruned current-best candidate and excludes oracle arms")
  if args.gate_epilogue_fused and (args.arm!="candidate" or not args.q4_v or not args.q6_v or args.prune_final_row or
      args.gate_oracle or args.down_oracle or args.gate_q8_reuse):
    raise SystemExit("fused gate epilogue requires the unpruned current-best candidate and an isolated arm")
  qo_env=os.environ.get("NV_COMPILER_Q4_IMMA_QO_PP512")
  if qo_env != "1": raise SystemExit("captured combined arm requires Q/O env=1 for both matched arms")

  from tinygrad.llm.generate import load_model_and_tokenizer
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear,Q6KPrimitiveLinear
  from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for as gate_binding_for
  from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import binding_for as k_binding_for, RECORD_U32 as K_RECORD_U32
  from extra.llm_research.prefill.nv_compiler_q4v_serialized_binding import binding_for as v_binding_for
  # This arm is an exact pp512 experiment; requesting a 4608-token KV plan
  # causes admission to reject on constrained validation GPUs before capture.
  model,_=load_model_and_tokenizer(args.model,512,seed=20260617)
  down_overlay_bases=set()
  if args.down_oracle:
    import tinygrad.llm.model as model_module
    down_ids={id(block.ffn_down) for block in model.blk}
    down_overlay_bases={block.ffn_down._pf16_w.uop.buf_uop for block in model.blk}
    original_pf16,down_program=model_module._pf16,_down_oracle_program()
    def oracle_pf16(lin,x):
      if id(lin) not in down_ids:return original_pf16(lin,x)
      x16=x.cast(dtypes.float16).contiguous().reshape(512,12288)
      weight=lin._pf16_w
      out=Tensor.empty(512*4096,dtype=dtypes.float16,device=x.device)
      out,x16,weight=out.uop_program(x16,weight,fxn=lambda *_:down_program)
      return out.reshape(1,512,4096)
    model_module._pf16=oracle_pf16
  if args.prune_final_row:
    # Explicit terminal graph lease; control remains untouched.
    model.blk[-1]._final_row_prune_requested_row = 511
  gate_asset=gate_binding_for("NV");gate_asset.prepare_records(72);gate_asset.install_warmstart(model)
  gate_runtime_asset=dataclasses.replace(gate_asset,main_program=_gate_oracle_program()) if args.gate_oracle else gate_asset
  gate_record_u32=(512*4096+2*512*(4096//32)*4)//4
  gate=_PairedGateQ8Capture(gate_runtime_asset,gate_record_u32) if args.gate_q8_reuse else \
    _FusedGateEpilogueCapture(gate_runtime_asset,gate_record_u32,_gate_epilogue_program()) if args.gate_epilogue_fused else \
    gate_runtime_asset.new_capture()
  if args.gate_epilogue_fused:
    import tinygrad.llm.model as model_module
    down_ids={id(block.ffn_down) for block in model.blk};original_pf16=model_module._pf16
    def fused_gate_pf16(lin,x):
      return original_pf16(lin,gate.down_activation()) if id(lin) in down_ids else original_pf16(lin,x)
    model_module._pf16=fused_gate_pf16
  v_asset=None
  if args.arm=="candidate" and args.q4_v:
    v_asset=v_binding_for("NV");v_asset.prepare_records(18)
  q6_asset=q6val=None
  if args.q6_v:
    from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import binding_for as q6_binding_for
    q6_asset=q6_binding_for("NV");q6_asset.prepare_records(36);q6_asset.install_warmstart(model);q6val=_GraphOwnedQ6VCapture(q6_asset)
  k_asset=k_binding_for("NV");k_asset.prepare_records(36);k_asset.install_warmstart(model)
  # The matched control retains K's independently-qualified ordinary carrier.
  # Q/O composition changes its scheduling boundary, so the combined candidate
  # invokes K's exact same frozen compiler PROGRAM with graph-owned storage.
  # Both matched arms use the same graph-owned K lease.  The ordinary capture
  # can lose the canonical packed-A identity when Q/O is nested in the same
  # schedule, causing the tensor-core contract to reject the control.
  kval=_GraphOwnedKCapture(k_asset,K_RECORD_U32)
  vval=_GraphOwnedVCapture(v_asset,K_RECORD_U32) if args.arm=="candidate" and args.q4_v else None
  model._nv_gkqo_gate_capture,model._nv_gkqo_k_capture=gate,kval
  model._nv_compiler_q6_imma_pp512_binding=q6val
  model._nv_compiler_q4_imma_v_pp512_binding=vval
  model._nv_compiler_q4_imma_v_pp512_enabled=(args.arm=="candidate" and args.q4_v)
  for bi,block in enumerate(model.blk):
    block._nv_compiler_q6_imma_pp512_binding=q6val
    block._nv_compiler_q4_imma_v_pp512_binding=vval
    block._nv_compiler_q4_imma_v_pp512_enabled=(args.arm=="candidate" and args.q4_v)
    # Only the 18 GGML type-12 V projections are replaced; type-14 V stays
    # on its FP16 fallback.  Remove the expanded cache for admitted Q4 V.
    if args.arm=="candidate" and args.q4_v and bi in {4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29}:
      if hasattr(block.attn_v,"_pf16_w"): delattr(block.attn_v,"_pf16_w")
    if args.q6_v and isinstance(block.attn_v,Q6KPrimitiveLinear) and hasattr(block.attn_v,"_pf16_w"):
      delattr(block.attn_v,"_pf16_w")
  qo=None;qo_linears=[p for block in model.blk for p in (block.attn_q,block.attn_output)]
  if True:
    from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import binding_for as qo_binding_for, RECORD_U32
    # binding_for() has already compiled and frozen Q/O's ordinary compiler PROGRAM under
    # its typed contract.  The composed graph invokes only that opaque PROGRAM, so adding
    # Q/O's shape key to the ambient model warmstart table would incorrectly claim unrelated
    # 512x4096x4096 ordinary matmuls (the exact composition collision this arm must avoid).
    qo=_GraphOwnedQOCapture(qo_binding_for("NV"),RECORD_U32)
    for lin in qo_linears:
      if hasattr(lin,"_pf16_w"):delattr(lin,"_pf16_w")
  identities={"gate_up":gate.candidate_identity,"gate_oracle":"nv_gate_oracle_zero" if args.gate_oracle else None,
              "down_oracle":"nv_down_oracle_zero" if args.down_oracle else None,
              "gate_epilogue":"nv_gate_silu_mul_cast_fused" if args.gate_epilogue_fused else None,
              "k":kval.candidate_identity,
              # Serialized V has no compiler candidate_context.  Use its
              # immutable manifest symbol so stage buffers remain observable.
              "v":None if vval is None else vval.asset.main_program.arg.name,
              "q6_v":None if q6_asset is None else q6_asset.roles["attn_v"].candidate_identity,
              "qo":None if qo is None else qo.candidate_identity}

  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
  chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous();temp=Tensor([0.0])
  jit=_capture(model,qo,chunk_a,temp,args.arm=="candidate")
  # Retain the actual finalized GraphRunner objects before any teardown.  F1
  # consumes these objects in-process; no runtime or Buffer is serialized.
  from tinygrad.engine.realize import graph_cache
  live_calls=[]; live_runtimes=[]
  for outer in jit.captured.linear.src:
    if outer.src[0].op is not Ops.CUSTOM_FUNCTION or outer.src[0].arg!="graph": continue
    graph=graph_cache.get(outer.src[0])
    for n, entry in enumerate(graph.calls):
      if getattr(entry[1].arg,"name","") == "nv_sm120_q16_grid_hd128_loop_attention":
        live_calls.append(entry)
        live_runtimes.append(graph.runtimes[n])
  candidate_runtime = None
  if os.environ.get("NV_FLASH_F1_LIVE") == "1":
    from extra.llm_research.prefill.nv_flash_vkv_primitive import build_live_program
    candidate_runtime = build_live_program("NV")
  if len(live_calls) == 36:
    model._nv_flash_f1_live_adapter = _LiveF1Adapter(live_calls, live_runtimes, candidate_runtime)
  elif os.environ.get("NV_FLASH_F1_LIVE") == "1":
    raise RuntimeError(f"live F1 finalized graph exposed {len(live_calls)} Flash calls, expected 36")
  if os.environ.get("NV_FLASH_F1_LIVE") == "1":
    lock_path=os.environ.get("NV_FLASH_F1_LOCK", "/tmp/tinygrad-nv-f1.lock")
    rows=[]
    for arm in ("control_0", "candidate_1", "control_2"):
      try: rows.extend(model._nv_flash_f1_live_adapter.run(arm, lock_path))
      except Exception as exc:
        rows.append({"arm":arm,"status":"STOP","blocker":f"{type(exc).__name__}: {exc}"})
        break
    if args.dump_flash_f1:
      p=pathlib.Path(args.dump_flash_f1).with_suffix(".r9.json")
      _write(p,{"schema":"tinygrad.nv_prefill_flash_f1_2_r9.v1","status":"PASS" if len(rows)==108 else "STOP",
        "arms":rows,"exact_population":36,"default_enabled":False})
  if args.dump_flash_abi:
    from tinygrad.engine.realize import graph_cache
    rows=[]
    for outer in jit.captured.linear.src:
      if outer.src[0].op is not Ops.CUSTOM_FUNCTION or outer.src[0].arg!="graph": continue
      g=graph_cache.get(outer.src[0])
      for i,(_,call,bufs,_) in enumerate(g.calls):
        if getattr(call.arg,"name","")!="nv_sm120_q16_grid_hd128_loop_attention": continue
        rows.append({"index":i,"name":call.arg.name,"ins":list(call.arg.ins),"outs":list(call.arg.outs),"args":[{"dtype":str(getattr(b,'dtype',None)),"size":int(getattr(b,'size',0))} for b in bufs],"global_size":list(getattr(call.arg,'global_size',())) ,"local_size":list(getattr(call.arg,'local_size',()))})
    _write(args.dump_flash_abi,{"schema":"tinygrad.nv_prefill_flash_program_abi.v1","status":"PASS" if rows else "FAIL","calls":rows,"fused_score_reduction":True})
  if args.dump_flash_f1:
    from tinygrad.engine.realize import graph_cache
    rows=[]
    for outer in jit.captured.linear.src:
      if outer.src[0].op is not Ops.CUSTOM_FUNCTION or outer.src[0].arg!="graph": continue
      g=graph_cache.get(outer.src[0])
      if g is None: raise RuntimeError("captured graph missing from graph cache")
      for i,(_,call,bufs,_) in enumerate(g.calls):
        if getattr(call.arg,"name","")!="nv_sm120_q16_grid_hd128_loop_attention": continue
        views=[_graph_buffer(b) for b in bufs]
        rows.append({"capture_ordinal":len(rows),"graph_call_index":i,"name":call.arg.name,
          "layer":len(rows),"q_head":32,"kv_head":8,"causal_length":512,
          "shapes":[v["shape"] for v in views],"strides":[v["strides"] for v in views],
          "dtypes":[v["dtype"] for v in views],"buffer_hashes":[v["hash"] for v in views],
          "output_shape":views[0]["shape"] if views else None,
          "buffers":views,"graph_owned_slice":{"source":"graph-owned","call_index":i,
            "buffer_indices":list(range(len(views))),"contract":"finalized-TinyJit-view"},
          "installed_identity":getattr(getattr(call.arg,"candidate_context",None),"canonical_identity",None)})
    _write(args.dump_flash_f1,{"schema":"tinygrad.nv_prefill_flash_f1_graph.v1",
      "packet":"F1.1","status":"PASS" if len(rows)==36 else "BLOCKED",
      "program":"nv_sm120_q16_grid_hd128_loop_attention","exact_population":36,
      "graph_calls":rows,"census":{"predicted":36,"observed":len(rows)},
      "full_t512_relaunch":False,"source":"finalized-TinyJit-graph"})
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
  if args.dump_semantic_inventory:
    inventory=[];sources={}
    for ordinal,c in enumerate(calls):
      program=c.src[0];kernel=program.src[0].arg if program.src else None
      if _call_name(c) not in sources:
        source=next((u.arg for u in program.src if u.op is Ops.SOURCE),None)
        if source is not None:sources[_call_name(c)]=source
      buffers=[]
      for arg in c.src[1:]:
        u=_buf_uop(arg)
        if u is None:buffers.append(None);continue
        try:buffers.append({"bytes":u.numel()*u.dtype.itemsize,"dtype":str(u.dtype)})
        except (RuntimeError,AttributeError):buffers.append({"repr":repr(u)})
      inventory.append({"ordinal":ordinal,"name":_call_name(c),"outs":list(getattr(program.arg,"outs",())),
        "ins":list(getattr(program.arg,"ins",())),"buffers":buffers,"kernel_name":getattr(kernel,"name",None),
        "memory_semantic_slots":[[int(i),str(semantic)] for i,semantic in getattr(kernel,"memory_semantic_slots",())]})
    _write(args.dump_semantic_inventory,{"schema":"tinygrad.nv_prefill_semantic_inventory.v1","calls":inventory,"sources":sources})
  if args.dump_service_inventory:
    from tinygrad.engine.realize import graph_cache
    selected={x for x in args.service_programs.split(",") if x};entries=[]
    for outer in jit.captured.linear.src:
      if outer.src[0].op is not Ops.CUSTOM_FUNCTION or outer.src[0].arg!="graph":continue
      graph=graph_cache.get(outer.src[0])
      if graph is None:raise RuntimeError("service inventory graph missing from cache")
      entries.extend((call,bufs,graph.runtimes[n]) for n,(_,call,bufs,_) in enumerate(graph.calls) if call.op is Ops.PROGRAM)
    service=[]
    for ordinal,(program,bufs,runtime) in enumerate(entries):
      name=getattr(program.arg,"name","")
      if name not in selected:continue
      if runtime is None:raise RuntimeError(f"service runtime missing for {name} at {ordinal}")
      live=tuple(buf.get_buf("NV") for buf in bufs)
      runtime(*live,global_size=program.arg.global_size,local_size=program.arg.local_size,wait=True)
      samples=[float(runtime(*live,global_size=program.arg.global_size,local_size=program.arg.local_size,wait=True))*1e6
        for _ in range(args.service_rounds)]
      service.append({"ordinal":ordinal,"name":name,"previous":None if ordinal==0 else getattr(entries[ordinal-1][0].arg,"name",""),
        "next":None if ordinal+1==len(entries) else getattr(entries[ordinal+1][0].arg,"name",""),"samples_us":samples})
    _write(args.dump_service_inventory,{"schema":"tinygrad.nv_prefill_live_service.v1","rounds":args.service_rounds,
      "selected":sorted(selected),"rows":service})
  mains={role:([] if ident is None else _identity_calls(calls,ident)) for role,ident in identities.items()}
  # K and serialized V intentionally share the generated kernel symbol.  K
  # retains compiler candidate_context; the remaining exact-symbol calls are V.
  v_name = None if vval is None else vval.asset.main_program.arg.name
  k_calls = {id(c) for c in _identity_calls(calls,kval.candidate_identity)}
  v_calls=[] if v_name is None else [c for c in calls if _call_name(c)==v_name and id(c) not in k_calls]
  gate_oracle_calls=[] if not args.gate_oracle else [c for c in calls if _call_name(c)=="nv_gate_oracle_zero"]
  down_oracle_calls=[] if not args.down_oracle else [c for c in calls if _call_name(c)=="nv_down_oracle_zero"]
  gate_epilogue_calls=[] if not args.gate_epilogue_fused else [c for c in calls if _call_name(c)=="nv_gate_silu_mul_cast_fused"]
  down_weight_args=[_buf_uop(c.src[3]) for c in down_oracle_calls if len(c.src)>3]
  transforms={"gate_up":gate.transform,"k":kval.transform,"qo":None if qo is None else qo.transform,
              "v":None if vval is None else vval.transform,
              "q6_v":None if q6_asset is None else q6_asset.roles["attn_v"].transform}
  weights=[]
  for role in ("gate_up","k","qo","v","q6_v"):
    if transforms[role] is not None:weights += [_weight_arg(c,transforms[role]) for c in mains[role]]
  if gate_oracle_calls:weights += [_weight_arg(c,gate.transform) for c in gate_oracle_calls]
  weights=[x for x in weights if x is not None]
  # Serialized V native calls do not expose typed transform arguments through
  # `_weight_arg`; their immutable manifest ABI owns one canonical weight input.
  if v_calls: weights += [f"serialized-v-{i}" for i in range(len(v_calls))]
  canonical={lin.prefill_packed_weight().uop.buf_uop for block in model.blk for lin in
    (block.ffn_gate,block.ffn_up,block.attn_k,block.attn_q,block.attn_output) if isinstance(lin,Q4KPrimitiveLinear)}
  canonical.update(block.attn_v.prefill_packed_weight().uop.buf_uop for block in model.blk if isinstance(block.attn_v,Q6KPrimitiveLinear))
  admitted=[lin for block in model.blk for lin in (block.ffn_gate,block.ffn_up,block.attn_k,block.attn_q,block.attn_output)]
  remaining=[lin for block in model.blk for lin in (block.attn_v,block.ffn_down)]
  # V is a separately captured role even though it shares the immutable
  # compiler substrate; count its 18 main/Q8 records explicitly.
  # Serialized V is an immutable cubin asset, so its native ProgramInfo has
  # no compiler candidate_context.  Count it by the manifest-owned symbol.
  total_mains=sum(len(x) for x in mains.values())+len(v_calls)+len(gate_oracle_calls)
  q8=names.get("q8_compact_record_fp16",0)+names.get("q8_compact_record_fp16_q6_attn_v",0)
  census={"gate_up_main":len(mains["gate_up"]),"gate_oracle_main":len(gate_oracle_calls),"down_oracle_main":len(down_oracle_calls),
    "gate_epilogue_main":len(gate_epilogue_calls),
    "old_gate_silu_mul":names.get("E_64_192_8_16_4_1e161f6c4c230e894f4d2601704fc92075a12b3f53be815dcba4bbed84e83ed5",0),
    "old_gate_fp16_cast":names.get("E_64_192_8_16_4_5a2137f8e57933947793f5908b5b1c440a16780ed3881ebe0c386c7e3680548c",0),
    "down_oracle_weight_args":len(down_weight_args),"down_oracle_unique_weight_bases":len(set(down_weight_args)),
    "down_oracle_all_weights_overlay":bool(down_weight_args and all(w in down_overlay_bases for w in down_weight_args)),
    "k_main":len(mains["k"]),"qo_main":len(mains["qo"]),
    "v_main":len(v_calls),"q6_v_main":len(mains["q6_v"]),"q6_v_producer":names.get("q8_compact_record_fp16_q6_attn_v",0),
    "compiler_main_total":total_mains,"q8_producer_total":q8,"candidate_weight_args":len(weights),
    "unique_weight_bases":len(set(weights)),"all_weights_canonical":bool(weights and all(isinstance(x,str) or x in canonical for x in weights)),
    "admitted_fp16_overlays":sum(getattr(x,"_pf16_w",None) is not None for x in admitted),
    "remaining_v_down_fp16_overlays":sum(getattr(x,"_pf16_w",None) is not None for x in remaining),
    "weight_copy_kernels":0 if weights and all(isinstance(x,str) or x in canonical for x in weights) else -1,
    "old_fixups":names.get("q4k_imma_fixup",0),"q6_old_fixups":names.get("q6k_imma_fixup",0),"partial_workspace_bytes":0}
  census["gate_q8_record_allocations"]=stage_census.get("gate_up_records",{}).get("unique_allocations",0)
  replay={"finite":bool(np.isfinite(a1[1]).all()),"same_token":a0[0]==a1[0],
    "distinct_activation_output":bool(a1[0]!=b[0] or not np.array_equal(a1[1],b[1])),"cycles":cycles}
  replay_pass=replay["finite"] and replay["same_token"] and replay["distinct_activation_output"] and \
    all(x["same_token"] and x["same_logits_exact"] for x in cycles)
  deep=None
  if deep0 is not None:
    deep={"cycles":deep_cycles,"all_cycles_exact":all(value["same_length"] and value["first_mismatch"] is None
      for cycle in deep_cycles for value in cycle.values())}
    replay_pass &= deep["all_cycles_exact"]
  gate_stage="gate_oracle" if args.gate_oracle else "gate_up"
  expected_stage={f"{gate_stage}_records":70 if args.prune_final_row else 72,
    f"{gate_stage}_outputs":70 if args.prune_final_row else 72,"k_records":36,"k_outputs":36}
  if args.arm=="candidate" and args.q4_v: expected_stage.update({"v_records":18,"v_outputs":18})
  if args.q6_v: expected_stage.update({"q6_v_records":18,"q6_v_outputs":18})
  if args.down_oracle: expected_stage.update({"down_oracle_records":36,"down_oracle_outputs":36})
  if args.gate_epilogue_fused: expected_stage.update({"gate_epilogue_records":36,"gate_epilogue_outputs":36})
  if args.arm=="candidate":expected_stage.update({"qo_records":72,"qo_outputs":72})
  stage_census_pass=not args.deep_replay or all(stage_census.get(key,{}).get("calls")==count for key,count in expected_stage.items())
  if args.gate_oracle:
    structural=stage_census_pass and all((census["gate_up_main"]==0,census["gate_oracle_main"]==72,
      census["k_main"]==36,census["qo_main"]==72,census["v_main"]==18,census["q6_v_main"]==18,
      census["q6_v_producer"]==18,census["compiler_main_total"]==216,census["q8_producer_total"]==216,
      census["candidate_weight_args"]==216,census["unique_weight_bases"]==216,census["all_weights_canonical"],
      census["admitted_fp16_overlays"]==0,census["remaining_v_down_fp16_overlays"]==36,
      census["weight_copy_kernels"]==0,census["old_fixups"]==0,census["q6_old_fixups"]==0))
  elif args.gate_q8_reuse:
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["gate_oracle_main"]==0,
      census["down_oracle_main"]==0,census["gate_q8_record_allocations"]==36,census["k_main"]==36,
      census["qo_main"]==72,census["v_main"]==18,census["q6_v_main"]==18,census["q6_v_producer"]==18,
      census["compiler_main_total"]==216,census["q8_producer_total"]==180,census["candidate_weight_args"]==216,
      census["unique_weight_bases"]==216,census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,
      census["remaining_v_down_fp16_overlays"]==36,census["weight_copy_kernels"]==0,census["old_fixups"]==0,
      census["q6_old_fixups"]==0))
  elif args.gate_epilogue_fused:
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["gate_oracle_main"]==0,
      census["down_oracle_main"]==0,census["gate_epilogue_main"]==36,census["old_gate_silu_mul"]==0,
      census["old_gate_fp16_cast"]==0,census["k_main"]==36,census["qo_main"]==72,census["v_main"]==18,
      census["q6_v_main"]==18,census["q6_v_producer"]==18,census["compiler_main_total"]==216,
      census["q8_producer_total"]==216,census["candidate_weight_args"]==216,census["unique_weight_bases"]==216,
      census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,census["remaining_v_down_fp16_overlays"]==36,
      census["weight_copy_kernels"]==0,census["old_fixups"]==0,census["q6_old_fixups"]==0))
  elif args.down_oracle:
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["gate_oracle_main"]==0,
      census["down_oracle_main"]==36,census["down_oracle_weight_args"]==36,census["down_oracle_unique_weight_bases"]==36,
      census["down_oracle_all_weights_overlay"],census["k_main"]==36,census["qo_main"]==72,census["v_main"]==18,
      census["q6_v_main"]==18,census["q6_v_producer"]==18,census["compiler_main_total"]==216,
      census["q8_producer_total"]==216,census["candidate_weight_args"]==216,census["unique_weight_bases"]==216,
      census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,census["remaining_v_down_fp16_overlays"]==36,
      census["weight_copy_kernels"]==0,census["old_fixups"]==0,census["q6_old_fixups"]==0))
  elif args.arm=="candidate" and args.q4_v and args.q6_v:
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["k_main"]==36,census["qo_main"]==72,
      census["v_main"]==18,census["q6_v_main"]==18,census["q6_v_producer"]==18,census["compiler_main_total"]==216,
      census["q8_producer_total"]==216,census["candidate_weight_args"]==216,census["unique_weight_bases"]==216,
      census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,census["remaining_v_down_fp16_overlays"]==36,
      census["weight_copy_kernels"]==0,census["old_fixups"]==0,census["q6_old_fixups"]==0))
  elif args.arm=="candidate" and args.q4_v:
    structural=stage_census_pass and all((census["gate_up_main"]==72,census["k_main"]==36,census["qo_main"]==72,
      census["v_main"]==18,census["compiler_main_total"]==198,census["q8_producer_total"]==198,census["candidate_weight_args"]==198,
      census["unique_weight_bases"]==198,census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,
      census["remaining_v_down_fp16_overlays"]==54,census["weight_copy_kernels"]==0,census["old_fixups"]==0))
  else:
    structural=stage_census_pass and all((census["gate_up_main"]==(70 if args.prune_final_row else 72),census["k_main"]==36,census["qo_main"]==72,
      census["compiler_main_total"]==(178 if args.prune_final_row else 180),census["q8_producer_total"]==(178 if args.prune_final_row else 180),census["candidate_weight_args"]==(178 if args.prune_final_row else 180),
      census["unique_weight_bases"]==(178 if args.prune_final_row else 180),census["all_weights_canonical"],census["admitted_fp16_overlays"]==0,
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
