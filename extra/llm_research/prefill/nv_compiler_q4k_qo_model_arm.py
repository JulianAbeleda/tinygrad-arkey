#!/usr/bin/env python3
"""Fresh-process default-off full-model gate for compiler-packed Q/O."""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, json, os, pathlib, statistics, time
from collections import Counter
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt.postrange import warmstart_candidate_state
from tinygrad.device import Compiled
from tinygrad.llm.prefill_route_observer import prefill_route_scope
from tinygrad.llm.prefill_routes import prefill_route_override
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import _call_and_sync,_call_name,_configure,_numpy_output,_program_calls

MODEL="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _write(path,payload):
  target=pathlib.Path(path);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps(payload,sort_keys=True))


def _profile(events):
  from collections import defaultdict
  from tinygrad.device import ProfileGraphEvent
  from tinygrad.helpers import ProfileRangeEvent
  by_name=defaultdict(list)
  for event in events:
    if isinstance(event,ProfileRangeEvent) and event.en is not None:by_name[str(event.name)].append(float(event.en-event.st)*1e-3)
    elif isinstance(event,ProfileGraphEvent):
      for ent in event.ents:by_name[str(ent.name)].append(float(event.sigs[ent.en_id]-event.sigs[ent.st_id])*1e-3)
  return {name:{"calls":len(vals),"device_ms":sum(vals)} for name,vals in sorted(by_name.items()) if vals}


def _hash_tensor(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _deep_stage_snapshot(model,binding):
  return {"records":[_hash_tensor(t) for t in binding.records],"outputs":[_hash_tensor(t) for t in binding.outputs],
    "kv":[_hash_tensor(block.cache_kv[:,:,:,:512,:]) for block in model.blk]}


def _reset_kv(model):
  for block in model.blk:
    live=block.cache_kv[:,:,:,:512,:]
    live.assign(Tensor.zeros(*live.shape,dtype=live.dtype,device=live.device)).realize()
  Device[Device.DEFAULT].synchronize()


def _graph_buffer_audit(jit):
  """Map each Flash argument to the graph-local writers of its allocation."""
  from tinygrad.engine.realize import graph_cache
  rows=[]
  for graph_no,call in enumerate(jit.captured.linear.src):
    if call.src[0].op is not Ops.CUSTOM_FUNCTION or call.src[0].arg!="graph":continue
    graph=graph_cache.get(call.src[0])
    if graph is None:continue
    def span(buf):
      hb=buf._buf
      try:return int(hb.va_addr),int(hb.va_addr)+int(hb.size)
      except (TypeError,ValueError):return None
    for j,runtime in enumerate(graph.runtimes):
      if runtime is None or runtime.name!="nv_sm120_q16_grid_hd128_loop_attention":continue
      args=[]
      for pos,buf in enumerate(graph.calls[j][2]):
        target=span(buf);writers=[]
        if target is not None:
          for wi in range(j):
            wrt=graph.runtimes[wi]
            if wrt is None:continue
            for out_pos in graph.calls[wi][1].arg.outs:
              other=span(graph.calls[wi][2][out_pos])
              if other is not None and max(target[0],other[0])<min(target[1],other[1]):
                writers.append({"index":wi,"name":wrt.name,"arg":out_pos,
                  "queue":graph.compute_queues[wrt.dev].index(graph.ji_schedule[wi][1]),"span":other})
        args.append({"arg":pos,"span":target,"writers":writers})
      window=[]
      for wi in range(max(0,j-10),j):
        wrt=graph.runtimes[wi]
        if wrt is None:continue
        window.append({"index":wi,"name":wrt.name,"queue":graph.compute_queues[wrt.dev].index(graph.ji_schedule[wi][1]),
          "outs":list(graph.calls[wi][1].arg.outs),"args":[span(x) for x in graph.calls[wi][2]]})
      rows.append({"graph":graph_no,"attention_index":j,"queue":graph.compute_queues[runtime.dev].index(graph.ji_schedule[j][1]),
        "deps":graph.prof_graph_deps[j] if graph.prof_graph_deps else [],"args":args,"preceding_window":window})
  return rows


def _capture(model,binding,chunk,temp,candidate):
  _configure(model,None)
  @TinyJit
  def run(tokens,temperature):return model.forward_greedy_with_logits(tokens,0,temperature)
  role_by_id={}
  if candidate:
    for block in model.blk:
      role_by_id[id(block.attn_q)]="attn_q";role_by_id[id(block.attn_output)]="attn_output"
    def route(lin,x):
      if (role:=role_by_id.get(id(lin))) is None:return None
      if tuple(x.shape)!=(1,512,4096):raise RuntimeError(f"unexpected Q/O activation {x.shape}")
      # Match route_prefill_linear's ordinary FP16 boundary. Q normally arrives
      # as FP16; Flash's O input may be FP32 and must cross the same explicit
      # cast before the exact compact-Q8 producer.
      x16=x.cast(dtypes.float16).contiguous()
      return binding.project(x16.reshape(512,4096),lin.prefill_packed_weight(),model_family="qwen3_8b",role=role).reshape(1,512,4096)
    override=prefill_route_override(route)
  else:override=contextlib.nullcontext()
  merged={**(model._pf16_warmstart or {}),**(model._packed_wmma_warmstart or {}) }
  with override,prefill_route_scope(True),warmstart_candidate_state(merged,model._packed_wmma_warmstart_contexts):
    for _ in range(3):
      if candidate:binding.begin_trace()
      _call_and_sync(run,chunk,temp)
  if run.captured is None:raise RuntimeError("Q/O whole-model arm did not capture")
  return run


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("candidate","fp16","compare"),required=True)
  ap.add_argument("--model",default=MODEL);ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--rounds",type=int,default=9)
  ap.add_argument("--out",required=True);ap.add_argument("--logits-npz",default="")
  ap.add_argument("--deep-replay",action="store_true")
  ap.add_argument("--reset-kv",action="store_true")
  ap.add_argument("--replay-cycles",type=int,default=1,
    help="number of B,A replay pairs checked against the first A result")
  ap.add_argument("--buffer-audit",default="",help="optional JSON path for Flash argument/writer allocation audit")
  ap.add_argument("--candidate-json",default="");ap.add_argument("--candidate-npz",default="")
  ap.add_argument("--fp16-json",default="");ap.add_argument("--fp16-npz",default="");args=ap.parse_args()
  if args.arm=="compare":
    cj,fj=json.loads(pathlib.Path(args.candidate_json).read_text()),json.loads(pathlib.Path(args.fp16_json).read_text())
    ca,fa=np.load(args.candidate_npz),np.load(args.fp16_npz);cl,fl=ca["logits"].astype(np.float32),fa["logits"].astype(np.float32)
    diff=np.abs(cl-fl);quality={"candidate_token":int(ca["token"]),"fp16_token":int(fa["token"]),
      "same_token":int(ca["token"])==int(fa["token"]),"finite":bool(np.isfinite(cl).all() and np.isfinite(fl).all()),
      "max_abs":float(diff.max()),"mean_abs":float(diff.mean()),"allclose_rtol_0p02_atol_0p5":bool(np.allclose(cl,fl,rtol=.02,atol=.5))}
    performance=cj["wall"]["min_ms"]<fj["wall"]["min_ms"]
    passed=cj["status"]==fj["status"]=="PASS" and all(quality[k] for k in ("same_token","finite","allclose_rtol_0p02_atol_0p5")) and performance
    payload={"schema":"tinygrad.nv_compiler_q4k_qo_model_compare.v1","status":"PASS" if passed else "NO_GO",
      "correctness":quality,"performance_pass":performance,"wall":{"candidate_min_ms":cj["wall"]["min_ms"],
      "fp16_min_ms":fj["wall"]["min_ms"],"candidate_minus_fp16_ms":cj["wall"]["min_ms"]-fj["wall"]["min_ms"],
      "candidate_tok_s":512/cj["wall"]["min_ms"]*1000,"fp16_tok_s":512/fj["wall"]["min_ms"]*1000}}
    _write(args.out,payload);return

  enabled=os.environ.get("NV_COMPILER_Q4_IMMA_QO_PP512")
  if args.arm=="candidate" and enabled!="1":raise SystemExit("candidate requires NV_COMPILER_Q4_IMMA_QO_PP512=1")
  if args.arm=="fp16" and enabled is not None:raise SystemExit("fp16 requires Q/O research env unset")
  if any(os.environ.get(k) is not None for k in ("NV_COMPILER_Q4_IMMA_PP512","NV_Q4_IMMA_PP512")):
    raise SystemExit("Q/O arm must remain separate from gate/up research bindings")
  from tinygrad.llm.generate import load_model_and_tokenizer
  model,_=load_model_and_tokenizer(args.model,4608,seed=20260617);binding=None
  qo_linears=[p for block in model.blk for p in (block.attn_q,block.attn_output)]
  if args.arm=="candidate":
    from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import binding_for
    binding=binding_for("NV");binding.prepare(72);binding.install_warmstart(model)
    for lin in qo_linears:
      if hasattr(lin,"_pf16_w"):delattr(lin,"_pf16_w")
    gc.collect()
  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
  chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous();temp=Tensor([0.0])
  jit=_capture(model,binding,chunk_a,temp,args.arm=="candidate")
  if args.buffer_audit:_write(args.buffer_audit,{"schema":"tinygrad.nv_qo_flash_buffer_audit.v1","rows":_graph_buffer_audit(jit)})
  if args.reset_kv:_reset_kv(model)
  a0=_numpy_output(_call_and_sync(jit,chunk_a,temp))
  deep0=_deep_stage_snapshot(model,binding) if args.deep_replay and binding is not None else None
  probe_ids=(0,1,70,71)
  state0=None if binding is None else ([binding.records[i].numpy().copy() for i in probe_ids],
    [binding.outputs[i].numpy().copy() for i in probe_ids])
  replay_cycles=[];b=None;a1=None
  for _ in range(args.replay_cycles):
    b=_numpy_output(_call_and_sync(jit,chunk_b,temp))
    if args.reset_kv:_reset_kv(model)
    a1=_numpy_output(_call_and_sync(jit,chunk_a,temp))
    cycle_diff=np.abs(a0[1]-a1[1])
    replay_cycles.append({"same_token":a0[0]==a1[0],"same_activation_exact":bool(np.array_equal(a0[1],a1[1])),
      "max_abs":float(cycle_diff.max()),"mean_abs":float(cycle_diff.mean())})
  assert b is not None and a1 is not None
  deep1=_deep_stage_snapshot(model,binding) if args.deep_replay and binding is not None else None
  state1=None if binding is None else ([binding.records[i].numpy().copy() for i in probe_ids],
    [binding.outputs[i].numpy().copy() for i in probe_ids])
  if args.logits_npz:
    target=pathlib.Path(args.logits_npz);target.parent.mkdir(parents=True,exist_ok=True);np.savez(target,token=np.int64(a1[0]),logits=a1[1])
  for _ in range(args.warmups):_call_and_sync(jit,chunk_a,temp)
  profile_start=len(Compiled.profile_events);samples=[]
  for _ in range(args.rounds):
    Device[Device.DEFAULT].synchronize();started=time.perf_counter_ns();_call_and_sync(jit,chunk_a,temp);samples.append((time.perf_counter_ns()-started)/1e6)
  calls=_program_calls(jit.captured.linear);names=Counter(_call_name(c) for c in calls)
  replay_diff=np.abs(a0[1]-a1[1])
  replay={"finite":bool(np.isfinite(a1[1]).all()),"same_token":a0[0]==a1[0],
    "same_activation_exact":bool(a0[0]==a1[0] and np.array_equal(a0[1],a1[1])),
    "same_activation_allclose":bool(a0[0]==a1[0] and np.allclose(a0[1],a1[1],rtol=2e-5,atol=2e-3)),
    "same_activation_max_abs":float(replay_diff.max()),"same_activation_mean_abs":float(replay_diff.mean()),
    "distinct_activation_output":bool(a1[0]!=b[0] or not np.array_equal(a1[1],b[1])),"cycles":replay_cycles}
  replay_pass=all(replay[k] for k in ("finite","same_token","same_activation_allclose","distinct_activation_output")) and \
    all(x["same_token"] and x["same_activation_exact"] for x in replay_cycles)
  common={"wall":{"samples_ms":samples,"min_ms":min(samples),"median_ms":statistics.median(samples),"tok_s":512/min(samples)*1000},
    "replay":replay,"stage_replay":None if state0 is None else {"probe_ids":probe_ids,
      "q8_records_exact_by_id":[bool(np.array_equal(a,b)) for a,b in zip(state0[0],state1[0])],
      "q8_record_changed_u32_by_id":[int(np.count_nonzero(a!=b)) for a,b in zip(state0[0],state1[0])],
      "main_outputs_exact_by_id":[bool(np.array_equal(a,b)) for a,b in zip(state0[1],state1[1])],
      "main_output_max_abs_by_id":[float(np.max(np.abs(a-b))) for a,b in zip(state0[1],state1[1])]},
    "device_profile":_profile(list(Compiled.profile_events[profile_start:])),"program_names":dict(sorted(names.items())),"token":a1[0]}
  if deep0 is not None:
    common["deep_replay"]={key:{"exact_by_index":[a==b for a,b in zip(deep0[key],deep1[key])],
      "first_mismatch":next((i for i,(a,b) in enumerate(zip(deep0[key],deep1[key])) if a!=b),None)} for key in deep0}
  if args.arm=="fp16":
    overlays=sum(getattr(p,"_pf16_w",None) is not None for p in qo_linears);passed=replay_pass and overlays==72 and names.get("q8_compact_record_fp16",0)==0
    payload={"schema":"tinygrad.nv_compiler_q4k_qo_model_arm.v1","arm":"fp16","status":"PASS" if passed else "FAIL",
      "qo_fp16_overlays":overlays,**common}
  else:
    identity=binding.candidate_identity;mains=[c for c in calls if c.src[0].src and
      getattr(c.src[0].src[0].arg,"candidate_context",None) is not None and c.src[0].src[0].arg.candidate_context.canonical_identity==identity]
    canonical={p.prefill_packed_weight().uop.buf_uop for p in qo_linears};weight_args=[];expanded=0
    for call in mains:
      for arg in call.src[1:]:
        try:size=arg.buf_uop.numel()*arg.buf_uop.dtype.itemsize
        except (RuntimeError,AttributeError):continue
        if arg.dtype==binding.transform.storage_dtype and size==4096*(4096//256)*36*4:weight_args.append(arg.buf_uop)
        if arg.dtype.itemsize==2 and size==4096*4096*2:expanded+=size
    census={"q8_producer":names.get(binding.producer.arg.name,0),"compiler_generated_main":len(mains),
      "old_fixup":names.get("q4k_imma_fixup",0),"candidate_weight_args":len(weight_args),"unique_weight_bases":len(set(weight_args)),
      "all_weights_canonical":bool(weight_args and all(w in canonical for w in weight_args)),"expanded_fp16_weight_bytes":expanded,
      "qo_fp16_overlays":sum(getattr(p,"_pf16_w",None) is not None for p in qo_linears),"q8_records":len(binding.records),"partial_workspace_bytes":0}
    passed=replay_pass and all((census["q8_producer"]==72,census["compiler_generated_main"]==72,census["old_fixup"]==0,
      census["candidate_weight_args"]==72,census["unique_weight_bases"]==72,census["all_weights_canonical"],
      census["expanded_fp16_weight_bytes"]==0,census["qo_fp16_overlays"]==0,census["q8_records"]==72))
    payload={"schema":"tinygrad.nv_compiler_q4k_qo_model_arm.v1","arm":"candidate","status":"PASS" if passed else "FAIL",
      "route":{"default_enabled":False,"candidate_identity":identity},"census":census,**common}
  _write(args.out,payload)
  if payload["status"]!="PASS":raise SystemExit(1)

if __name__=="__main__":main()
