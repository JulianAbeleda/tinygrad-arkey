#!/usr/bin/env python3
"""Gate 8: locked R31 geometry-only A/B for the ordered Q6 all-partials fixup."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, time
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import UOp
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_reduction_policy import LAUNCH_SHARED_BYTES, _buf, _compile_ast
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record, _run
from extra.llm_research.prefill.nv_q6_oracle_fixup_geometry import (
  BLOCK, GRID, M, N, ROWS, COLS, SYMBOL, TILE_ELEMS, TILES, contract_record, four_slice_scatter_source)
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import (
  K, K256, OWNERS, TILES_M, build_packed_one_body_ast, build_reduction_schedule, cpu_all_partials, emit_ordered_fixup)
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin


GATE = "G8_FOUR_SLICE_SCATTER_GEOMETRY"
RELEASE_ENV = "NV_Q6_GATE8_GPU_RELEASED"
LOCK_ENV = "NV_Q6_GPU_LOCK_HELD"
LOCK_PATH = "/tmp/nv-q6-oracle-gpu.lock"
CONTROL = "one_tile_256"
CANDIDATE = "four_slice_128"


def _stats(values): return {"samples_us":values,"min_us":min(values),"median_us":statistics.median(values),"max_us":max(values)}
def _windows(values): return {"r9":_stats(values[:9]),"r31":_stats(values)}
def _sha(value:np.ndarray) -> str: return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def _paired(candidate,control):
  delta=[a-b for a,b in zip(candidate,control)]; median=statistics.median(delta)
  mad=statistics.median(abs(x-median) for x in delta)
  return {**_windows(delta),"paired_mad_us":mad,"candidate_wins":sum(x<0 for x in delta),"pairs":len(delta)}


def _blocked(args,reason:str) -> int:
  result={"schema":"tinygrad.nv_q6_oracle_fixup_geometry.v1","gate":GATE,"verdict":"BLOCKED_BEFORE_GPU",
    "reason":reason,"release_env":RELEASE_ENV,"lock_env":LOCK_ENV,"lock_path":LOCK_PATH,"gpu_work_started":False}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True));return 2


def _compile_cuda(source:str,root:pathlib.Path):
  out=root/CANDIDATE;out.mkdir(parents=True,exist_ok=True)
  source_path=out/f"{CANDIDATE}.cu";source_path.write_text(source)
  started=time.perf_counter();binary=Device["NV"].compiler.compile(source);compile_ms=(time.perf_counter()-started)*1e3
  cubin_path=out/f"{CANDIDATE}.cubin";cubin_path.write_bytes(binary)
  census=analyze_cubin(cubin_path,out/"sass",SYMBOL)
  return binary,{"source":str(source_path),"source_sha256":hashlib.sha256(source.encode()).hexdigest(),
    "cubin":str(cubin_path),"cubin_sha256":hashlib.sha256(binary).hexdigest(),"compile_ms":compile_ms,
    "sass":census["summary"],"sass_artifacts":{k:census[k] for k in ("sass_json","disassembly","resources")}}


def _resource_record(artifact):
  sass=artifact["sass"];resources=sass.get("resources") or {};families=sass.get("families") or {}
  keys=("LDG","STG","FADD","IADD","IMAD","ISETP","BRA","BAR","LDL","STL","ATOM","MEMBAR")
  return {"instruction_total":sass.get("instruction_total"),"registers":resources.get("registers"),
    "stack_bytes":resources.get("stack_bytes"),"shared_static_bytes":resources.get("shared_static_bytes"),
    "local_static_bytes":resources.get("local_static_bytes"),"families":{key:families.get(key,0) for key in keys}}


def _resource_gate(record):
  fam=record["families"]
  checks={"registers_at_most_84":record["registers"] is not None and record["registers"]<=84,
    "stack_zero":record["stack_bytes"]==0,"ldl_zero":fam["LDL"]==0,"stl_zero":fam["STL"]==0,
    "atom_zero":fam["ATOM"]==0,"membar_zero":fam["MEMBAR"]==0,"bar_zero":fam["BAR"]==0}
  return checks,all(checks.values())


def main() -> int:
  parser=argparse.ArgumentParser()
  parser.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  parser.add_argument("--rounds",type=int,default=31)
  parser.add_argument("--warmups",type=int,default=3)
  parser.add_argument("--out",type=pathlib.Path,required=True)
  parser.add_argument("--artifacts",type=pathlib.Path,required=True)
  args=parser.parse_args()
  if args.rounds!=31:raise ValueError("Gate 8 requires exactly R31")
  if os.environ.get(RELEASE_ENV)!="1":return _blocked(args,f"explicit GPU release missing: set {RELEASE_ENV}=1 only after Gate 7")
  if os.environ.get(LOCK_ENV)!="1":return _blocked(args,f"outer lock marker missing: run under {LOCK_PATH}")

  args.artifacts.mkdir(parents=True,exist_ok=True)
  source=four_slice_scatter_source()
  candidate_binary,candidate_artifact=_compile_cuda(source,args.artifacts)
  candidate_resources=_resource_record(candidate_artifact)
  resource_checks,resource_passed=_resource_gate(candidate_resources)
  if not resource_passed:
    result={"schema":"tinygrad.nv_q6_oracle_fixup_geometry.v1","gate":GATE,"verdict":"FAIL_CLOSED_RESOURCES",
      "gpu_work_started":False,"contract":contract_record(),"compiler":{"candidate":candidate_artifact},
      "resources":{"candidate":candidate_resources},"resource_checks":resource_checks}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True));return 1

  schedule=build_reduction_schedule();slot_map,counts,_=schedule.arrays()
  descriptor_bytes=np.concatenate((slot_map.reshape(-1),counts)).astype(np.int32,copy=False)
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  counts_t=Tensor(counts,device="NV").contiguous().realize()

  model=pathlib.Path(args.model);meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K:raise RuntimeError(info)
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K);wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  broad=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
        np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(broad).reshape(-1),device="NV").contiguous().realize()

  wide_artifacts=args.artifacts/"wide";wide_artifacts.mkdir(parents=True,exist_ok=True)
  direct=_run("wide_direct",M,N,K,halfs,wide_q8,args.rounds,wide_artifacts,(128,128,2,4,256))
  direct_source=(wide_artifacts/"wide_direct.cu").read_text();direct_binary=Device["NV"].compiler.compile(direct_source)
  import re
  match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',direct_source)
  if match is None:raise RuntimeError("trusted direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),direct_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  main_ast=build_packed_one_body_ast("ascending")
  main_name,main_binary,main_artifact=_compile_ast(main_ast,"main_gate8_anchor",args.artifacts)
  p=lambda length,dtype,slot:UOp.placeholder((length,),dtype,slot)
  control_ast=emit_ordered_fixup(p(M*N,dtypes.float32,0),p(2*OWNERS*TILE_ELEMS,dtypes.float32,1),
    p(TILES*3,dtypes.int32,2),p(TILES,dtypes.int32,3))
  control_name,control_binary,control_artifact=_compile_ast(control_ast,"fixup_gate8_control",args.artifacts)
  main_program=NVProgram(Device["NV"],main_name,main_binary,shared_mem=LAUNCH_SHARED_BYTES)
  control_program=NVProgram(Device["NV"],control_name,control_binary)
  candidate_program=NVProgram(Device["NV"],SYMBOL,candidate_binary)
  partials=Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize()
  control_output=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  candidate_output=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()

  def launch_main():
    return main_program(_buf(partials),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6
  def launch_control():
    return control_program(_buf(control_output),_buf(partials),_buf(slot_map_t),_buf(counts_t),
      global_size=(TILES,1,1),local_size=(256,1,1),wait=True)*1e6
  def launch_candidate():
    return candidate_program(_buf(candidate_output),_buf(partials),_buf(slot_map_t),_buf(counts_t),
      global_size=GRID,local_size=BLOCK,wait=True)*1e6

  launch_main();raw=partials.numpy().reshape(2*OWNERS,ROWS,COLS);partials_before=_sha(raw)
  launch_control();control=control_output.numpy();launch_candidate();candidate_first=candidate_output.numpy()
  launch_candidate();candidate_second=candidate_output.numpy();partials_after=_sha(partials.numpy())
  cpu=cpu_all_partials(raw,schedule);ref_diff=np.abs(candidate_second-expected);close=np.isclose(candidate_second,expected,rtol=2e-5,atol=2e-3)
  correctness={"candidate_control_bit_exact":bool(np.array_equal(candidate_second.view(np.uint32),control.view(np.uint32))),
    "candidate_repeat_bit_exact":bool(np.array_equal(candidate_first.view(np.uint32),candidate_second.view(np.uint32))),
    "control_cpu_fold_bit_exact":bool(np.array_equal(control.view(np.uint32),cpu.view(np.uint32))),
    "partials_unchanged":partials_before==partials_after,"finite":bool(np.isfinite(candidate_second).all()),
    "trusted_max_abs":float(ref_diff.max()),"trusted_mean_abs":float(ref_diff.mean()),
    "trusted_failing_count":int(np.count_nonzero(~close)),"trusted_passed":bool(close.all()),
    "control_sha256":_sha(control),"candidate_sha256":_sha(candidate_second),"partials_sha256":partials_before}
  correctness_passed=all(correctness[key] for key in ("candidate_control_bit_exact","candidate_repeat_bit_exact",
    "control_cpu_fold_bit_exact","partials_unchanged","finite","trusted_passed")) and bool(direct["passed"])

  for warmup in range(args.warmups):
    launch_main()
    if warmup%2==0:launch_control();launch_candidate()
    else:launch_candidate();launch_control()
  samples={CONTROL:{"fixup":[],"total":[]},CANDIDATE:{"fixup":[],"total":[]},"main":[]}
  orders=[]
  for round_index in range(args.rounds):
    main_us=launch_main();samples["main"].append(main_us)
    order=(CONTROL,CANDIDATE) if round_index%2==0 else (CANDIDATE,CONTROL);orders.append(list(order));round_fixup={}
    for arm in order:round_fixup[arm]=launch_control() if arm==CONTROL else launch_candidate()
    for arm in (CONTROL,CANDIDATE):
      samples[arm]["fixup"].append(round_fixup[arm]);samples[arm]["total"].append(main_us+round_fixup[arm])
  timing={"main":_windows(samples["main"])}
  for arm in (CONTROL,CANDIDATE):timing[arm]={key:_windows(value) for key,value in samples[arm].items()}
  paired=_paired(samples[CANDIDATE]["total"],samples[CONTROL]["total"])
  r9_pass=paired["r9"]["median_us"]<=-3.0 and sum(x<0 for x in paired["r9"]["samples_us"])>=7
  material=max(3.0,3.0*paired["paired_mad_us"])
  r31_pass=paired["r31"]["median_us"]<=-material and paired["candidate_wins"]>=24
  gates={"correctness":correctness_passed,"resources":resource_passed,"r9":r9_pass,"r31":r31_pass,
    "required_r9_improvement_us":3.0,"required_r9_wins":7,"r31_materiality_us":material,"required_r31_wins":24}
  passed=all(gates[key] for key in ("correctness","resources","r9","r31"))
  result={"schema":"tinygrad.nv_q6_oracle_fixup_geometry.v1","gate":GATE,
    "verdict":"PROMOTE_FOUR_SLICE_SCATTER_GEOMETRY" if passed else "RETAIN_ONE_TILE_256",
    "shape":{"M":M,"N":N,"K":K},"contract":contract_record(),"ownership":schedule.json_record(),
    "descriptor_sha256":hashlib.sha256(descriptor_bytes.tobytes()).hexdigest(),"correctness":correctness,
    "compiler":{"main":main_artifact,"control":control_artifact,"candidate":candidate_artifact},
    "resources":{"control":_resource_record(control_artifact),"candidate":candidate_resources},
    "resource_checks":resource_checks,"timing":timing,"paired_candidate_minus_control":paired,"launch_orders":orders,
    "reset_contract":{"counter_state_present":False,"required_reset_us":0.0,"timed_total_formula":"shared main_us + fixup_us"},
    "progress_contract":{"unique_writer":True,"atomics":False,"membar":False,"spin":False,"inter_cta_dependency":False},
    "gates":gates,"gpu_lock":{"path":LOCK_PATH,"acquired":True},"gpu_release":{"env":RELEASE_ENV,"released":True}}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True));return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
