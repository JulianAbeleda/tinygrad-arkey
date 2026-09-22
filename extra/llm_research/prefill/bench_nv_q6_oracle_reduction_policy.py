#!/usr/bin/env python3
"""Locked R31 reduction-policy qualification for the admitted packed one-body Q6 route."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, statistics, time
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record, _run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import ROWS, COLS, SHARED_BYTES
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import (
  ALL_PARTIALS_ASCENDING, LLAMA_STANDALONE_ASCENDING, LLAMA_STANDALONE_NONFINAL_FIRST, ARM_NAMES,
  M, N, K, OWNERS, K256, TILES_M, TILES, TILE_ELEMS, SEGMENT_ORDER_BUILDER_PATCH,
  ast_signature, build_packed_one_body_ast, build_reduction_schedule, cpu_all_partials, cpu_direct_final,
  direct_final_writeback, direct_output_from_partials, emit_ordered_fixup, supports_nonfinal_first)
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
LLAMA_MAIN_US,LLAMA_FIXUP_US=201.216,8.640


def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")
def _stats(xs): return {"samples_us":xs,"min_us":min(xs),"median_us":statistics.median(xs),"max_us":max(xs)}
def _windows(xs): return {"r9":_stats(xs[:9]),"r31":_stats(xs)}
def _paired(lhs,rhs):
  delta=[a-b for a,b in zip(lhs,rhs)]
  med=statistics.median(delta)
  mad=statistics.median([abs(x-med) for x in delta])
  return {**_windows(delta),"paired_mad_us":mad,"candidate_wins":sum(x<0 for x in delta),"pairs":len(delta)}


def _sha256_array(value:np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _compile_ast(ast:UOp,label:str,root:pathlib.Path):
  out=root/label
  out.mkdir(parents=True,exist_ok=True)
  started=time.perf_counter()
  program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  render_ms=(time.perf_counter()-started)*1e3
  match=re.search(r'extern "C" __global__ void (?:__launch_bounds__\(\d+\) )?(\w+)\(',source)
  if match is None: raise RuntimeError(f"rendered symbol missing for {label}")
  source_path=out/f"{label}.cu"
  source_path.write_text(source)
  started=time.perf_counter()
  binary=Device["NV"].compiler.compile(source)
  compile_ms=(time.perf_counter()-started)*1e3
  cubin=out/f"{label}.cubin"
  cubin.write_bytes(binary)
  census_artifact=analyze_cubin(cubin,out/"sass",match.group(1))
  return match.group(1),binary,{"source":str(source_path),"source_bytes":len(source),
    "source_sha256":hashlib.sha256(source.encode()).hexdigest(),"cubin":str(cubin),
    "cubin_sha256":hashlib.sha256(binary).hexdigest(),"render_ms":render_ms,"compile_ms":compile_ms,
    "sass":census_artifact["summary"],"sass_artifacts":{k:census_artifact[k] for k in ("sass_json","disassembly","resources")}}


def _arm_spec(name:str):
  if name==ALL_PARTIALS_ASCENDING:return {"output_policy":"all_partials","segment_order":"ascending","direct_final":False}
  if name==LLAMA_STANDALONE_ASCENDING:return {"output_policy":"llama_standalone","segment_order":"ascending","direct_final":True}
  if name==LLAMA_STANDALONE_NONFINAL_FIRST:return {"output_policy":"llama_standalone","segment_order":"nonfinal_first","direct_final":True}
  raise ValueError(name)


def _ordered_direct_final_writeback(base_ast:UOp,destination:UOp,*,name:str,segment_order:str) -> UOp:
  if segment_order=="ascending":return direct_final_writeback(base_ast,destination,name=name)
  if segment_order!="nonfinal_first":raise ValueError(segment_order)
  # Validate the frozen terminal shape through the shared adapter, then change only its physical-to-logical classifier.
  direct_final_writeback(base_ast,destination,name=name)
  body_end=base_ast.src[0]
  stores,segment=body_end.src
  bid=next(x for x in base_ast.toposort() if x.op is Ops.SPECIAL and x.arg=="gidx0")
  work=TILES*K256
  owner_start=bid*work//OWNERS; owner_stop=(bid+1)*work//OWNERS
  tile0=owner_start//K256; boundary=(tile0+1)*K256; first_stop=owner_stop.minimum(boundary)
  segment_count=1+(owner_stop>boundary).cast(dtypes.int32)
  logical_segment=segment_count-1-segment
  second=logical_segment>0
  tile=second.where(tile0+1,tile0); epoch_start=second.where(owner_start*0,owner_start-tile0*K256)
  segment_depth=second.where(owner_stop-first_stop,first_stop-owner_start)
  is_final=(epoch_start+segment_depth).eq(K256)
  rewritten=[]
  for store in stores.src:
    scratch_index=store.src[0].src[1]; z=scratch_index%TILE_ELEMS
    wr,mc=z//COLS,z%COLS; mt,nt=tile%TILES_M,tile//TILES_M
    destination_index=(mt*COLS+mc)*N+nt*ROWS+wr
    gate=store.src[2]
    rewritten.append(store.replace(src=(store.src[0],store.src[1],gate&is_final.eq(False))))
    rewritten.append(destination[destination_index].store(store.src[1],gate=gate&is_final))
  terminal=body_end.replace(src=(UOp.group(*rewritten),segment))
  return base_ast.replace(src=(terminal,),arg=KernelInfo(name=name,opts_to_apply=()))


def _balanced_order(arms:tuple[str,...], round_index:int) -> tuple[str,...]:
  if len(arms)==2:return arms if round_index%2==0 else tuple(reversed(arms))
  if len(arms)==3:
    a,b,c=arms
    return ((a,b,c),(b,c,a),(c,a,b),(c,b,a),(a,c,b),(b,a,c))[round_index%6]
  return arms


def _resource_signature(artifact):
  sass=artifact["sass"]
  resources=sass.get("resources") or {}
  families=sass.get("families") or {}
  keys=("IMMA","LDSM","LDG","STG","STS","BAR","LDL","STL","FADD","FFMA","ATOM","MEMBAR")
  return {"instruction_total":sass.get("instruction_total"),"registers":resources.get("registers"),
    "stack_bytes":resources.get("stack_bytes"),"shared_static_bytes":resources.get("shared_static_bytes"),
    "local_static_bytes":resources.get("local_static_bytes"),"families":{k:families.get(k,0) for k in keys}}


def _write_blocked(args, arms, reason):
  result={"schema":"tinygrad.nv_q6_oracle_reduction_policy.v1","verdict":"BLOCKED_BEFORE_GPU",
    "requested_arms":list(arms),"blocker":{"reason":reason,"builder_patch":SEGMENT_ORDER_BUILDER_PATCH},
    "gpu_work_started":False,"progress_unsafe_spin_implemented":False}
  args.out.parent.mkdir(parents=True,exist_ok=True)
  args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True))
  return 2


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31)
  ap.add_argument("--warmups",type=int,default=3)
  ap.add_argument("--arms",default=",".join(ARM_NAMES))
  ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--artifacts",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.rounds!=31:raise ValueError("reduction qualification requires exactly R31")
  arms=tuple(x.strip() for x in args.arms.split(",") if x.strip())
  if len(set(arms))!=len(arms) or any(x not in ARM_NAMES for x in arms):raise ValueError(f"invalid arms {arms}")
  if ALL_PARTIALS_ASCENDING not in arms:raise ValueError("all-partials anchor is mandatory")
  if LLAMA_STANDALONE_NONFINAL_FIRST in arms and not supports_nonfinal_first():
    return _write_blocked(args,arms,"nonfinal-first segment order is not expressible by the current admitted builder")

  args.artifacts.mkdir(parents=True,exist_ok=True)
  schedule=build_reduction_schedule()
  slot_map,counts,final_indices=schedule.arrays()
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  counts_t=Tensor(counts,device="NV").contiguous().realize()
  final_indices_t=Tensor(final_indices,device="NV").contiguous().realize()

  model=pathlib.Path(args.model)
  meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K:raise RuntimeError(info)
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K)
  wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  broad=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
        np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(broad).reshape(-1),device="NV").contiguous().realize()

  wide_artifacts=args.artifacts/"wide"
  wide_artifacts.mkdir(parents=True,exist_ok=True)
  direct=_run("wide_direct",M,N,K,halfs,wide_q8,args.rounds,wide_artifacts,(128,128,2,4,256))
  direct_source=(wide_artifacts/"wide_direct.cu").read_text()
  direct_binary=Device["NV"].compiler.compile(direct_source)
  match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',direct_source)
  if match is None:raise RuntimeError("trusted direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),direct_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  states={}
  for arm in arms:
    spec=_arm_spec(arm)
    base_ast=build_packed_one_body_ast(spec["segment_order"])
    main_ast=(_ordered_direct_final_writeback(base_ast,UOp.placeholder((M*N,),dtypes.float32,3),
      name=f"nv_q6_{arm}",segment_order=spec["segment_order"])
      if spec["direct_final"] else base_ast)
    main_name,main_binary,main_artifact=_compile_ast(main_ast,f"main_{arm}",args.artifacts)
    p=lambda n,t,i:UOp.placeholder((n,),t,i)
    fix_ast=emit_ordered_fixup(p(M*N,dtypes.float32,0),p(2*OWNERS*TILE_ELEMS,dtypes.float32,1),
      p(TILES*3,dtypes.int32,2),p(TILES,dtypes.int32,3),
      p(TILES,dtypes.int32,4) if spec["direct_final"] else None,direct_final=spec["direct_final"])
    fix_name,fix_binary,fix_artifact=_compile_ast(fix_ast,f"fixup_{arm}",args.artifacts)
    states[arm]={"spec":spec,"main":NVProgram(Device["NV"],main_name,main_binary,shared_mem=LAUNCH_SHARED_BYTES),
      "fixup":NVProgram(Device["NV"],fix_name,fix_binary),"main_ast":ast_signature(main_ast),
      "compiler":{"main":main_artifact,"fixup":fix_artifact},
      "partials":Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize(),
      "output":Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()}

  def launch_main(state):
    args0=(_buf(state["partials"]),_buf(halfs),_buf(q8))
    if state["spec"]["direct_final"]:args0+=(_buf(state["output"]),)
    return state["main"](*args0,global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6
  def launch_fixup(state):
    args0=(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),_buf(counts_t))
    if state["spec"]["direct_final"]:args0+=(_buf(final_indices_t),)
    return state["fixup"](*args0,global_size=(TILES,1,1),local_size=(256,1,1),wait=True)*1e6

  correctness={}
  raw_by_arm={}
  output_by_arm={}
  pre_output_by_arm={}
  for arm in arms:
    state=states[arm]
    launch_main(state)
    pre_output=state["output"].numpy()
    raw=state["partials"].numpy().reshape(2*OWNERS,ROWS,COLS)
    launch_fixup(state)
    got=state["output"].numpy()
    cpu=cpu_direct_final(raw,pre_output,schedule) if state["spec"]["direct_final"] else cpu_all_partials(raw,schedule)
    ref_diff=np.abs(got-expected)
    close=np.isclose(got,expected,rtol=2e-5,atol=2e-3)
    expected_slots=set(schedule.predecessor_slots if state["spec"]["direct_final"] else schedule.active_slots)
    untouched_slots=set(range(2*OWNERS))-expected_slots
    correctness[arm]={"finite":bool(np.isfinite(got).all()),
      "gpu_cpu_fold_bit_exact":bool(np.array_equal(got.view(np.uint32),cpu.view(np.uint32))),
      "written_slots_finite":bool(np.isfinite(raw[list(expected_slots)]).all()),
      "untouched_slots_nan":bool(np.isnan(raw[list(untouched_slots)]).all()),
      "reference_max_abs":float(ref_diff.max()),"reference_mean_abs":float(ref_diff.mean()),
      "reference_max_rel":float(np.max(ref_diff/np.maximum(np.abs(expected),np.float32(1e-30)))),
      "reference_failing_count":int(np.count_nonzero(~close)),"reference_passed":bool(close.all()),
      "written_partial_sha256":_sha256_array(raw[sorted(expected_slots)]),
      "full_partial_sha256":_sha256_array(raw),"pre_fix_output_sha256":_sha256_array(pre_output),
      "final_output_sha256":_sha256_array(got)}
    raw_by_arm[arm]=raw
    output_by_arm[arm]=got
    pre_output_by_arm[arm]=pre_output

  anchor_raw=raw_by_arm[ALL_PARTIALS_ASCENDING]
  anchor_output=output_by_arm[ALL_PARTIALS_ASCENDING]
  expected_direct=direct_output_from_partials(anchor_raw,schedule)
  diagnostics={}
  pred=list(schedule.predecessor_slots)
  for arm in arms:
    if arm==ALL_PARTIALS_ASCENDING:continue
    diagnostics[arm]={"predecessor_partials_anchor_bit_exact":bool(np.array_equal(
      raw_by_arm[arm][pred].view(np.uint32),anchor_raw[pred].view(np.uint32))),
      "direct_final_anchor_bit_exact":bool(np.array_equal(
        pre_output_by_arm[arm].view(np.uint32),expected_direct.view(np.uint32))),
      "final_output_anchor_bit_exact":bool(np.array_equal(
        output_by_arm[arm].view(np.uint32),anchor_output.view(np.uint32)))}

  for warmup in range(args.warmups):
    for arm in _balanced_order(arms,warmup):launch_main(states[arm]);launch_fixup(states[arm])
  samples={arm:{"main":[],"fixup":[],"reset":[],"total":[]} for arm in arms}
  launch_orders=[]
  for round_index in range(args.rounds):
    order=_balanced_order(arms,round_index)
    launch_orders.append(list(order))
    for arm in order:
      main_us=launch_main(states[arm])
      fixup_us=launch_fixup(states[arm])
      reset_us=0.0
      samples[arm]["main"].append(main_us)
      samples[arm]["fixup"].append(fixup_us)
      samples[arm]["reset"].append(reset_us)
      samples[arm]["total"].append(main_us+fixup_us+reset_us)
  timing={arm:{kind:_windows(values) for kind,values in kinds.items()} for arm,kinds in samples.items()}
  pairs={}
  if LLAMA_STANDALONE_ASCENDING in arms:
    pairs["llama_ascending_minus_all_partials"]=_paired(samples[LLAMA_STANDALONE_ASCENDING]["total"],samples[ALL_PARTIALS_ASCENDING]["total"])
  if LLAMA_STANDALONE_NONFINAL_FIRST in arms and LLAMA_STANDALONE_ASCENDING in arms:
    pairs["nonfinal_first_minus_ascending_standalone"]=_paired(
      samples[LLAMA_STANDALONE_NONFINAL_FIRST]["total"],samples[LLAMA_STANDALONE_ASCENDING]["total"])

  correctness_pass=bool(direct["passed"] and all(all(v for k,v in rec.items() if k in (
    "finite","gpu_cpu_fold_bit_exact","written_slots_finite","untouched_slots_nan","reference_passed")) for rec in correctness.values())
    and all(all(x.values()) for x in diagnostics.values()))
  def arm_correct(arm):
    record=correctness[arm]
    flags=("finite","gpu_cpu_fold_bit_exact","written_slots_finite","untouched_slots_nan","reference_passed")
    return bool(direct["passed"] and all(record[x] for x in flags) and
      (arm==ALL_PARTIALS_ASCENDING or all(diagnostics[arm].values())))
  comparison_verdicts={}
  comparison_arms={"llama_ascending_minus_all_partials":(LLAMA_STANDALONE_ASCENDING,ALL_PARTIALS_ASCENDING),
    "nonfinal_first_minus_ascending_standalone":(LLAMA_STANDALONE_NONFINAL_FIRST,LLAMA_STANDALONE_ASCENDING)}
  for comparison,paired in pairs.items():
    candidate,anchor=comparison_arms[comparison]
    exact=arm_correct(candidate) and arm_correct(anchor)
    median=paired["r31"]["median_us"]; material=max(3.0,3.0*paired["paired_mad_us"])
    if not exact:verdict="FAIL_CORRECTNESS"
    elif median <= -material and paired["candidate_wins"] >= 24:verdict="PROMOTE_CANDIDATE"
    elif abs(median) < material:verdict="RETAIN_CONTROL_NO_MATERIAL_WIN"
    else:verdict="REJECT_CANDIDATE"
    comparison_verdicts[comparison]={"candidate":candidate,"anchor":anchor,"verdict":verdict,
      "correctness_passed":exact,"materiality_threshold_us":material,"required_wins":24}
  result={"schema":"tinygrad.nv_q6_oracle_reduction_policy.v1","shape":{"M":M,"N":N,"K":K},
    "arms":{arm:_arm_spec(arm) for arm in arms},"ownership":schedule.json_record(),
    "reset_contract":{"counter_state_present":False,"required_reset_us":0.0,
      "timed_total_formula":"main_us + fixup_us + required_reset_us"},
    "correctness":correctness,"candidate_anchor_diagnostics":diagnostics,"reference":{"kind":"compiler_wide_direct","result":direct},
    "structure":{arm:states[arm]["main_ast"] for arm in arms},
    "compiler":{arm:states[arm]["compiler"] for arm in arms},
    "resources":{arm:{kind:_resource_signature(artifact) for kind,artifact in states[arm]["compiler"].items()} for arm in arms},
    "timing":timing,"paired":pairs,"comparison_verdicts":comparison_verdicts,"launch_orders":launch_orders,
    "segment_order_confound":{"intentional_arm":LLAMA_STANDALONE_NONFINAL_FIRST,
      "isolated_comparison":"nonfinal-first standalone versus ascending standalone"},
    "baselines":{"llama_main_us":LLAMA_MAIN_US,"llama_fixup_us":LLAMA_FIXUP_US,"llama_total_us":LLAMA_MAIN_US+LLAMA_FIXUP_US},
    "gpu_lock":{"mode":"outer_flock_required","path":"/tmp/nv-q6-oracle-gpu.lock",
      "acquired":os.environ.get("NV_Q6_GPU_LOCK_HELD")=="1"},
    "progress_unsafe_spin_implemented":False,"correctness_passed":correctness_pass,
    "verdict":"QUALIFIED_REDUCTION_SEQUENCE" if correctness_pass else "FAIL_CLOSED_CORRECTNESS"}
  args.out.parent.mkdir(parents=True,exist_ok=True)
  args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True))
  return 0 if correctness_pass else 1


if __name__=="__main__":raise SystemExit(main())
