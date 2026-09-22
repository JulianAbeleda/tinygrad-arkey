#!/usr/bin/env python3
"""R31 qualification for a genuine 170-CTA packed Q6 Stream-K body."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, statistics, time
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_full_streamk import _combine_sources,_fixup_source
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record,_run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import ROWS,COLS,SHARED_BYTES,q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

M,N,K,OWNERS,K256=512,4096,12288,170,48
TILES_M,TILES_N,TILES=4,32,128
TILE_ELEMS=ROWS*COLS
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
ARMS=("duplicated_body_packed_anchor","one_physical_body_packed")

def _buf(t): return t.uop.buffer.get_buf("NV")
def _stats(x): return {"samples_us":x,"min_us":min(x),"median_us":statistics.median(x),"max_us":max(x)}
def _windows(x): return {"r9":_stats(x[:9]),"r31":_stats(x)}
def _paired(lhs,rhs):
  delta=[a-b for a,b in zip(lhs,rhs)]
  return {**_windows(delta),"candidate_wins":sum(x<0 for x in delta),"pairs":len(delta)}

def _schedule():
  slots=[[] for _ in range(TILES)];records=[];coverage=np.zeros(TILES*K256,np.int16);owner_counts=[]
  for owner in range(OWNERS):
    start=owner*TILES*K256//OWNERS;stop=(owner+1)*TILES*K256//OWNERS;tile0=start//K256;boundary=(tile0+1)*K256
    first_stop=min(stop,boundary);segments=[(0,tile0,start-tile0*K256,first_stop-tile0*K256)]
    if stop>boundary: segments.append((1,tile0+1,0,stop-boundary))
    owner_counts.append(len(segments))
    for plane,tile,k_begin,k_end in segments:
      slot=plane*OWNERS+owner;slots[tile].append(slot);coverage[tile*K256+k_begin:tile*K256+k_end]+=1
      records.append({"owner":owner,"plane":plane,"slot":slot,"tile":tile,"k_begin":k_begin,"k_end":k_end})
  for tile_slots in slots: tile_slots.sort()
  invariants={"all_work_once":bool(np.all(coverage==1)),"owner_segment_count_one_or_two":bool(set(owner_counts)<=set((1,2))),
    "positive_segments":all(x["k_end"]>x["k_begin"] for x in records),
    "plane_major_slots":all(x==sorted(x) for x in slots),"record_count":len(records),"covered_work_units":int(coverage.sum())}
  return slots,records,invariants

def _ast(segment:int|None):
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  return q6_oracle_broad_cta_kernel(ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),prefetch_second_panel=True,factor_dA=False,oracle_publisher=True,
    weight_scale_contract="trusted_fp16_packed",streamk_owners=OWNERS,streamk_segment=segment or 0,
    streamk_segments_in_cta=segment is None)

def _compile_ast(ast,label:str,root:pathlib.Path):
  out=root/label;out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
  program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE);render_ms=(time.perf_counter()-started)*1e3
  match=re.search(r'__launch_bounds__\(256\) (\w+)\(',source)
  if match is None: raise RuntimeError("rendered symbol missing")
  source_path=out/f"{label}.cu";source_path.write_text(source);started=time.perf_counter();binary=Device["NV"].compiler.compile(source)
  compile_ms=(time.perf_counter()-started)*1e3;cubin=out/f"{label}.cubin";cubin.write_bytes(binary)
  census_artifact=analyze_cubin(cubin,out/"sass",match.group(1));census=census_artifact["summary"]
  artifact={"source":str(source_path),"source_bytes":len(source),"source_sha256":hashlib.sha256(source.encode()).hexdigest(),
    "cubin":str(cubin),"cubin_sha256":hashlib.sha256(binary).hexdigest(),"render_ms":render_ms,"compile_ms":compile_ms,
    "sass":census,"sass_artifacts":{k:census_artifact[k] for k in ("sass_json","disassembly","resources")}}
  return match.group(1),binary,artifact,source

def _ast_proof(ast):
  ranges={x.arg[0]:x for x in ast.toposort() if x.op is Ops.RANGE and x.arg[0] in (1498,1499)}
  segment,epoch=ranges.get(1498),ranges.get(1499);barriers=[x for x in ast.toposort() if x.op is Ops.BARRIER]
  ends=[x for x in ast.toposort() if x.op is Ops.END]
  reset_or_partial=[] if segment is None or epoch is None else [x for x in ast.toposort() if x.op is Ops.STORE and segment in x.ranges and epoch not in x.ranges]
  return {"one_segment_range":segment is not None,"one_epoch_range":epoch is not None,
    "epoch_extent_depends_on_segment":bool(segment is not None and epoch is not None and segment in epoch.src[0].ranges),
    "segment_only_stores_present":bool(reset_or_partial),"barrier_nodes":len(barriers),
    "separate_segment_epoch_ends":bool(segment is not None and epoch is not None and
      any(epoch in x.ended_ranges and segment not in x.ended_ranges for x in ends) and
      any(segment in x.ended_ranges and epoch not in x.ended_ranges for x in ends))}

def _resource_signature(artifact):
  summary=artifact["sass"];resources=summary.get("resources") or {};families=summary.get("families",{})
  return {"registers":resources.get("registers"),"stack_bytes":resources.get("stack_bytes"),
    "local_static_bytes":resources.get("local_static_bytes"),"shared_static_bytes":resources.get("shared_static_bytes"),
    "LDL":families.get("LDL",0),"STL":families.get("STL",0),"IMMA":families.get("IMMA",0),"BAR":families.get("BAR",0)}

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31);ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--artifacts",type=pathlib.Path,required=True);a=ap.parse_args()
  if a.rounds != 31: raise ValueError("one-body qualification requires exactly R31")
  a.artifacts.mkdir(parents=True,exist_ok=True);model=pathlib.Path(a.model);meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ != GGML_Q6_K: raise RuntimeError(info)
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K);wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  broad=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
        np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(broad).reshape(-1),device="NV").contiguous().realize()

  wide_artifacts=a.artifacts/"wide";wide_artifacts.mkdir(parents=True,exist_ok=True)
  direct=_run("wide_direct",M,N,K,halfs,wide_q8,a.rounds,wide_artifacts,(128,128,2,4,256))
  direct_source=(a.artifacts/"wide"/"wide_direct.cu").read_text();direct_binary=Device["NV"].compiler.compile(direct_source)
  match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',direct_source)
  if match is None: raise RuntimeError("trusted direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),direct_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  segment_states=[]
  for segment in (0,1):
    name,binary,artifact,source=_compile_ast(_ast(segment),f"anchor_segment_{segment}",a.artifacts)
    segment_states.append((name,binary,artifact,source))
  anchor_name,anchor_binary,anchor_artifact=_combine_sources(segment_states[0][3],segment_states[1][3],a.artifacts,"packed_anchor")
  candidate_ast=_ast(None);candidate_name,candidate_binary,candidate_artifact,candidate_source=_compile_ast(
    candidate_ast,"one_physical_body_packed",a.artifacts)
  states={"duplicated_body_packed_anchor":{"program":NVProgram(Device["NV"],anchor_name,anchor_binary,shared_mem=LAUNCH_SHARED_BYTES),
      "compiler":{"full":anchor_artifact,"segments":[x[2] for x in segment_states]}},
    "one_physical_body_packed":{"program":NVProgram(Device["NV"],candidate_name,candidate_binary,shared_mem=LAUNCH_SHARED_BYTES),
      "compiler":{"full":candidate_artifact}}}

  slots,ownership,ownership_invariants=_schedule();max_segments=max(map(len,slots));slot_map=np.full((TILES,max_segments),-1,np.int32)
  for tile,tile_slots in enumerate(slots):slot_map[tile,:len(tile_slots)]=tile_slots
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  fix_source=_fixup_source(max_segments);fix_path=a.artifacts/"plane_major_fixup.cu";fix_path.write_text(fix_source)
  fix_binary=Device["NV"].compiler.compile(fix_source);fix_cubin=a.artifacts/"plane_major_fixup.cubin";fix_cubin.write_bytes(fix_binary)
  fix_census=analyze_cubin(fix_cubin,a.artifacts/"sass_fixup","q6_oracle_fixup")
  fix_artifact={"source":str(fix_path),"cubin":str(fix_cubin),"cubin_sha256":hashlib.sha256(fix_binary).hexdigest(),
    "sass":fix_census["summary"],"sass_artifacts":{k:fix_census[k] for k in ("sass_json","disassembly","resources")}}
  fix=NVProgram(Device["NV"],"q6_oracle_fixup",fix_binary)
  def cpu_fixup(raw):
    raw=raw.reshape(2*OWNERS,ROWS,COLS);out=np.empty((M,N),np.float32)
    for tile,tile_slots in enumerate(slots):
      reduced=raw[tile_slots[0]].copy()
      for slot in tile_slots[1:]:reduced+=raw[slot]
      mt,nt=tile%TILES_M,tile//TILES_M;out[mt*COLS:(mt+1)*COLS,nt*ROWS:(nt+1)*ROWS]=reduced.T
    return out

  correctness={};outputs={};partials={}
  for arm,state in states.items():
    state["partials"]=Tensor.full((2*OWNERS*TILE_ELEMS),float("nan"),device="NV").contiguous().realize()
    state["output"]=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
    state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
    fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)
    got=state["output"].numpy();raw=state["partials"].numpy();cpu=cpu_fixup(raw);ref_diff=np.abs(got-expected)
    close=np.isclose(got,expected,rtol=2e-5,atol=2e-3);outputs[arm]=got;partials[arm]=raw
    correctness[arm]={"finite":bool(np.isfinite(got).all()),"gpu_fixup_cpu_bit_exact":bool(np.array_equal(got.view(np.uint32),cpu.view(np.uint32))),
      "reference_max_abs":float(ref_diff.max()),"reference_mean_abs":float(ref_diff.mean()),
      "reference_failing_count":int(np.count_nonzero(~close)),"reference_passed":bool(close.all())}
  partial_exact=bool(np.array_equal(partials[ARMS[0]].view(np.uint32),partials[ARMS[1]].view(np.uint32)))
  final_exact=bool(np.array_equal(outputs[ARMS[0]].view(np.uint32),outputs[ARMS[1]].view(np.uint32)))

  ast_proof=_ast_proof(candidate_ast);anchor_source=pathlib.Path(anchor_artifact["source"]).read_text()
  resources={arm:_resource_signature(states[arm]["compiler"]["full"]) for arm in ARMS}
  anchor_res,candidate_res=resources[ARMS[0]],resources[ARMS[1]]
  structural={**ast_proof,"candidate_one_global_kernel":len(re.findall(r'extern "C" __global__',candidate_source))==1,
    "candidate_has_no_spliced_segment_helpers":"q6_segment_" not in candidate_source,
    "anchor_has_two_segment_helpers":all(x in anchor_source for x in ("q6_segment_0","q6_segment_1")),
    "candidate_five_sass_barriers":candidate_res["BAR"]==5,
    "anchor_has_duplicated_imma_body":candidate_res["IMMA"]>0 and anchor_res["IMMA"]==2*candidate_res["IMMA"]}
  structural_pass=bool(all(structural.values()))
  compare_keys=("stack_bytes","local_static_bytes","LDL","STL")
  resource_comparable=all(anchor_res[k] is not None and candidate_res[k] is not None for k in compare_keys)
  resource_pass=bool(resource_comparable and all(candidate_res[k]<=anchor_res[k] for k in compare_keys))

  samples={arm:{"main":[],"fixup":[],"pair":[]} for arm in ARMS}
  for round_idx in range(a.rounds):
    order=ARMS if round_idx%2==0 else tuple(reversed(ARMS))
    for arm in order:
      state=states[arm];main_us=state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),
        local_size=(256,1,1),wait=True,timeout=120000)*1e6
      fix_us=fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)*1e6
      samples[arm]["main"].append(main_us);samples[arm]["fixup"].append(fix_us);samples[arm]["pair"].append(main_us+fix_us)
  paired={kind:_paired(samples[ARMS[1]][kind],samples[ARMS[0]][kind]) for kind in ("main","fixup","pair")}
  timing={arm:{kind:_windows(values) for kind,values in samples[arm].items()} for arm in ARMS}
  performance_pass=bool(timing[ARMS[0]]["pair"]["r31"]["median_us"]-timing[ARMS[1]]["pair"]["r31"]["median_us"]>=3.0 and
    paired["pair"]["candidate_wins"]>=24)
  correctness_pass=bool(direct["passed"] and all(x["finite"] and x["gpu_fixup_cpu_bit_exact"] and x["reference_passed"] for x in correctness.values())
    and partial_exact and final_exact and all(ownership_invariants.values()))
  promotion=bool(correctness_pass and structural_pass and resource_pass and performance_pass)
  verdict=("PROMOTE_ONE_PHYSICAL_BODY_PACKED" if promotion else ("CORRECT_NOT_PROMOTED" if correctness_pass else "FAIL_CLOSED_CORRECTNESS"))
  result={"schema":"tinygrad.nv_q6_oracle_streamk_single_body_packed.v1","shape":{"M":M,"N":N,"K":K},
    "launch":{"grid":[OWNERS,1,1],"block":[256,1,1],"shared_bytes":LAUNCH_SHARED_BYTES},
    "ownership":{"records":ownership,"max_segments_per_tile":max_segments,"invariants":ownership_invariants,
      "plane_major_slot_map":slot_map.tolist()},"reference":{"kind":"compiler_wide_direct","result":direct},
    "correctness":correctness,"candidate_vs_anchor":{"partials_bit_exact":partial_exact,"final_bit_exact":final_exact},
    "structure":structural,"resources":resources,"resource_no_regression":resource_pass,"timing":timing,"paired":paired,
    "performance_gate":{"minimum_pair_median_improvement_us":3.0,"minimum_pair_wins":24,"passed":performance_pass},
    "compiler":{"arms":{arm:states[arm]["compiler"] for arm in ARMS},"fixup":fix_artifact},
    "gpu_lock":{"mode":"outer_flock_required","path":"/tmp/nv-q6-oracle-gpu.lock"},
    "promotion_eligible":promotion,"verdict":verdict,"passed":promotion}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,sort_keys=True))
  return 0 if promotion else 1
if __name__=="__main__":raise SystemExit(main())
