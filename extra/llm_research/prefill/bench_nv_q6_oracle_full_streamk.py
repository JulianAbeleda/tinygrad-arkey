#!/usr/bin/env python3
"""Full M512/N4096/K12288 qualifier for the persistent broad Q6 oracle."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, statistics, time
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record, _run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import COLS,ROWS,SHARED_BYTES,q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

M,N,K,OWNERS,K256=512,4096,12288,170,48
TILES_M,TILES_N,TILES=4,32,128
TILE_ELEMS=ROWS*COLS
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
LLAMA_MAIN_US,LLAMA_FIXUP_US=201.216,8.640


def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")
def _stats(x:list[float]): return {"samples_us":x,"min_us":min(x),"median_us":statistics.median(x),"max_us":max(x)}


def _render(segment:int, artifacts:pathlib.Path, factor_dA:bool=True):
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_oracle_broad_cta_kernel(ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),prefetch_second_panel=True,factor_dA=factor_dA,
    oracle_publisher=True,depth=37,streamk_owners=OWNERS,streamk_segment=segment)
  started=time.perf_counter(); program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE); render_ms=(time.perf_counter()-started)*1e3
  da_suffix="_factor_da" if factor_dA else ""
  name=f"nv_q6_oracle_broad_cta_prefetch{da_suffix}_oracle_publisher_streamk_s{segment}"
  path=artifacts/f"{name}.cu"; path.write_text(source)
  started=time.perf_counter(); binary=Device["NV"].compiler.compile(source); compile_ms=(time.perf_counter()-started)*1e3
  cubin=artifacts/f"{name}.cubin"; cubin.write_bytes(binary)
  census_artifact=analyze_cubin(cubin,artifacts/f"sass_{'factored_da' if factor_dA else 'direct_da'}_s{segment}",name)
  census=census_artifact["summary"]
  return name,binary,{"render_ms":render_ms,"compile_ms":compile_ms,"source":str(path),"source_bytes":len(source),
    "cubin":str(cubin),"cubin_sha256":hashlib.sha256(binary).hexdigest(),"sass":census,
    "sass_artifacts":{k:census_artifact[k] for k in ("sass_json","disassembly","resources")}}


def _combine_sources(source0:str,source1:str,artifacts:pathlib.Path, arm:str|None=None):
  pattern=r'extern "C" __global__ void __launch_bounds__\(256\) \w+\((.*?)\) \{\n  int gidx0 = blockIdx.x; /\* 170 \*/\n'
  parts=[]; preamble=None; params=None
  for index,source in enumerate((source0,source1)):
    match=re.search(pattern,source,re.DOTALL)
    if match is None: raise RuntimeError(f"generated segment {index} signature mismatch")
    if preamble is None: preamble=source[:match.start()]; params=match.group(1)
    elif params!=match.group(1): raise RuntimeError("generated segment ABIs differ")
    parts.append(f"__device__ __forceinline__ void q6_segment_{index}({match.group(1)}, int gidx0) {{\n"+source[match.end():])
  assert preamble is not None and params is not None
  name=f"nv_q6_oracle_broad_streamk_170{f'_{arm}' if arm is not None else ''}"
  wrapper=(preamble+"\n".join(parts)+f'''\nextern "C" __global__ void __launch_bounds__(256) {name}({params}) {{
    int owner=blockIdx.x;
    q6_segment_0(data0_5570560,data1_20643840,data2_1769472,owner);
    __syncthreads();
    q6_segment_1(data0_5570560,data1_20643840,data2_1769472,owner);
  }}\n''')
  path=artifacts/f"{name}.cu";path.write_text(wrapper);started=time.perf_counter();binary=Device["NV"].compiler.compile(wrapper)
  compile_ms=(time.perf_counter()-started)*1e3;cubin=artifacts/f"{name}.cubin";cubin.write_bytes(binary)
  census_artifact=analyze_cubin(cubin,artifacts/(f"sass_full_{arm}" if arm is not None else "sass_full"),name)
  census=census_artifact["summary"]
  return name,binary,{"compile_ms":compile_ms,"source":str(path),"source_bytes":len(wrapper),"cubin":str(cubin),
    "cubin_sha256":hashlib.sha256(binary).hexdigest(),"sass":census,
    "sass_artifacts":{k:census_artifact[k] for k in ("sass_json","disassembly","resources")}}


def _ownership():
  slots=[[] for _ in range(TILES)]; records=[]; work=TILES*K256
  for owner in range(OWNERS):
    start=owner*work//OWNERS; stop=(owner+1)*work//OWNERS; tile0=start//K256; boundary=(tile0+1)*K256
    first_stop=min(stop,boundary); slots[tile0].append(owner)
    records.append({"owner":owner,"slot":owner,"tile":tile0,"k_begin":start-tile0*K256,"k_end":first_stop-tile0*K256})
    if stop>boundary:
      slots[tile0+1].append(OWNERS+owner)
      records.append({"owner":owner,"slot":OWNERS+owner,"tile":tile0+1,"k_begin":0,"k_end":stop-boundary})
  return slots,records


def _fixup_source(max_segments:int) -> str:
  return f'''extern "C" __global__ void q6_oracle_fixup(float *out,const float *partials,const int *map) {{
    int tile=blockIdx.x,mt=tile%{TILES_M},nt=tile/{TILES_M};
    for(int z=threadIdx.x;z<{TILE_ELEMS};z+=256) {{ int wr=z/{COLS},mc=z%{COLS}; float v=0.0f;
      #pragma unroll
      for(int j=0;j<{max_segments};j++) {{ int slot=map[tile*{max_segments}+j]; if(slot>=0)v+=partials[slot*{TILE_ELEMS}+z]; }}
      out[(mt*{COLS}+mc)*{N}+nt*{ROWS}+wr]=v;
    }}
  }}'''


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31); ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--artifacts",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.rounds<31: raise ValueError("full qualification requires R31")
  args.artifacts.mkdir(parents=True,exist_ok=True)
  model=pathlib.Path(args.model); meta=read_metadata(model); info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K: raise RuntimeError(f"illegal fixture {info}")
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K); wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  records=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      records.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
                                  np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(records,axis=0).reshape(-1),device="NV").contiguous().realize()

  direct=_run("wide_direct",M,N,K,halfs,wide_q8,args.rounds,args.artifacts,(128,128,2,4,256))
  direct_source=(args.artifacts/"wide_direct.cu").read_text(); compiler=Device["NV"].compiler
  direct_binary=compiler.compile(direct_source); match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',direct_source)
  if match is None: raise RuntimeError("direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),direct_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  arm_states={}
  for arm,factor_dA in (("direct_da",False),("factored_da",True)):
    _,_,c0=_render(0,args.artifacts,factor_dA); _,_,c1=_render(1,args.artifacts,factor_dA)
    full_name,full_binary,full_compiler=_combine_sources(pathlib.Path(c0["source"]).read_text(),
      pathlib.Path(c1["source"]).read_text(),args.artifacts,arm)
    arm_states[arm]={"factor_dA":factor_dA,
      "program":NVProgram(Device["NV"],full_name,full_binary,shared_mem=LAUNCH_SHARED_BYTES),
      "partials":Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize(),
      "compiler":{"full":full_compiler,"segment0_diagnostic":c0,"segment1_diagnostic":c1}}

  slots,ownership=_ownership(); max_segments=max(map(len,slots)); slot_map=np.full((TILES,max_segments),-1,np.int32)
  for tile,tile_slots in enumerate(slots): slot_map[tile,:len(tile_slots)]=tile_slots
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  fix_source=_fixup_source(max_segments); (args.artifacts/"fixup.cu").write_text(fix_source)
  fix_binary=compiler.compile(fix_source); fix_cubin=args.artifacts/"fixup.cubin"; fix_cubin.write_bytes(fix_binary)
  fix_census_artifact=analyze_cubin(fix_cubin,args.artifacts/"sass_fixup","q6_oracle_fixup")
  fix_census=fix_census_artifact["summary"]
  fix_compiler={"source":str(args.artifacts/"fixup.cu"),"source_bytes":len(fix_source),"cubin":str(fix_cubin),
    "cubin_sha256":hashlib.sha256(fix_binary).hexdigest(),"sass":fix_census,
    "sass_artifacts":{k:fix_census_artifact[k] for k in ("sass_json","disassembly","resources")}}
  fix=NVProgram(Device["NV"],"q6_oracle_fixup",fix_binary)
  def cpu_fixup(raw):
    raw=raw.reshape(2*OWNERS,ROWS,COLS); cpu=np.empty((M,N),np.float32)
    for tile,tile_slots in enumerate(slots):
      mt,nt=tile%TILES_M,tile//TILES_M; reduced=raw[tile_slots[0]].copy()
      for slot in tile_slots[1:]: reduced+=raw[slot]
      cpu[mt*COLS:(mt+1)*COLS,nt*ROWS:(nt+1)*ROWS]=reduced.T
    return cpu

  arm_correctness={}; arm_outputs={}
  for arm,state in arm_states.items():
    state["output"]=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
    state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),
      wait=True,timeout=120000)
    fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)
    got=state["output"].numpy(); cpu=cpu_fixup(state["partials"].numpy()); fix_diff=np.abs(got-cpu)
    ref_diff=np.abs(got-expected); ref_ok=np.isclose(got,expected,rtol=2e-5,atol=2e-3)
    arm_outputs[arm]=got
    arm_correctness[arm]={"finite":bool(np.isfinite(got).all()),"gpu_fixup_cpu_exact":bool(np.array_equal(got,cpu)),
      "gpu_fixup_cpu_max_abs":float(fix_diff.max()),"trusted_reference":"compiler_wide_direct",
      "reference_max_abs":float(ref_diff.max()),"reference_mean_abs":float(ref_diff.mean()),
      "reference_failing_count_rtol2e5_atol2e3":int(np.count_nonzero(~ref_ok)),
      "reference_allclose_rtol2e5_atol2e3":bool(ref_ok.all())}

  samples={arm:{"main":[],"fixup":[],"pair":[]} for arm in arm_states}
  for round_idx in range(args.rounds):
    order=("direct_da","factored_da") if round_idx%2==0 else ("factored_da","direct_da")
    for arm in order:
      state=arm_states[arm]
      main_us=state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),
        local_size=(256,1,1),wait=True,timeout=120000)*1e6
      fix_us=fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),global_size=(TILES,1,1),
        local_size=(256,1,1),wait=True)*1e6
      samples[arm]["main"].append(main_us); samples[arm]["fixup"].append(fix_us)
      samples[arm]["pair"].append(main_us+fix_us)
  def windows(xs): return {"r9":_stats(xs[:9]),"r31":_stats(xs)}
  direct_samples,factored_samples=samples["direct_da"],samples["factored_da"]
  paired_main=[d-f for d,f in zip(direct_samples["main"],factored_samples["main"])]
  paired_pair=[d-f for d,f in zip(direct_samples["pair"],factored_samples["pair"])]
  main,fix_samples,pair=factored_samples["main"],factored_samples["fixup"],factored_samples["pair"]
  main_med=statistics.median(main); fix_med=statistics.median(fix_samples)
  arm_diff=np.abs(arm_outputs["direct_da"]-arm_outputs["factored_da"])
  arm_results={arm:{"factor_dA":state["factor_dA"],"correctness":arm_correctness[arm],
    "timing":{kind:windows(values) for kind,values in samples[arm].items()},"compiler":state["compiler"]}
    for arm,state in arm_states.items()}
  legacy_correctness=arm_correctness["factored_da"]
  result={"schema":"tinygrad.nv_q6_oracle_full_streamk.factor_da_gate0.v1",
    "legacy_schema":"tinygrad.nv_q6_oracle_full_streamk.v1","shape":{"M":M,"N":N,"K":K},"owners":OWNERS,
    "ownership":{"work_units":TILES*K256,"owner_work_lengths":sorted({((owner+1)*TILES*K256//OWNERS)-(owner*TILES*K256//OWNERS) for owner in range(OWNERS)}),
      "segment_lengths":sorted({r["k_end"]-r["k_begin"] for r in ownership}),
      "segment_count":len(ownership),"max_segments_per_tile":max_segments,
      "segment_census":{str(n):sum(len(x)==n for x in slots) for n in range(1,max_segments+1)}},
    "correctness":legacy_correctness,
    "timing":{"main":windows(main),"fixup":windows(fix_samples),"pair":windows(pair)},
    "arms":arm_results,
    "paired":{"alternated_call_order":True,"rounds":args.rounds,
      "main_direct_minus_factored_us":windows(paired_main),"pair_direct_minus_factored_us":windows(paired_pair),
      "factored_main_wins":sum(x>0 for x in paired_main),"factored_pair_wins":sum(x>0 for x in paired_pair)},
    "direct_vs_factored":{"exact":bool(np.array_equal(arm_outputs["direct_da"],arm_outputs["factored_da"])),
      "max_abs":float(arm_diff.max()),"mean_abs":float(arm_diff.mean())},
    "baselines":{"llama_main_us":LLAMA_MAIN_US,"llama_pair_us":LLAMA_MAIN_US+LLAMA_FIXUP_US,
      "llama_main_5pct_us":LLAMA_MAIN_US*1.05,"llama_pair_5pct_us":(LLAMA_MAIN_US+LLAMA_FIXUP_US)*1.05},
    "comparison":{"main_median_us":main_med,"pair_median_us":statistics.median(pair),
      "main_vs_llama_ratio":main_med/LLAMA_MAIN_US,"pair_vs_llama_ratio":statistics.median(pair)/(LLAMA_MAIN_US+LLAMA_FIXUP_US),
      "main_within_5pct":main_med<=LLAMA_MAIN_US*1.05,"pair_within_5pct":statistics.median(pair)<=(LLAMA_MAIN_US+LLAMA_FIXUP_US)*1.05,
      "direct_main_median_us":statistics.median(direct_samples["main"]),
      "direct_pair_median_us":statistics.median(direct_samples["pair"]),
      "paired_main_direct_minus_factored_median_us":statistics.median(paired_main),
      "paired_pair_direct_minus_factored_median_us":statistics.median(paired_pair)},
    "compiler":{"full":arm_states["factored_da"]["compiler"]["full"],
      "segment0_diagnostic":arm_states["factored_da"]["compiler"]["segment0_diagnostic"],
      "segment1_diagnostic":arm_states["factored_da"]["compiler"]["segment1_diagnostic"],
      "fixup":fix_compiler,"direct":direct},
    "reference":{"kind":"compiler_wide_direct","trusted":bool(direct["passed"]),"result":direct},
    "gpu_lock":{"mode":"outer_flock_required","path":"/tmp/nv-q6-oracle-gpu.lock"},"passed":False}
  result["passed"]=bool(direct["passed"] and all(c["finite"] and c["gpu_fixup_cpu_exact"] and
    c["reference_failing_count_rtol2e5_atol2e3"]==0 for c in arm_correctness.values()))
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
