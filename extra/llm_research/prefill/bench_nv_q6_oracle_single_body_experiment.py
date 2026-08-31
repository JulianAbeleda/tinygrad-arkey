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
from extra.llm_research.prefill.nv_q6_oracle_single_body_experiment import q6_oracle_single_body_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

M,N,K,OWNERS,K256=512,4096,12288,170,48
TILES_M,TILES_N,TILES=4,32,128
TILE_ELEMS=ROWS*COLS
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
LLAMA_MAIN_US,LLAMA_FIXUP_US=201.216,8.640


def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")
def _stats(x:list[float]): return {"samples_us":x,"min_us":min(x),"median_us":statistics.median(x),"max_us":max(x)}


def _render(segment:int, artifacts:pathlib.Path):
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_oracle_broad_cta_kernel(ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),prefetch_second_panel=True,factor_dA=True,
    oracle_publisher=True,depth=37,streamk_owners=OWNERS,streamk_segment=segment)
  started=time.perf_counter(); program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE); render_ms=(time.perf_counter()-started)*1e3
  name=f"nv_q6_oracle_broad_cta_prefetch_factor_da_oracle_publisher_streamk_s{segment}"
  path=artifacts/f"{name}.cu"; path.write_text(source)
  started=time.perf_counter(); binary=Device["NV"].compiler.compile(source); compile_ms=(time.perf_counter()-started)*1e3
  cubin=artifacts/f"{name}.cubin"; cubin.write_bytes(binary); census=analyze_cubin(cubin,artifacts/f"sass_s{segment}",name)["summary"]
  return name,binary,{"render_ms":render_ms,"compile_ms":compile_ms,"source":str(path),"source_bytes":len(source),
    "cubin":str(cubin),"cubin_sha256":hashlib.sha256(binary).hexdigest(),"sass":census}


def _combine_sources(source0:str,source1:str,artifacts:pathlib.Path):
  pattern=r'extern "C" __global__ void __launch_bounds__\(256\) \w+\((.*?)\) \{\n  int gidx0 = blockIdx.x; /\* 170 \*/\n'
  parts=[]; preamble=None; params=None
  for index,source in enumerate((source0,source1)):
    match=re.search(pattern,source,re.DOTALL)
    if match is None: raise RuntimeError(f"generated segment {index} signature mismatch")
    if preamble is None: preamble=source[:match.start()]; params=match.group(1)
    elif params!=match.group(1): raise RuntimeError("generated segment ABIs differ")
    parts.append(f"__device__ __forceinline__ void q6_segment_{index}({match.group(1)}, int gidx0) {{\n"+source[match.end():])
  assert preamble is not None and params is not None
  name="nv_q6_oracle_broad_streamk_170"
  wrapper=(preamble+"\n".join(parts)+f'''\nextern "C" __global__ void __launch_bounds__(256) {name}({params}) {{
    int owner=blockIdx.x;
    q6_segment_0(data0_5570560,data1_20643840,data2_1769472,owner);
    __syncthreads();
    q6_segment_1(data0_5570560,data1_20643840,data2_1769472,owner);
  }}\n''')
  path=artifacts/f"{name}.cu";path.write_text(wrapper);started=time.perf_counter();binary=Device["NV"].compiler.compile(wrapper)
  compile_ms=(time.perf_counter()-started)*1e3;cubin=artifacts/f"{name}.cubin";cubin.write_bytes(binary)
  census=analyze_cubin(cubin,artifacts/"sass_full",name)["summary"]
  return name,binary,{"compile_ms":compile_ms,"source":str(path),"source_bytes":len(wrapper),"cubin":str(cubin),
    "cubin_sha256":hashlib.sha256(binary).hexdigest(),"sass":census}


def _render_single_body(artifacts:pathlib.Path):
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_oracle_single_body_kernel(ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),streamk_owners=OWNERS)
  started=time.perf_counter();program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE);render_ms=(time.perf_counter()-started)*1e3
  name="nv_q6_oracle_prefetch_factor_da_oracle_publisher_streamk_single_body"
  path=artifacts/f"{name}.cu";path.write_text(source);started=time.perf_counter();binary=Device["NV"].compiler.compile(source)
  compile_ms=(time.perf_counter()-started)*1e3;cubin=artifacts/f"{name}.cubin";cubin.write_bytes(binary)
  census=analyze_cubin(cubin,artifacts/"sass_single_body",name)["summary"]
  return name,binary,{"render_ms":render_ms,"compile_ms":compile_ms,"source":str(path),"source_bytes":len(source),
    "cubin":str(cubin),"cubin_sha256":hashlib.sha256(binary).hexdigest(),"runtime_segment_grid_axis":True,"sass":census}


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

  n0,b0,c0=_render(0,args.artifacts); n1,b1,c1=_render(1,args.artifacts)
  baseline_name,baseline_binary,baseline_compiler=_combine_sources(pathlib.Path(c0["source"]).read_text(),pathlib.Path(c1["source"]).read_text(),args.artifacts)
  candidate_name,candidate_binary,candidate_compiler=_render_single_body(args.artifacts)
  baseline_program=NVProgram(Device["NV"],baseline_name,baseline_binary,shared_mem=LAUNCH_SHARED_BYTES)
  candidate_program=NVProgram(Device["NV"],candidate_name,candidate_binary,shared_mem=LAUNCH_SHARED_BYTES)
  baseline_partials=Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize()
  candidate_partials=Tensor.full((2*OWNERS*TILE_ELEMS,),float("nan"),device="NV").contiguous().realize()
  baseline_program(_buf(baseline_partials),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  candidate_program(_buf(candidate_partials),_buf(halfs),_buf(q8),global_size=(OWNERS,2,1),local_size=(256,1,1),wait=True,timeout=120000)

  slots,ownership=_ownership(); max_segments=max(map(len,slots)); slot_map=np.full((TILES,max_segments),-1,np.int32)
  for tile,tile_slots in enumerate(slots): slot_map[tile,:len(tile_slots)]=tile_slots
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  fix_source=_fixup_source(max_segments); (args.artifacts/"fixup.cu").write_text(fix_source)
  fix_binary=compiler.compile(fix_source); fix_cubin=args.artifacts/"fixup.cubin"; fix_cubin.write_bytes(fix_binary)
  fix_census=analyze_cubin(fix_cubin,args.artifacts/"sass_fixup","q6_oracle_fixup")["summary"]
  fix=NVProgram(Device["NV"],"q6_oracle_fixup",fix_binary)
  baseline_output=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  candidate_output=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  fix(_buf(baseline_output),_buf(baseline_partials),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)
  fix(_buf(candidate_output),_buf(candidate_partials),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)
  baseline_got=baseline_output.numpy(); got=candidate_output.numpy()
  baseline_raw=baseline_partials.numpy();candidate_raw=candidate_partials.numpy();raw=candidate_raw.reshape(2*OWNERS,ROWS,COLS)
  cpu=np.empty((M,N),np.float32)
  for tile,tile_slots in enumerate(slots):
    mt,nt=tile%TILES_M,tile//TILES_M; reduced=sum((raw[s] for s in tile_slots[1:]),raw[tile_slots[0]].copy())
    cpu[mt*COLS:(mt+1)*COLS,nt*ROWS:(nt+1)*ROWS]=reduced.T
  fix_diff=np.abs(got-cpu); ref_diff=np.abs(got-expected); baseline_diff=np.abs(got-baseline_got)

  baseline_samples=[];candidate_samples=[];fix_samples=[]
  for round_idx in range(args.rounds):
    calls=((baseline_program,baseline_partials),(candidate_program,candidate_partials)) if round_idx%2==0 else \
          ((candidate_program,candidate_partials),(baseline_program,baseline_partials))
    for program,partials in calls:
      grid=(OWNERS,2,1) if program is candidate_program else (OWNERS,1,1)
      sample=program(_buf(partials),_buf(halfs),_buf(q8),global_size=grid,local_size=(256,1,1),wait=True,timeout=120000)*1e6
      (candidate_samples if program is candidate_program else baseline_samples).append(sample)
    fix_samples.append(fix(_buf(candidate_output),_buf(candidate_partials),_buf(slot_map_t),global_size=(TILES,1,1),local_size=(256,1,1),wait=True)*1e6)
  def windows(xs): return {"r9":_stats(xs[:9]),"r31":_stats(xs)}
  recoveries=[base-candidate for base,candidate in zip(baseline_samples,candidate_samples)]
  pair=[a+b for a,b in zip(candidate_samples,fix_samples)]; main_med=statistics.median(candidate_samples)
  candidate_wins=sum(candidate<base for base,candidate in zip(baseline_samples,candidate_samples))
  result={"schema":"tinygrad.nv_q6_oracle_single_body_experiment.v1","shape":{"M":M,"N":N,"K":K},"owners":OWNERS,
    "ownership":{"work_units":TILES*K256,"owner_work_lengths":sorted({((owner+1)*TILES*K256//OWNERS)-(owner*TILES*K256//OWNERS) for owner in range(OWNERS)}),
      "segment_lengths":sorted({r["k_end"]-r["k_begin"] for r in ownership}),
      "segment_count":len(ownership),"max_segments_per_tile":max_segments,
      "segment_census":{str(n):sum(len(x)==n for x in slots) for n in range(1,max_segments+1)}},
    "correctness":{"finite":bool(np.isfinite(got).all()),"candidate_partials_baseline_exact":bool(np.array_equal(candidate_raw,baseline_raw,equal_nan=True)),
      "candidate_fixup_baseline_exact":bool(np.array_equal(got,baseline_got)),"candidate_fixup_baseline_max_abs":float(baseline_diff.max()),
      "gpu_fixup_cpu_exact":bool(np.array_equal(got,cpu)),
      "gpu_fixup_cpu_max_abs":float(fix_diff.max()),"reference_max_abs":float(ref_diff.max()),"reference_mean_abs":float(ref_diff.mean()),
      "reference_allclose_rtol2e5_atol2e3":bool(np.allclose(got,expected,rtol=2e-5,atol=2e-3))},
    "timing":{"baseline_main":windows(baseline_samples),"candidate_main":windows(candidate_samples),
      "paired_recovery":windows(recoveries),"candidate_wins":candidate_wins,"alternated_call_order":True,
      "fixup":windows(fix_samples),"candidate_pair":windows(pair)},
    "baselines":{"llama_main_us":LLAMA_MAIN_US,"llama_pair_us":LLAMA_MAIN_US+LLAMA_FIXUP_US,
      "llama_main_5pct_us":LLAMA_MAIN_US*1.05,"llama_pair_5pct_us":(LLAMA_MAIN_US+LLAMA_FIXUP_US)*1.05},
    "comparison":{"baseline_main_median_us":statistics.median(baseline_samples),"candidate_main_median_us":main_med,
      "paired_recovery_median_us":statistics.median(recoveries),"pair_median_us":statistics.median(pair),
      "main_vs_llama_ratio":main_med/LLAMA_MAIN_US,"pair_vs_llama_ratio":statistics.median(pair)/(LLAMA_MAIN_US+LLAMA_FIXUP_US),
      "main_within_5pct":main_med<=LLAMA_MAIN_US*1.05,"pair_within_5pct":statistics.median(pair)<=(LLAMA_MAIN_US+LLAMA_FIXUP_US)*1.05},
    "compiler":{"baseline":baseline_compiler,"candidate":candidate_compiler,"fixup":{"sass":fix_census},"direct":direct},
    "gpu_lock":{"path":"/tmp/nv-q6-oracle-gpu.lock","used":True},"passed":False,"promote":False}
  result["passed"]=bool(result["correctness"]["finite"] and result["correctness"]["candidate_partials_baseline_exact"] and
    result["correctness"]["candidate_fixup_baseline_exact"] and result["correctness"]["gpu_fixup_cpu_exact"])
  result["promote"]=bool(result["passed"] and candidate_wins>=24 and statistics.median(recoveries)>0)
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
