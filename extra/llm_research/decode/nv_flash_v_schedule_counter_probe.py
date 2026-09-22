#!/usr/bin/env python3
"""Cold/hot NCU discriminator for surviving typed V schedule constructions."""
from __future__ import annotations

import argparse, contextlib, json, os, pathlib, statistics, subprocess, tempfile

from tinygrad import Context
from extra.llm_research.decode.nv_flash_bounded_counter_probe import (HARNESS, METRICS, NVCC, _ncu, _render,
  _sass_load_grammar, _unroll_column_source)
from extra.llm_research.decode.nv_flash_load_wall_probe import _ptxas_resources


def _async_tail_source(source:str,symbol:str,columns:int=1)->tuple[str,str]:
  """Move the selected late V column through warp-private async shared staging."""
  new_symbol=symbol+"_async"
  source=source.replace(symbol+"(",new_symbol+"(",1)
  anchor="  int alu11 = (alu2<<2);\n"
  issue=f'''  __shared__ __align__(16) uint4 async_vtail[{256*columns}];
  int async_tid = threadIdx.y*32+threadIdx.x;
  int async_alu27 = (alu9+alu10+alu11);
'''
  for ci,j in enumerate(range(8-columns,8)):
    for half in range(2):
      idx=f"((async_tid*{columns}+{ci})*2+{half})";off=524288+j*64+half*32
      issue+=f'  unsigned async_s{ci}_{half} = (unsigned)__cvta_generic_to_shared(async_vtail+{idx});\n'
      issue+=f'  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(async_s{ci}_{half}), "l"(data2_1048576+(async_alu27+{off})));\n'
  issue+='  asm volatile("cp.async.commit_group;");\n'
  if source.count(anchor)!=1:raise RuntimeError("could not locate async issue anchor")
  source=source.replace(anchor,anchor+issue,1)
  loads="";wait='  asm volatile("cp.async.wait_group 0;");\n'
  for ci,j in enumerate(range(8-columns,8)):
    for half in range(2):
      val=3+ci*2+half;off=524288+j*64+half*32;idx=f"((async_tid*{columns}+{ci})*2+{half})"
      loads+=f'  uint4 val{val} = (*((uint4*)((data2_1048576+(alu27+{off})))));\n'
      wait+=f'  uint4 val{val} = async_vtail[{idx}];\n'
  if source.count(loads)!=1:raise RuntimeError("could not locate selected V tail loads")
  return new_symbol,source.replace(loads,wait,1)


def _prefetch_v_source(source:str,symbol:str,columns:int)->tuple[str,str]:
  """Have one lane per V row-group prefetch selected late columns into L2."""
  new_symbol=f"{symbol}_vprefetch{columns}"
  source=source.replace(symbol+"(",new_symbol+"(",1)
  anchor="  int alu11 = (alu2<<2);\n"
  lines=["  if (alu2 == 0) {"]
  for j in range(8-columns,8):
    for half in (0,32):
      off=524288+j*64+half
      lines.append(f'    asm volatile("prefetch.global.L2 [%0];" :: "l"(data2_1048576+(alu9+alu10+{off})));')
  lines.append("  }")
  issue="\n".join(lines)+"\n"
  if source.count(anchor)!=1:raise RuntimeError("could not locate V prefetch anchor")
  return new_symbol,source.replace(anchor,anchor+issue,1)

def _warp_probability_barriers(source:str,symbol:str)->tuple[str,str]:
  """The first two barriers in shared-probability arms are warp-private exchanges; keep the final CTA combine barrier."""
  new_symbol=symbol+"_warpbar"
  source=source.replace(symbol+"(",new_symbol+"(",1)
  count=source.count("  __syncthreads();")
  if count not in (2,3):raise RuntimeError("expected one/two probability barriers plus final CTA barrier")
  return new_symbol,source.replace("  __syncthreads();","  __syncwarp();",count-1)


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("warpprob","warpprob_unroll","warpprob_vtail1","vtail1","vdimmajor","vsharedprob","vsharedprob_warp","vsharedprob_vdim","vllama_topology","vllama_topology_warp","vasync1","vasync1_installed","vasync2","vasync4","vprefetch1","vprefetch2","vprefetch4","vevictlast","vstaticpersist"),required=True)
  ap.add_argument("--passes",type=int,default=500);ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--persist-numerator",type=int,default=15)
  ap.add_argument("--artifacts-dir",type=pathlib.Path);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  with Context(NV_FLASH_LOAD_SCHEDULE=1):
    os.environ.pop("NV_FLASH_KV_EVICT_LAST",None)
    control,control_src=_render(6,768,v_pipeline_tail=1) if a.arm=="vasync1_installed" else _render(6,768)
    if a.arm in ("warpprob","warpprob_unroll"): candidate,candidate_src=_render(6,768,warp_probability_ownership=True)
    elif a.arm=="warpprob_vtail1": candidate,candidate_src=_render(6,768,warp_probability_ownership=True,v_pipeline_tail=1)
    elif a.arm=="vtail1" or a.arm.startswith("vasync"): candidate,candidate_src=_render(6,768,v_pipeline_tail=int(a.arm[6]) if a.arm.startswith("vasync") else 1)
    elif a.arm in ("vsharedprob","vsharedprob_warp"): candidate,candidate_src=_render(6,768,shared_probability_ownership=True)
    elif a.arm=="vsharedprob_vdim": candidate,candidate_src=_render(6,768,v_pipeline_tail=8,v_dimension_major=True,shared_probability_ownership=True)
    elif a.arm in ("vllama_topology","vllama_topology_warp"): candidate,candidate_src=_render(6,768,v_pipeline_tail=8,v_dimension_major=True,shared_probability_ownership=True,packed_pv_f16=True)
    elif a.arm in ("vevictlast","vstaticpersist"):
      env="NV_FLASH_KV_EVICT_LAST" if a.arm=="vevictlast" else "NV_FLASH_KV_STATIC_PERSISTING"
      os.environ[env]=str(a.persist_numerator) if a.arm=="vstaticpersist" else "1";candidate,candidate_src=_render(6,768,v_pipeline_tail=1)
      renamed=candidate+("_evictlast" if a.arm=="vevictlast" else "_staticpersist");candidate_src=candidate_src.replace(candidate+"(",renamed+"(",1);candidate=renamed
      os.environ.pop(env,None)
    elif a.arm.startswith("vprefetch"): candidate,candidate_src=_render(6,768)
    else: candidate,candidate_src=_render(6,768,v_pipeline_tail=8,v_dimension_major=True)
  if a.arm.startswith("vasync"):candidate,candidate_src=_async_tail_source(candidate_src,candidate,int(a.arm[6]))
  elif a.arm=="warpprob_unroll":candidate,candidate_src,_=_unroll_column_source(candidate_src,candidate)
  elif a.arm.startswith("vprefetch"):candidate,candidate_src=_prefetch_v_source(candidate_src,candidate,int(a.arm[-1]))
  elif a.arm in ("vsharedprob_warp","vllama_topology_warp"):candidate,candidate_src=_warp_probability_barriers(candidate_src,candidate)
  if a.arm in ("vevictlast","vstaticpersist"):
    helper_start=candidate_src.index("__device__ __forceinline__ uint4 tg_nv_ld_evict_last")
    helper_end=candidate_src.index("\n\n#define INFINITY",helper_start)
    candidate_src=candidate_src[helper_start:helper_end]+"\n"+candidate_src[candidate_src.index('extern "C"'):]
  else:candidate_src=candidate_src[candidate_src.index('extern "C"'):]
  source=HARNESS.replace("__CONTROL_SOURCE__",control_src).replace("__CANDIDATE_SOURCE__",candidate_src)
  source=source.replace("__CONTROL__",control).replace("__CANDIDATE__",candidate)
  source=source.replace("__CONTROL_SPLITS__","6").replace("__CAND_SPLITS__","6")
  source=source.replace("__CAND_OUT__",str(32*6*130)).replace("__COMPARE__","1")
  if a.artifacts_dir is not None:a.artifacts_dir.mkdir(parents=True,exist_ok=True)
  wd=contextlib.nullcontext(str(a.artifacts_dir)) if a.artifacts_dir is not None else tempfile.TemporaryDirectory(prefix="nv_flash_v_sched_")
  with wd as td:
    cu,binary=pathlib.Path(td)/"probe.cu",pathlib.Path(td)/"probe";cu.write_text(source)
    build=subprocess.run([NVCC,"-arch=sm_120a","-O3","-lineinfo","-std=c++17","--ptxas-options=-v",str(cu),"-o",str(binary)],capture_output=True,text=True)
    if build.returncode:raise RuntimeError(build.stderr[-12000:])
    run=subprocess.run([str(binary),str(a.passes),str(a.reps)],capture_output=True,text=True,check=True)
    counters={arm:{state:_ncu(binary,symbol,"none" if state=="hot" else "all") for state in ("hot","cold")}
      for arm,symbol in (("control",control),("candidate",candidate))}
  controls=[];candidates=[];exact=None
  import re
  for line in run.stdout.splitlines():
    if m:=re.match(r"exact_mismatches=(\d+) max_abs=([0-9.eE+-]+)",line):exact={"bit_mismatches":int(m.group(1)),"max_abs":float(m.group(2))}
    if m:=re.match(r"rep=\d+ control=([0-9.]+) candidate=([0-9.]+)",line):controls.append(float(m.group(1)));candidates.append(float(m.group(2)))
  result={"schema":"tinygrad.nv_flash_v_schedule_counter_probe.v1","arm":a.arm,"shape":{"splits":6,"token_bound":768},
    "exact":exact,"control":{"symbol":control,"median_us":statistics.median(controls),"samples_us":controls,
      "resources":_ptxas_resources(build.stderr,control),"sass":_sass_load_grammar(binary,control)},
    "candidate":{"symbol":candidate,"median_us":statistics.median(candidates),"samples_us":candidates,
      "resources":_ptxas_resources(build.stderr,candidate),"sass":_sass_load_grammar(binary,candidate)},
    "ncu":{"metrics":METRICS,"arms":counters}}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
