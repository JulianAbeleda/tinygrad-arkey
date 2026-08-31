#!/usr/bin/env python3
"""Marginal K256 cost qualifier for the persistent broad Q6 oracle body."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record, _stats
from extra.llm_research.prefill.bench_nv_q6_oracle_cta_sweep import _fixture
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import COLS,K,ROWS,SHARED_BYTES,q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

DEPTHS=(1,2,4,8,16,36,37)
LLAMA_GROSS_EPOCH_US=5.57
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024


def _fit(points:list[tuple[int,float]]) -> dict[str,float]:
  slopes=[(yj-yi)/(xj-xi) for i,(xi,yi) in enumerate(points) for xj,yj in points[i+1:]]
  slope=float(statistics.median(slopes))
  intercept=float(statistics.median([y-slope*x for x,y in points]))
  xs=np.asarray([x for x,_ in points],np.float64); ys=np.asarray([y for _,y in points],np.float64)
  ols_slope,ols_intercept=np.polyfit(xs,ys,1)
  residuals=[float(y-(intercept+slope*x)) for x,y in points]
  return {"method":"theil_sen_pairwise_median","intercept_us":intercept,"slope_us_per_k256":slope,
    "ols_intercept_us":float(ols_intercept),"ols_slope_us_per_k256":float(ols_slope),
    "max_abs_residual_us":max(map(abs,residuals)),"median_abs_residual_us":float(statistics.median(map(abs,residuals)))}


def _one(depth:int, rounds:int, root:pathlib.Path, *, prefetch:bool=True, factor_dA:bool=True,
         combined_initial:bool=False) -> dict[str,object]:
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_oracle_broad_cta_kernel(ph(ROWS*COLS,dtypes.float32,0),ph(depth*ROWS*105,dtypes.uint16,1),
    ph(depth*2*COLS*36,dtypes.uint32,2),prefetch_second_panel=prefetch,factor_dA=factor_dA,
    combined_initial_publish=combined_initial,oracle_publisher=True,depth=depth)
  render_start=time.perf_counter(); program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE); render_ms=(time.perf_counter()-render_start)*1e3
  suffix="prefetch" if prefetch else "serial"
  if combined_initial: suffix += "_combined_publish"
  if factor_dA: suffix += "_factor_da"
  name=f"nv_q6_oracle_broad_cta_{suffix}_oracle_publisher_d{depth}"
  arm_dir=root/f"depth_{depth}"; arm_dir.mkdir(parents=True,exist_ok=True)
  source_path=arm_dir/f"{name}.cu"; source_path.write_text(source)
  compile_start=time.perf_counter(); binary=Device["NV"].compiler.compile(source); compile_ms=(time.perf_counter()-compile_start)*1e3
  cubin_path=arm_dir/f"{name}.cubin"; cubin_path.write_bytes(binary)
  census=analyze_cubin(cubin_path,arm_dir/"sass",name)["summary"]

  blocks,q,scales,reference=_fixture(depth,COLS)
  epoch_blocks=np.ascontiguousarray(blocks.transpose(1,0,2)).view(np.uint16).reshape(-1)
  record=np.concatenate([_record(q[e*K:(e+1)*K],scales[e]) for e in range(depth)],axis=0).reshape(-1)
  arrays=(np.full(ROWS*COLS,np.nan,np.float32),epoch_blocks,record)
  dev=Device["NV"]; bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in arrays]
  for buf,array in zip(bufs,arrays): dev.allocator._copyin(buf,memoryview(array.tobytes()))
  runner=NVProgram(dev,name,binary,shared_mem=LAUNCH_SHARED_BYTES)
  runner(*bufs,global_size=(1,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  raw=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(raw,bufs[0])
  got=np.frombuffer(raw,np.float32,count=ROWS*COLS).reshape(ROWS,COLS); diff=np.abs(got-reference)
  timing=_stats([runner(*bufs,global_size=(1,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6 for _ in range(rounds)])
  resources=census["resources"] or {}; exact=bool(np.array_equal(got,reference))
  spill_free=bool(resources.get("stack_bytes",1)==0 and resources.get("local_static_bytes",1)==0 and
                  all(region.get("LDL",0)==region.get("STL",0)==0 for region in census["spill_regions"].values()) and
                  census["families"].get("LDL",0)==0 and census["families"].get("STL",0)==0)
  return {"depth":depth,"shape":{"M":ROWS,"N":COLS,"K":depth*K,"block":[256,1,1],"grid":[1,1,1]},
    "correctness":{"exact":exact,"finite":bool(np.isfinite(got).all()),"max_abs":float(diff.max()),"mean_abs":float(diff.mean())},
    "timing":timing,"compiler":{"render_ms":render_ms,"compile_wall_ms":compile_ms,"source":str(source_path),
      "source_bytes":len(source),"cubin":str(cubin_path),"cubin_sha256":hashlib.sha256(binary).hexdigest()},
    "sass":{"instruction_total":census["instruction_total"],"families":census["families"],"resources":resources,
      "spill_regions":census["spill_regions"]},"spill_free":spill_free,
    "feasible":bool(exact and spill_free and resources.get("registers",256)<=255 and LAUNCH_SHARED_BYTES<=58_880)}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--rounds",type=int,default=31)
  ap.add_argument("--no-factor-da",action="store_true"); ap.add_argument("--serial-q8",action="store_true")
  ap.add_argument("--combined-initial",action="store_true")
  ap.add_argument("--out",type=pathlib.Path,required=True); ap.add_argument("--artifacts",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.rounds < 31: raise ValueError("depth qualification requires R31 or greater")
  args.artifacts.mkdir(parents=True,exist_ok=True)
  arms=[_one(depth,args.rounds,args.artifacts,prefetch=not args.serial_q8,factor_dA=not args.no_factor_da,
             combined_initial=args.combined_initial) for depth in DEPTHS]
  fit=_fit([(int(x["depth"]),float(x["timing"]["median_us"])) for x in arms])
  investment_bar=LLAMA_GROSS_EPOCH_US*1.05
  passed=bool(all(x["correctness"]["exact"] and x["sass"]["resources"].get("registers",256)<=255 for x in arms) and
              fit["slope_us_per_k256"]<=investment_bar)
  result={"schema":"tinygrad.nv_q6_oracle_depth.v1","depths":list(DEPTHS),"rounds":args.rounds,
    "contract":{"publisher":"straight_line_oracle","q8":"serial" if args.serial_q8 else "rolling_prefetch",
      "arithmetic":"direct" if args.no_factor_da else "factor_dA","combined_initial_publish":args.combined_initial,
      "timing_model":"T(depth)=intercept+slope*depth; never T(1)*depth"},
    "fit":fit,"gate":{"llama_gross_epoch_us":LLAMA_GROSS_EPOCH_US,"investment_bar_5pct_us":investment_bar,"passed":passed,
      "decision":"INVEST_FULL_170_OWNER_ROUTE" if passed else "NO_GO_PERSISTENT_BROAD_ROUTE"},"arms":arms}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True)); return 0 if all(x["correctness"]["exact"] for x in arms) else 1


if __name__ == "__main__": raise SystemExit(main())
