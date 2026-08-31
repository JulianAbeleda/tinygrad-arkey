#!/usr/bin/env python3
"""Exact oracle-shaped Q6_K CTA topology sweep.

This is a research qualifier, not a route selector.  It holds the canonical
Q6_K/Q8 inputs and 8-warp ownership fixed while sweeping K-block and output
column reuse.  A candidate is investable only when it is exact, spill-free,
launch-feasible, and its normalized work projects at least the configured
full-main recovery against the fresh pinned geometry.
"""
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
from extra.llm_research.prefill.nv_native_fragment_k16_gate import q6_packed_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

ROWS, BLOCK_K = 128, 256
PINNED_K_BLOCKS, PINNED_COL_GROUPS = 2, 1
PINNED_HISTORICAL_US = 8.176
CURRENT_WIDE_MAIN_US = 318.8
LLAMA_MAIN_US = 201.216


def _stats(samples:list[float]) -> dict[str, object]:
  return {"samples_us":samples, "min_us":min(samples), "median_us":statistics.median(samples), "max_us":max(samples)}


def _placeholders(k_blocks:int, cols:int) -> tuple[UOp, ...]:
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  return (ph(ROWS*cols,dtypes.float32,0), ph(ROWS*k_blocks*105,dtypes.uint16,1),
          ph(k_blocks*BLOCK_K*cols,dtypes.int8,2), ph(k_blocks*8*cols,dtypes.float32,3))


def _fixture(k_blocks:int, cols:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng=np.random.default_rng(20260907 + 100*k_blocks + cols)
  blocks=rng.integers(0,256,(ROWS,k_blocks,210),dtype=np.uint8)
  blocks[:,:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8)
  activation=rng.integers(-4,5,(k_blocks*BLOCK_K,cols),dtype=np.int8)
  activation_scale=np.full((k_blocks,8,cols),.0625,np.float32)
  reference=np.zeros((ROWS,cols),np.float32)
  for block in range(k_blocks):
    raw=blocks[:,block]; quant=np.empty((ROWS,16,16),np.int8)
    for group in range(16):
      half,pgrp=group//8,group%8; ql=half*64+(pgrp%4)*16; qh=half*32+(pgrp%2)*16
      quant[:,group]=((((raw[:,ql:ql+16]>>(4 if pgrp>=4 else 0))&15)|
        (((raw[:,128+qh:128+qh+16]>>((pgrp//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
    scales=raw[:,192:208].view(np.int8)
    for pair in range(8):
      dot0=quant[:,2*pair].astype(np.int32)@activation[block*BLOCK_K+32*pair:block*BLOCK_K+32*pair+16].astype(np.int32)
      dot1=quant[:,2*pair+1].astype(np.int32)@activation[block*BLOCK_K+32*pair+16:block*BLOCK_K+32*pair+32].astype(np.int32)
      reference += (.03125*activation_scale[block,pair])*(scales[:,2*pair,None].astype(np.float32)*dot0+
        scales[:,2*pair+1,None].astype(np.float32)*dot1)
  return blocks,activation,activation_scale,reference


def _copyout(dev, buf, count:int) -> np.ndarray:
  raw=memoryview(bytearray(buf.size)); dev.allocator._copyout(raw,buf)
  return np.frombuffer(raw,np.float32,count=count).copy()


def _one(k_blocks:int, col_groups:int, rounds:int, artifacts:pathlib.Path) -> dict[str, object]:
  cols=16*col_groups; name=f"nv_native_fragment_q6_cta_128x{cols}x{k_blocks*BLOCK_K}"
  ast=q6_packed_cta_kernel(*_placeholders(k_blocks,cols),k_blocks,col_groups=col_groups)
  render_start=time.perf_counter()
  program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  render_ms=(time.perf_counter()-render_start)*1e3
  shape_dir=artifacts/f"k{k_blocks}_cg{col_groups}"; shape_dir.mkdir(parents=True,exist_ok=True)
  source_path=shape_dir/f"{name}.cu"; source_path.write_text(source)
  compile_start=time.perf_counter(); binary=Device["NV"].compiler.compile(source); compile_ms=(time.perf_counter()-compile_start)*1e3
  cubin_path=shape_dir/f"{name}.cubin"; cubin_path.write_bytes(binary)
  census=analyze_cubin(cubin_path,shape_dir/"sass",name)["summary"]

  blocks,activation,activation_scale,reference=_fixture(k_blocks,cols)
  dev=Device["NV"]; host=(np.empty(ROWS*cols,np.float32),blocks,activation,activation_scale)
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,array in zip(bufs[1:],host[1:]): dev.allocator._copyin(buf,memoryview(array.tobytes()))
  runner=NVProgram(dev,name,binary)
  runner(*bufs,global_size=(1,1,1),local_size=(256,1,1),wait=True)
  got=_copyout(dev,bufs[0],ROWS*cols).reshape(ROWS,cols)
  samples=[runner(*bufs,global_size=(1,1,1),local_size=(256,1,1),wait=True)*1e6 for _ in range(rounds)]
  timing=_stats(samples); median=float(timing["median_us"]); work=ROWS*cols*k_blocks*BLOCK_K
  resources=census["resources"] or {}
  exact=bool(np.array_equal(got,reference)); diff=np.abs(got-reference)
  return {"shape":{"rows":ROWS,"cols":cols,"k":k_blocks*BLOCK_K,"k_blocks":k_blocks,"col_groups":col_groups,
                   "block":[256,1,1],"grid":[1,1,1]},
    "correctness":{"exact":exact,"finite":bool(np.isfinite(got).all()),"max_abs":float(diff.max()),"mean_abs":float(diff.mean())},
    "timing":timing,"normalization":{"output_elements_times_k":work,"median_us_per_output_element_k":median/work,
      "giga_output_element_k_per_s":work/(median*1e3)},
    "compiler":{"render_ms":render_ms,"compile_wall_ms":compile_ms,"source":str(source_path),"cubin":str(cubin_path),
      "cubin_sha256":hashlib.sha256(binary).hexdigest(),"source_wmma_calls":source.count("__WMMA_8_16_16_signed_char_int(")},
    "sass":{"instruction_total":census["instruction_total"],"families":census["families"],"resources":resources,
      "spill_regions":census["spill_regions"]},
    "feasible":bool(exact and resources and resources.get("stack_bytes")==0 and resources.get("local_static_bytes")==0 and
      resources.get("registers",256)<=255 and resources.get("shared_static_bytes",99_999)<=48*1024)}


def main() -> int:
  parser=argparse.ArgumentParser()
  parser.add_argument("--rounds",type=int,default=9)
  parser.add_argument("--out",type=pathlib.Path,required=True)
  parser.add_argument("--artifacts",type=pathlib.Path,required=True)
  parser.add_argument("--required-recovery-us",type=float,default=23.5)
  args=parser.parse_args()
  if args.rounds < 9: raise ValueError("qualification requires R9 or greater")
  args.artifacts.mkdir(parents=True,exist_ok=True)
  arms=[_one(kb,cg,args.rounds,args.artifacts) for kb in (1,2) for cg in (1,2,4)]
  pinned=next(x for x in arms if x["shape"]["k_blocks"]==PINNED_K_BLOCKS and x["shape"]["col_groups"]==PINNED_COL_GROUPS)
  pinned_norm=float(pinned["normalization"]["median_us_per_output_element_k"])
  for arm in arms:
    ratio=float(arm["normalization"]["median_us_per_output_element_k"])/pinned_norm
    projected=CURRENT_WIDE_MAIN_US*ratio
    arm["projection"]={"normalized_ratio_vs_fresh_pinned":ratio,"projected_full_main_us":projected,
      "projected_recovery_us":CURRENT_WIDE_MAIN_US-projected,"projected_gap_vs_llama_us":projected-LLAMA_MAIN_US}
  feasible=[x for x in arms if x["feasible"]]
  best=min(feasible,key=lambda x:float(x["normalization"]["median_us_per_output_element_k"])) if feasible else None
  recovery=float(best["projection"]["projected_recovery_us"]) if best else float("-inf")
  passed=bool(best and recovery>=args.required_recovery_us)
  result={"schema":"tinygrad.nv_q6_oracle_cta_sweep.v1","rounds":args.rounds,
    "baselines":{"historical_pinned":{"shape":"128x16xK512","median_us":PINNED_HISTORICAL_US},
      "fresh_pinned_median_us":pinned["timing"]["median_us"],"current_wide_main_us":CURRENT_WIDE_MAIN_US,"llama_main_us":LLAMA_MAIN_US},
    "gate":{"required_projected_recovery_us":args.required_recovery_us,"passed":passed,
      "decision":"INVEST_STREAMK_INTEGRATION" if passed else "NO_GO_ORACLE_CTA_TOPOLOGY",
      "best_shape":best["shape"] if best else None,"projected_recovery_us":recovery if best else None,
      "projection_note":"Normalized single-CTA scaling is a screening projection, not a full-route measurement."},
    "arms":arms}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True))
  return 0 if all(x["correctness"]["exact"] for x in arms) else 1


if __name__ == "__main__": raise SystemExit(main())
