#!/usr/bin/env python3
"""Exact R9 qualifier for the broad llama-shaped Q6 CTA package."""
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
from extra.llm_research.prefill.bench_nv_q6_oracle_cta_sweep import _fixture
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import COLS,K,ROWS,SHARED_BYTES,q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

CURRENT_MAIN_US, LLAMA_MAIN_US, REQUIRED_RECOVERY_US = 318.8, 201.216, 23.5
WORK_UNITS, OWNERS = 128*48, 170
LAUNCH_SHARED_BYTES = SHARED_BYTES+1024


def _stats(xs): return {"samples_us":xs,"min_us":min(xs),"median_us":statistics.median(xs),"max_us":max(xs)}


def _record(q:np.ndarray, scales:np.ndarray) -> np.ndarray:
  out=np.zeros((2,COLS,36),np.uint32)
  out[:,:,:4]=np.ascontiguousarray(scales.reshape(8,COLS).reshape(2,4,COLS).transpose(0,2,1)).view(np.uint32)
  out.view(np.uint8).reshape(2,COLS,144)[:,:,16:144]=q.reshape(2,128,COLS).transpose(0,2,1).view(np.uint8)
  return out


def _arm(prefetch:bool, rounds:int, replicas:int, root:pathlib.Path) -> dict[str,object]:
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_oracle_broad_cta_kernel(ph(replicas*ROWS*COLS,dtypes.float32,0),ph(ROWS*105,dtypes.uint16,1),
    ph(2*COLS*36,dtypes.uint32,2),replicas=replicas,prefetch_second_panel=prefetch)
  render_start=time.perf_counter(); program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE); render_ms=(time.perf_counter()-render_start)*1e3
  name=f"nv_q6_oracle_broad_cta_{'prefetch' if prefetch else 'serial'}"
  arm_dir=root/name; arm_dir.mkdir(parents=True,exist_ok=True); source_path=arm_dir/f"{name}.cu"; source_path.write_text(source)
  compile_start=time.perf_counter(); binary=Device["NV"].compiler.compile(source); compile_ms=(time.perf_counter()-compile_start)*1e3
  cubin_path=arm_dir/f"{name}.cubin"; cubin_path.write_bytes(binary); census=analyze_cubin(cubin_path,arm_dir/"sass",name)["summary"]

  blocks,q,scales,reference=_fixture(1,COLS); record=_record(q,scales)
  dev=Device["NV"]; arrays=(np.full(replicas*ROWS*COLS,np.nan,np.float32),blocks.view(np.uint16).reshape(-1),record.reshape(-1))
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in arrays]
  for buf,array in zip(bufs,arrays): dev.allocator._copyin(buf,memoryview(array.tobytes()))
  runner=NVProgram(dev,name,binary,shared_mem=LAUNCH_SHARED_BYTES)
  runner(*bufs,global_size=(replicas,1,1),local_size=(256,1,1),wait=True,timeout=120000)
  raw=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(raw,bufs[0])
  got=np.frombuffer(raw,np.float32,count=replicas*ROWS*COLS).reshape(replicas,ROWS,COLS)
  timing=_stats([runner(*bufs,global_size=(replicas,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6 for _ in range(rounds)])
  resources=census["resources"] or {}; median=float(timing["median_us"])
  # Each full owner executes 6144/170 normalized K256 units. This is a
  # screening projection only; a pass must still survive the full route.
  launch_floor_us=4.0; per_unit=max(0.0,median-launch_floor_us)
  projected=launch_floor_us+per_unit*(WORK_UNITS/OWNERS)
  exact=bool(np.array_equal(got,np.broadcast_to(reference,got.shape)))
  feasible=bool(exact and resources.get("registers",256)<=255 and resources.get("stack_bytes",1)==0 and
    resources.get("local_static_bytes",1)==0 and LAUNCH_SHARED_BYTES<=58_880)
  return {"prefetch_second_panel":prefetch,"shape":{"rows":ROWS,"cols":COLS,"k":K,"replicas":replicas,
      "block":[256,1,1],"payload_shared_bytes":SHARED_BYTES,"launch_shared_bytes":LAUNCH_SHARED_BYTES},
    "correctness":{"exact":exact,"finite":bool(np.isfinite(got).all()),"max_abs":float(np.max(np.abs(got-reference)))},
    "timing":timing,"projection":{"launch_floor_us":launch_floor_us,"owner_work_units":WORK_UNITS/OWNERS,
      "projected_full_main_us":projected,"projected_recovery_us":CURRENT_MAIN_US-projected,"screening_only":True},
    "compiler":{"render_ms":render_ms,"compile_wall_ms":compile_ms,"source":str(source_path),"source_bytes":len(source),
      "cubin":str(cubin_path),"cubin_sha256":hashlib.sha256(binary).hexdigest()},
    "sass":{"instruction_total":census["instruction_total"],"families":census["families"],"resources":resources,
      "spill_regions":census["spill_regions"]},"feasible":feasible}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--rounds",type=int,default=9); ap.add_argument("--replicas",type=int,default=1)
  ap.add_argument("--out",type=pathlib.Path,required=True); ap.add_argument("--artifacts",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.rounds < 9: raise ValueError("qualification requires R9 or greater")
  arms=[_arm(False,args.rounds,args.replicas,args.artifacts),_arm(True,args.rounds,args.replicas,args.artifacts)]
  candidate=arms[1]; recovery=float(candidate["projection"]["projected_recovery_us"])
  passed=bool(candidate["feasible"] and recovery>=REQUIRED_RECOVERY_US)
  result={"schema":"tinygrad.nv_q6_oracle_broad_cta.v1","baselines":{"current_main_us":CURRENT_MAIN_US,
      "llama_main_us":LLAMA_MAIN_US,"llama_5pct_gate_us":LLAMA_MAIN_US*1.05},
    "gate":{"required_projected_recovery_us":REQUIRED_RECOVERY_US,"passed":passed,
      "decision":"INVEST_FULL_ROUTE" if passed else "NO_GO_BROAD_CTA"},"arms":arms}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,sort_keys=True))
  return 0 if all(x["correctness"]["exact"] for x in arms) else 1


if __name__ == "__main__": raise SystemExit(main())
