#!/usr/bin/env python3
"""Execute the BoltBeam KV/RoPE cache-store artifact across its symbolic ABI surface."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate
from tinygrad.uop.ops import UOp

HKV,HD,MAXC=8,128,19

def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")

def _run_case(cache_dtype,vparts:int,pos:int,seed:int,reps:int) -> dict:
  rng=np.random.default_rng(seed)
  np_dtype=np.float16 if cache_dtype is dtypes.float16 else np.float32
  cache_np=rng.uniform(-3,3,(2,1,HKV,MAXC,HD)).astype(np_dtype)
  initial=cache_np.copy();k_np=rng.normal(0,.4,(HKV,HD)).astype(np.float32)
  v_np=rng.normal(0,.4,(HKV*HD,vparts)).astype(np.float32) if vparts > 1 else rng.normal(0,.4,HKV*HD).astype(np.float32)
  angles=np.arange(MAXC*(HD//2),dtype=np.float32).reshape(MAXC,HD//2)*np.float32(.00091)
  freqs_np=np.concatenate((np.cos(angles),np.sin(angles)),axis=1).astype(np.float32)
  cache=Tensor(cache_np,dtype=cache_dtype,device="NV").realize();k=Tensor(k_np.reshape(-1),device="NV").realize()
  v=Tensor(v_np,device="NV").realize();freqs=Tensor(freqs_np,device="NV").realize()
  candidate={"family":"kv_rope_store.v1","Hkv":HKV,"Hd":HD,"max_context":MAXC,"vparts":vparts}
  emitter,ticket=lower_authorized_candidate(candidate,(("decode_kv_store_fusion","kv_rope_store"),))
  ast=emitter(UOp.placeholder((2,1,HKV,MAXC,HD),cache_dtype,0),UOp.placeholder((HKV*HD,),dtypes.float32,1),
    UOp.placeholder((HKV*HD,vparts),dtypes.float32,2) if vparts > 1 else UOp.placeholder((HKV*HD,),dtypes.float32,2),
    UOp.placeholder((MAXC,HD),dtypes.float32,3))
  program=to_program(ast,Device["NV"].renderer);runtime=get_runtime("NV",program)
  assert tuple(var.arg[0] for var in program.arg.vars) == ("start_pos",)
  global_size,local_size=program.arg.launch_dims({});buffers=[_buf(t) for t in (cache,k,v,freqs)]
  def run(wait=False): return runtime(*[buffers[i] for i in program.arg.globals],global_size=global_size,
    local_size=local_size,vals=(pos,),wait=wait)
  run(True);timing=[run(True)*1e6 for _ in range(reps)];got=cache.numpy()
  half=HD//2;k1=k_np[:,:half];k2=k_np[:,half:];cos=freqs_np[pos,:half];sin=freqs_np[pos,half:]
  expected_k=np.concatenate((k1*cos-k2*sin,k2*cos+k1*sin),axis=1).astype(np_dtype)
  if vparts == 1: expected_v=v_np.reshape(HKV,HD).astype(np_dtype)
  else:
    summed=v_np[:,0].copy()
    for part in range(1,vparts): summed=np.add(summed,v_np[:,part],dtype=np.float32)
    expected_v=summed.reshape(HKV,HD).astype(np_dtype)
  slot=np.stack((expected_k,expected_v));got_slot=got[:,0,:,pos,:]
  outside=got.copy();outside[:,0,:,pos,:]=initial[:,0,:,pos,:]
  err=np.abs(got_slot.astype(np.float32)-slot.astype(np.float32));atol=2e-3 if cache_dtype is dtypes.float16 else 5e-6
  return {"cache_dtype":cache_dtype.name,"vparts":vparts,"start_pos":pos,
    "tickets":[row.to_dict() for row in ticket.tickets],
    "program_key":program.key.hex() if isinstance(program.key,bytes) else str(program.key),
    "payload_sha256":hashlib.sha256(initial.tobytes()+k_np.tobytes()+v_np.tobytes()+freqs_np.tobytes()).hexdigest(),
    "correctness":{"pass":bool(np.isfinite(got_slot).all() and err.max() <= atol and np.array_equal(outside,initial)),
      "max_abs":float(err.max()),"atol":atol,"outside_cache_bit_exact":bool(np.array_equal(outside,initial))},
    "timing_us":{"median":statistics.median(timing),"samples":timing}}

def main() -> int:
  ap=argparse.ArgumentParser();ap.add_argument("--out",type=pathlib.Path,required=True);ap.add_argument("--reps",type=int,default=11);args=ap.parse_args()
  if not str(Device.DEFAULT).startswith("NV"): raise RuntimeError(f"native NV required, got {Device.DEFAULT}")
  rows=[_run_case(dtype,vparts,pos,202609020+di*100+vparts*10+pos,args.reps)
    for di,dtype in enumerate((dtypes.float16,dtypes.float32)) for vparts in (1,4) for pos in (0,7,MAXC-1)]
  report={"schema":"tinygrad.boltbeam_kv_store_qualification.v1","device":str(Device.DEFAULT),
    "git_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
    "cases":rows,"pass":all(row["correctness"]["pass"] for row in rows)}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"pass":report["pass"],"cases":[{"dtype":r["cache_dtype"],"vparts":r["vparts"],
    "pos":r["start_pos"],"max_abs":r["correctness"]["max_abs"],"outside_exact":r["correctness"]["outside_cache_bit_exact"],
    "median_us":r["timing_us"]["median"]} for r in rows]},indent=2))
  return 0 if report["pass"] else 1

if __name__ == "__main__": raise SystemExit(main())
