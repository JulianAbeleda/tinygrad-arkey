#!/usr/bin/env python3
"""Bit-exact isolated geometry sweep for the promoted 1x4096 native RMSNorm body."""
from __future__ import annotations

import argparse, json, pathlib, sqlite3, statistics, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program

WARPS = (1, 2, 4, 8, 16)

def _program(warps:int) -> KernelProgram:
  spec = DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=warps,
    x_dtype=dtypes.float16, weight_dtype=dtypes.float16, out_dtype=dtypes.float16, x_rank=1, native=True)
  return KernelProgram("research.nv_rmsnorm_native_geometry", f"rmsnorm_native_1_4096_w{warps}",
    KernelProgramProvenance.RESEARCH_ONLY, emit_decode_rmsnorm_kernel(spec),
    output_spec=OutputSpec((4096,), dtypes.float16))

def measure(out:pathlib.Path, replays:int, warmup:int, reps:int) -> None:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"DEV=NV required, got {dev}")
  rng = np.random.default_rng(20260826)
  x = Tensor(rng.normal(0, .2, 4096).astype(np.float16), device=dev).contiguous().realize()
  w = Tensor(rng.normal(1, .05, 4096).astype(np.float16), device=dev).contiguous().realize()
  rows, ref = [], None
  for warps in WARPS:
    program, dst = _program(warps), Tensor.empty(4096, dtype=dtypes.float16, device=dev)
    @TinyJit
    def run(a:Tensor, weight:Tensor): return execute_research_program(dst, a, weight, program=program)
    run(x,w).realize(); got=run(x,w).realize()
    Device[dev].synchronize(); arr = np.asarray(got.numpy())
    if ref is None: ref = arr.copy()
    exact = bool(np.array_equal(arr, ref))
    for _ in range(warmup): run(x,w).realize()
    Device[dev].synchronize()
    samples=[]
    for _ in range(reps):
      Device[dev].synchronize(); start=time.perf_counter_ns()
      for _ in range(replays): run(x,w).realize()
      Device[dev].synchronize(); samples.append((time.perf_counter_ns()-start)/1000/replays)
    rows.append({"warps":warps, "kernel":program.program_id, "bit_exact_to_w1":exact,
      "samples_us":samples, "median_us":statistics.median(samples)})
  out.write_text(json.dumps({"schema":"tinygrad.nv_rmsnorm_native_geometry.v1", "replays":replays,
    "warmup":warmup, "reps":reps, "control_warps":16,
    "verdict_order":[r["warps"] for r in sorted(rows,key=lambda r:r["median_us"])], "rows":rows}, indent=2, sort_keys=True)+"\n")

def parse(meta:pathlib.Path, trace:pathlib.Path, out:pathlib.Path) -> None:
  data=json.loads(meta.read_text()); con=sqlite3.connect(trace)
  names={int(i):str(v) for i,v in con.execute("select id,value from StringIds")}
  vals={r["kernel"]:[] for r in data["rows"]}
  for start,end,short in con.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    name=names.get(int(short),"")
    if name in vals: vals[name].append((end-start)/1000.0)
  for row in data["rows"]:
    samples=sorted(vals[row["kernel"]]); row["instances"]=len(samples)
    row["median_us"]=float(np.median(samples)) if samples else None
  control=next(r["median_us"] for r in data["rows"] if r["warps"]==16)
  for row in data["rows"]: row["delta_vs_w16_us"]=None if row["median_us"] is None else row["median_us"]-control
  data["control_warps"]=16; data["verdict_order"]=[r["warps"] for r in sorted(data["rows"], key=lambda r:r["median_us"])]
  out.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True)
  m=sub.add_parser("measure"); m.add_argument("--out",type=pathlib.Path,required=True);m.add_argument("--replays",type=int,default=1000);m.add_argument("--warmup",type=int,default=50);m.add_argument("--reps",type=int,default=7)
  p=sub.add_parser("parse");p.add_argument("--meta",type=pathlib.Path,required=True);p.add_argument("--trace",type=pathlib.Path,required=True);p.add_argument("--out",type=pathlib.Path,required=True)
  a=ap.parse_args(); measure(a.out,a.replays,a.warmup,a.reps) if a.mode=="measure" else parse(a.meta,a.trace,a.out)
