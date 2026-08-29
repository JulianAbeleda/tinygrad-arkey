#!/usr/bin/env python3
"""One-variable Ridx0 unroll gate for the compiler-owned pp512 Q4 body."""
from __future__ import annotations

import argparse, json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

LOOP = "  for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {"


def _resources(cubin:pathlib.Path) -> dict[str, int]:
  cp = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-resource-usage", str(cubin)],
                      capture_output=True, text=True, check=True)
  text = cp.stdout+cp.stderr
  def one(pattern:str) -> int:
    m = re.search(pattern, text); return int(m.group(1)) if m else -1
  return {"registers":one(r"REG:(\d+)"), "stack":one(r"STACK:(\d+)"),
          "shared":one(r"SHARED:(\d+)"), "local":one(r"LOCAL:(\d+)")}


def main() -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--source",type=pathlib.Path,default=pathlib.Path(
    "docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/production_gate_guarded_k64.cu"))
  ap.add_argument("--out-dir",type=pathlib.Path,required=True)
  ap.add_argument("--unrolls",default="1,2,4,8")
  ap.add_argument("--warmup",type=int,default=10);ap.add_argument("--reps",type=int,default=9)
  args=ap.parse_args(); base=args.source.read_text()
  if base.count(LOOP)!=1: raise RuntimeError(f"expected one exact Ridx0 loop, found {base.count(LOOP)}")
  args.out_dir.mkdir(parents=True,exist_ok=True); rows=[]
  for factor in [int(x) for x in args.unrolls.split(",")]:
    if factor not in (1,2,4,8,16,32,64): raise RuntimeError(f"illegal factor {factor}")
    src=base if factor==1 else base.replace(LOOP,f"  #pragma unroll {factor}\n{LOOP}")
    sp=args.out_dir/f"gateup-k64-unroll{factor}.cu"; bp=sp.with_suffix(".cubin"); jp=sp.with_suffix(".json")
    sp.write_text(src); bp.write_bytes(NVRTCCompiler("sm_120a",ptx=False,cache_key=f"prefill_gateup_k64_unroll{factor}_v1").compile(src))
    cp=subprocess.run([sys.executable,str(ROOT/"extra/llm_research/prefill/nv_prefill_gateup_ncu_bridge.py"),
      "--source",str(sp),"--cubin",str(bp),"--warmup",str(args.warmup),"--reps",str(args.reps),"--out",str(jp)],
      cwd=ROOT,capture_output=True,text=True)
    if cp.returncode: raise RuntimeError(f"unroll {factor} launch failed: {cp.stdout[-1000:]}\n{cp.stderr[-2000:]}")
    rec=json.loads(jp.read_text());rows.append({"factor":factor,"resources":_resources(bp),"launch":rec})
    print(json.dumps(rows[-1],sort_keys=True),flush=True)
  hashes={r["launch"]["output_sha256"] for r in rows}; base_us=rows[0]["launch"]["median_us"]
  result={"schema":"tinygrad.nv_prefill_gateup_unroll_discriminator.v1","rows":rows,
    "all_outputs_bit_exact":len(hashes)==1,"winner":min(rows,key=lambda r:r["launch"]["median_us"])["factor"]}
  for r in rows:r["median_ratio_vs_base"]=r["launch"]["median_us"]/base_us
  result["passed"]=bool(result["all_outputs_bit_exact"] and all(r["launch"]["passed"] for r in rows))
  out=args.out_dir/"result.json";out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
  return 0 if result["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
