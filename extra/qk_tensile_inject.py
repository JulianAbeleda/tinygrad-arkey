#!/usr/bin/env python3
"""TPE-7c — inject the precompiled Tensile kernel as a tinygrad realize/JIT graph node WITHOUT UOp surgery:
1. build a trivial custom_kernel whose fxn has Tensile's exact launch grid (4,96,1)/(128,1,1) and references all 3
   buffers (so ProgramInfo carries the right dims + globals);
2. realize it once to codegen a valid Ops.PROGRAM + populate runtime_cache (capture its key via a get_runtime hook);
3. overwrite runtime_cache[key] = TensileRunner -> realize uses the precompiled Tensile kernel instead;
4. a fresh same-shaped custom_kernel hits the same key -> Tensile runs on the new buffers; verify vs fp16 oracle.

ffn_gate/up: C[FF,T]=W[FF,IN]@A[IN,T] (col-major C[m,n]=A[m,k]B[k,n], m=T=512). Research-only; no model route/default.
  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_inject.py
"""
from __future__ import annotations
import json, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
import tinygrad.engine.realize as R
from extra.qk_tensile_runtime import TensileRunner
from extra.qk_tensile_hcq_launch import unbundle

T, IN, FF = 512, 4096, 12288   # m=T, k=IN, n=FF
GM, GN, LSZ = 4, 96, 128       # Tensile grid: global (4,96,1), local (128,1,1)

def trivial_fxn(c:UOp, a:UOp, b:UOp) -> UOp:
  # kernel with Tensile's launch geometry; body is a placeholder (TensileRunner replaces execution). Must touch a,b,c.
  REST = (FF*T)//(GM*GN*LSZ)                                # 128
  bm = UOp.range(GM, 0, AxisType.GLOBAL)
  bn = UOp.range(GN, 1, AxisType.GLOBAL)
  lane = UOp.range(LSZ, 2, AxisType.LOCAL)
  inner = UOp.range(REST, 3, AxisType.LOOP)
  cf = c.reshape(GM, GN, LSZ, REST)                         # 4 x 96 x 128 x 128
  val = (a.reshape(-1)[0] + b.reshape(-1)[0]).cast(c.dtype)  # scalar; reference a,b so they stay inputs
  return cf[bm, bn, lane, inner].store(val).end(inner, lane, bn, bm).sink(arg=KernelInfo(opts_to_apply=()))

def main():
  assert Device.DEFAULT == "AMD"; dev = Device[Device.DEFAULT]
  cap = {json.loads(l)["role"]: json.loads(l) for l in open("bench/qk-tensile-extraction/kernarg_all.jsonl")}["ffn_gate_up"]
  elf = unbundle(); runner = TensileRunner(dev, "ffn_gate_up", cap, elf)

  captured_keys=[]
  orig=R.get_runtime
  def hook(device, ast, cache=True):
    captured_keys.append((ast.key, device, ast.arg.global_size, ast.arg.local_size))
    return orig(device, ast, cache)
  R.get_runtime=hook
  try:
    # 1+2: trivial kernel realize -> codegen PROGRAM + cache + capture key
    Tensor.manual_seed(1)
    A1=Tensor.randn(IN,T,dtype=dtypes.half).contiguous().realize(); B1=Tensor.randn(FF,IN,dtype=dtypes.half).contiguous().realize()
    C1=Tensor.zeros(FF,T,dtype=dtypes.half).contiguous().realize(); dev.synchronize()
    out1=C1.custom_kernel(A1,B1,fxn=trivial_fxn)[0]; out1.realize(); dev.synchronize()
    # find the trivial kernel's program key (dims (4,96,1)/(128,1,1))
    keymatch=[k for k,d,gs,ls in captured_keys if tuple(gs)==(GM,GN,1) or (len(gs)>=2 and gs[0]==GM and gs[1]==GN)]
    print("captured program dims:", [(tuple(gs),tuple(ls)) for k,d,gs,ls in captured_keys][-5:])
    if not keymatch: print("NO key matched Tensile dims -> codegen produced different dims");
    key=keymatch[-1] if keymatch else captured_keys[-1][0]
    # 3: swap the runtime for that program
    R.runtime_cache[(key, Device.DEFAULT)] = runner
    # 4: fresh same-shape custom_kernel -> same key -> TensileRunner runs
    Tensor.manual_seed(2)
    A2=Tensor.randn(IN,T,dtype=dtypes.half).contiguous().realize(); B2=Tensor.randn(FF,IN,dtype=dtypes.half).contiguous().realize()
    C2=Tensor.zeros(FF,T,dtype=dtypes.half).contiguous().realize()
    oracle=(B2.float()@A2.float()).realize(); dev.synchronize()
    out2=C2.custom_kernel(A2,B2,fxn=trivial_fxn)[0]; out2.realize(); dev.synchronize()
    rel=((C2.float()-oracle).abs().max()/(oracle.float().abs().max()+1e-6)).item()
  finally:
    R.get_runtime=orig
  res=dict(schema="qk_tensile_inject_v1", phase="TPE-7c", approach="custom_kernel(trivial,tensile-dims)+runtime_cache swap",
           injected_correct=rel<2e-2, rel_err=round(rel,6),
           verdict="PASS" if rel<2e-2 else "PARTIAL/KILL",
           note="if PASS: the precompiled Tensile kernel runs through tinygrad's realize path driven by TensileRunner "
                "(in-graph node), no UOp surgery. Next: confirm under TinyJit + one-block harness.")
  pathlib.Path("bench/qk-tensile-extraction/inject.json").write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2)); print("\nTPE-7c INJECT VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
