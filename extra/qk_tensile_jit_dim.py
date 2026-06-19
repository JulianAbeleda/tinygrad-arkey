#!/usr/bin/env python3
"""A1 (research-measurement scope) — JIT-dim proof: one injected Tensile node runs under TinyJit/HCQGraph with the
correct launch dims.

Structural fact (CG-1/A1 experiments): tinygrad always reserves grid dim0 for local threads, so a custom_kernel can't
EMIT the Tensile workgroup grid (4,96,1)/(128,1,1). Scope-permitted fix (probe-local, no UOp surgery): monkeypatch the
compute queue's exec so that when the program is a TensileRunner, it uses the runner's own Tensile dims (not the host
trivial-kernel's). This fixes BOTH eager (HCQProgram.__call__) and JIT/HCQGraph (queue.exec line 175) paths.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_jit_dim.py
"""
from __future__ import annotations
import json, pathlib
from tinygrad import Tensor, Device, dtypes, UOp, TinyJit
from tinygrad.uop.ops import AxisType, KernelInfo
import tinygrad.engine.realize as R
import tinygrad.runtime.ops_amd as ops_amd
from extra.qk_tensile_runtime import TensileRunner
from extra.qk_tensile_hcq_launch import unbundle

T, IN, FF = 512, 4096, 12288
GM, GN, LSZ = 4, 96, 128

# --- probe-local patch: TensileRunner forces its own Tensile dims at the queue exec (covers the JIT/HCQGraph path) ---
_orig_exec = ops_amd.AMDComputeQueue.exec
def _patched_exec(self, prg, args_state, global_size, local_size):
  if isinstance(prg, TensileRunner): global_size, local_size = prg.tensile_global, prg.tensile_local
  return _orig_exec(self, prg, args_state, global_size, local_size)
ops_amd.AMDComputeQueue.exec = _patched_exec

def trivial_fxn(c:UOp, a:UOp, b:UOp) -> UOp:
  REST = (FF*T)//(GM*GN*LSZ)
  bm = UOp.range(GM, 0, AxisType.GLOBAL); bn = UOp.range(GN, 1, AxisType.GLOBAL)
  lane = UOp.range(LSZ, 2, AxisType.LOCAL); inner = UOp.range(REST, 3, AxisType.LOOP)
  cf = c.reshape(GM, GN, LSZ, REST); val = (a.reshape(-1)[0] + b.reshape(-1)[0]).cast(c.dtype)
  return cf[bm, bn, lane, inner].store(val).end(inner, lane, bn, bm).sink(arg=KernelInfo(opts_to_apply=()))

def main():
  assert Device.DEFAULT == "AMD"; dev = Device[Device.DEFAULT]
  cap = {json.loads(l)["role"]: json.loads(l) for l in open("bench/qk-tensile-extraction/kernarg_all.jsonl")}["ffn_gate_up"]
  elf = unbundle(); runner = TensileRunner(dev, "ffn_gate_up", cap, elf)

  keys=[]
  o=R.get_runtime
  def hook(d,a,cache=True):
    try:
      if a.op.name=="PROGRAM": keys.append(a.key)
    except: pass
    return o(d,a,cache)
  R.get_runtime=hook

  # warmup realize to codegen the trivial kernel + capture its program key, then swap runtime -> TensileRunner
  Tensor.manual_seed(1)
  A=Tensor.randn(IN,T,dtype=dtypes.half).contiguous().realize(); B=Tensor.randn(FF,IN,dtype=dtypes.half).contiguous().realize()
  C=Tensor.zeros(FF,T,dtype=dtypes.half).contiguous().realize(); dev.synchronize()
  C.custom_kernel(A,B,fxn=trivial_fxn)[0].realize(); dev.synchronize()
  key=keys[-1]
  R.runtime_cache[(key, Device.DEFAULT)] = runner

  # TinyJit a fn that runs the (now-Tensile) custom_kernel; call >=3x so cnt2+ replays via HCQGraph
  @TinyJit
  def jfn(A, B):
    C = Tensor.zeros(FF, T, dtype=dtypes.half).contiguous()
    return C.custom_kernel(A, B, fxn=trivial_fxn)[0].realize()

  rels=[]
  for i in range(4):
    Tensor.manual_seed(10+i)
    Ai=Tensor.randn(IN,T,dtype=dtypes.half).contiguous().realize(); Bi=Tensor.randn(FF,IN,dtype=dtypes.half).contiguous().realize()
    oracle=(Bi.float()@Ai.float()).realize(); dev.synchronize()
    out=jfn(Ai, Bi); dev.synchronize()
    rels.append(round(((out.float()-oracle).abs().max()/(oracle.float().abs().max()+1e-6)).item(), 6))
    print(f"  jit call {i} (cnt->{'replay' if i>=2 else 'ignore/capture'}): rel_err {rels[-1]}")
  R.get_runtime=o

  jit_replays_correct = all(r < 2e-2 for r in rels[2:])   # cnt>=2 = HCQGraph replays
  res=dict(schema="qk_tensile_jit_dim_v1", phase="A1", role="ffn_gate_up",
           approach="queue.exec dim-override for TensileRunner (probe-local, no UOp surgery)",
           tensile_dims={"global":list(runner.tensile_global),"local":list(runner.tensile_local)},
           rel_errs=rels, jit_captured=True, jit_replays_correct=jit_replays_correct,
           verdict="PASS" if jit_replays_correct else "KILL",
           note="if PASS: injected Tensile node is JIT/HCQGraph-capturable with correct dims via the queue.exec override.")
  pathlib.Path("bench/qk-tensile-extraction/jit_dim_proof.json").write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2)); print("\nA1 JIT-DIM VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
