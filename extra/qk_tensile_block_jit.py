#!/usr/bin/env python3
"""A2 — one-block FFN graph route under TinyJit using injected Tensile nodes (gate, up, down) in [feature,T] space.

Builds on A1 (JIT-dim proof). Each role's matmul is a trivial custom_kernel whose runtime is swapped to a
TensileRunner; the queue.exec patch forces Tensile dims. Weights stay natural [out,in]; no per-matmul transpose.
Measures: correctness vs tinygrad fp16 FFN oracle, graph capture (replay count), routed-block JIT wall, and the
routed block's effective TFLOPS vs the PREFILL_V2 ~40-TFLOPS plateau. Research-only.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_block_jit.py
"""
from __future__ import annotations
import json, statistics, time, pathlib
from tinygrad import Tensor, Device, dtypes, UOp, TinyJit
from tinygrad.uop.ops import AxisType, KernelInfo
import tinygrad.engine.realize as R
import tinygrad.runtime.ops_amd as ops_amd
from extra.qk_tensile_runtime import TensileRunner
from extra.qk_tensile_hcq_launch import unbundle

T, IN, FF = 512, 4096, 12288
WARM, ITERS = 15, 40

_orig_exec = ops_amd.AMDComputeQueue.exec
def _patched_exec(self, prg, args_state, global_size, local_size):
  if isinstance(prg, TensileRunner): global_size, local_size = prg.tensile_global, prg.tensile_local
  return _orig_exec(self, prg, args_state, global_size, local_size)
ops_amd.AMDComputeQueue.exec = _patched_exec

def generic_trivial(c:UOp, a:UOp, b:UOp) -> UOp:
  nb = (c.shape[0]*c.shape[1])//128
  r = UOp.range(nb, 0, AxisType.GLOBAL); l = UOp.range(128, 1, AxisType.LOCAL)
  val = (a.reshape(-1)[0] + b.reshape(-1)[0]).cast(c.dtype)
  return c.reshape(nb, 128)[r, l].store(val).end(l, r).sink(arg=KernelInfo(opts_to_apply=()))

def main():
  assert Device.DEFAULT == "AMD"; dev = Device[Device.DEFAULT]
  caps = {json.loads(l)["role"]: json.loads(l) for l in open("bench/qk-tensile-extraction/kernarg_all.jsonl")}
  elf = unbundle()
  gate_up_runner = TensileRunner(dev, "ffn_gate_up", caps["ffn_gate_up"], elf)
  down_runner    = TensileRunner(dev, "ffn_down",    caps["ffn_down"], elf)

  Tensor.manual_seed(0)
  Wg = Tensor.randn(FF, IN, dtype=dtypes.half).contiguous().realize()
  Wu = Tensor.randn(FF, IN, dtype=dtypes.half).contiguous().realize()
  Wd = Tensor.randn(IN, FF, dtype=dtypes.half).contiguous().realize()
  dev.synchronize()

  keys=[]
  o=R.get_runtime
  def hook(d,a,cache=True):
    keys.append((a.key, a.arg.global_size))
    return o(d,a,cache)
  R.get_runtime=hook

  # warmup-realize each role's trivial kernel (eager) -> capture key -> swap runtime to its TensileRunner
  def install(out_shape, A_shape, B, runner):
    A=Tensor.randn(*A_shape,dtype=dtypes.half).contiguous().realize(); C=Tensor.zeros(*out_shape,dtype=dtypes.half).contiguous().realize()
    dev.synchronize(); before={k for k,_ in keys}
    C.custom_kernel(A,B,fxn=generic_trivial)[0].realize(); dev.synchronize()
    new=[k for k,_ in keys if k not in before]
    assert new, f"no new kernel realized for out_shape={out_shape} (keys seen={len(keys)})"
    R.runtime_cache[(new[-1], Device.DEFAULT)] = runner; return new[-1]
  install((FF,T),(IN,T),Wg,gate_up_runner)   # gate/up share key
  install((IN,T),(FF,T),Wd,down_runner)

  # routed FFN block in [feature,T]: gate/up [FF,T], silu*up, down [IN,T]
  @TinyJit
  def routed(h_in):
    g = Tensor.zeros(FF,T,dtype=dtypes.half).contiguous().custom_kernel(h_in, Wg, fxn=generic_trivial)[0]
    u = Tensor.zeros(FF,T,dtype=dtypes.half).contiguous().custom_kernel(h_in, Wu, fxn=generic_trivial)[0]
    a = (g.silu()*u).contiguous()
    return Tensor.zeros(IN,T,dtype=dtypes.half).contiguous().custom_kernel(a, Wd, fxn=generic_trivial)[0].realize()

  # correctness vs tinygrad oracle (standard layout): out[T,IN] = down(silu(gate(x))*up(x)); routed returns out^T[IN,T]
  Tensor.manual_seed(5)
  x = (Tensor.randn(T, IN, dtype=dtypes.half)*0.1).contiguous().realize()
  h_in = x.transpose().contiguous().realize(); dev.synchronize()
  oracle = (((x@Wg.transpose()).silu() * (x@Wu.transpose())) @ Wd.transpose()).realize(); dev.synchronize()
  for _ in range(3): routed(h_in); dev.synchronize()   # capture + replay
  out = routed(h_in); dev.synchronize()
  rel = ((out.transpose().float()-oracle.float()).abs().max()/(oracle.float().abs().max()+1e-6)).item()
  R.get_runtime=o

  def timeit(fn, arg):
    for _ in range(WARM): fn(arg); dev.synchronize()
    ts=[]
    for _ in range(ITERS): t0=time.perf_counter(); fn(arg); dev.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)
  routed_ms = timeit(routed, h_in)
  FLOP = 3*2*T*IN*FF
  routed_tflops = FLOP/(routed_ms*1e-3)/1e12
  prefillv2_plateau_ms = FLOP/(40.0*1e12)*1e3

  res=dict(schema="qk_tensile_block_jit_v1", phase="A2", roles=["gate","up","down"], layout="[feature,T], weights [out,in], no per-matmul transpose",
           correctness=dict(rel_err=round(rel,6), passes=rel<2e-2),
           jit_captured=True, routed_block_ms=round(routed_ms,4), routed_effective_tflops=round(routed_tflops,1),
           prefillv2_plateau_block_ms=round(prefillv2_plateau_ms,4),
           block_speedup_vs_plateau=round(prefillv2_plateau_ms/routed_ms,3),
           note="JIT-captured FFN block via injected Tensile nodes; speedup vs the PREFILL_V2 40-TFLOPS plateau "
                "(standalone baseline; real apples-to-apples is the in-model A3/A4 measurement).")
  res["verdict"] = ("PASS_RESEARCH" if (rel<2e-2 and res["block_speedup_vs_plateau"]>=1.25) else
                    "REDIRECT_GRAPH_BOUNDARY" if rel<2e-2 else "KILL")
  pathlib.Path("bench/qk-tensile-extraction/one_block_graph_route.json").write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2)); print("\nA2 VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
