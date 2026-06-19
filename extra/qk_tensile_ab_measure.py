#!/usr/bin/env python3
"""Clean clock-controlled A/B: PREFILL_V2 fp16-WMMA (Tensile OFF) vs +Tensile (ON), all roles.

Two separate TinyJits of forward (one traced flag-off, one flag-on) under the warmstart-OPTS context, measured
INTERLEAVED (round-robin) so clock drift hits both equally -> the trustworthy warm pp512 speedup with qo+gateup+
down all routing. (Subprocess A/B is clock-unreliable: the same ON run varied 3433->7321 tok/s across processes.)

Run: DEV=AMD PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PYTHONPATH=. .venv/bin/python extra/qk_tensile_ab_measure.py
"""
from __future__ import annotations
import os, sys, time, statistics
import tinygrad.llm.model as Mod
import tinygrad.codegen.opt.postrange as pr
from tinygrad import Tensor, TinyJit, Device, GlobalCounters

def main():
  model_path = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  assert os.environ.get("PREFILL_V2"), "run with PREFILL_V2=1"
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  import extra.qk_tensile_inmodel as TI
  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(model_path, 2048)
  N = PREFILL_UBATCH
  TI.install(Device[Device.DEFAULT])                      # eager Tensile install (needed for the ON route)
  t = Tensor([5,6,7,8,9,10]*200 + [0]*(2048-1200), dtype="int32").reshape(1, 2048)
  chunk = t[:, 0:N].contiguous(); temp = Tensor([0.0])
  saved = pr._WARMSTART_OPTS

  def build(flag):
    Mod.PREFILL_TENSILE_GEMM = flag
    for b in model.blk: b._use_flash, b._prefill_v2 = False, True
    return TinyJit(model.forward)

  def run_once(j):
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try: j(chunk, 0, temp).realize(); Device[Device.DEFAULT].synchronize()
    finally: pr._WARMSTART_OPTS = saved

  # TinyJit traces on FIRST CALL -> set the flag, build, and trace each jit before changing the flag.
  TI.ROUTE_COUNT.clear()
  joff = build(False); run_once(joff)                    # traces with flag OFF
  rc_off = dict(TI.ROUTE_COUNT); TI.ROUTE_COUNT.clear()
  jon = build(True); run_once(jon)                       # traces with flag ON
  rc_on = dict(TI.ROUTE_COUNT)
  assert not rc_off, f"OFF jit unexpectedly routed: {rc_off}"
  for _ in range(8): run_once(joff); run_once(jon)
  # interleaved measurement (round-robin) for clock fairness
  toff, ton = [], []
  for _ in range(25):
    for j, acc in ((joff, toff), (jon, ton)):
      pr._WARMSTART_OPTS = model._pf16_warmstart
      try:
        GlobalCounters.reset(); t0 = time.perf_counter(); j(chunk, 0, temp).realize()
        Device[Device.DEFAULT].synchronize(); acc.append(time.perf_counter() - t0)
      finally: pr._WARMSTART_OPTS = saved
  def stats(ts): return N/statistics.median(ts), N/min(ts)
  off_med, off_best = stats(toff); on_med, on_best = stats(ton)
  print(f"OFF (PREFILL_V2 fp16-WMMA):  median {off_med:6.0f} tok/s  best {off_best:6.0f}   route={rc_off}")
  print(f"ON  (+Tensile all roles):    median {on_med:6.0f} tok/s  best {on_best:6.0f}   route={rc_on}")
  print(f"SPEEDUP  median {on_med/off_med:.3f}x  best {on_best/off_best:.3f}x   (strong gate = 1.35x)")

if __name__ == "__main__": main()
