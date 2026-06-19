#!/usr/bin/env python3
"""Diagnostic: localize WHY the isolated Tensile 66 TFLOPS doesn't transfer in-model (prefill).

Builds OFF (fp16-WMMA) and ON (+Tensile) prefill jits, warms both (past warmstart-search), then captures ONE warm
replay's per-kernel GPU times via ProfileRangeEvent (clean — not DEBUG=2 during warmstart search). Compares the
gateup/down/qo matmul times OFF vs ON, plus the transpose/zeros overhead the ON route adds.
- If tensile_* kernels are FASTER than the OFF matmuls but ON-total ≈ OFF-total -> the transpose/zeros overhead is
  the culprit (fixable: transpose-free route).
- If tensile_* ≈ OFF matmuls -> Tensile-in-model isn't faster (route dead; integration/grid/layout problem).

Run: DEV=AMD PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PMC=0 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_tensile_diag.py
"""
from __future__ import annotations
import os, collections
import tinygrad.llm.model as Mod
import tinygrad.codegen.opt.postrange as pr
from tinygrad import Tensor, TinyJit, Device
from tinygrad.device import Compiled

def capture_replay(j, chunk, temp, model, saved):
  pr._WARMSTART_OPTS = model._pf16_warmstart
  try:
    base = len(Compiled.profile_events)
    j(chunk, 0, temp).realize(); Device[Device.DEFAULT].synchronize(); Device[Device.DEFAULT]._at_profile_finalize()
  finally: pr._WARMSTART_OPTS = saved
  evs = Compiled.profile_events[base:]
  dur = collections.defaultdict(float); cnt = collections.defaultdict(int)
  for e in evs:
    if type(e).__name__ == "ProfileRangeEvent" and getattr(e, "en", None) is not None:
      nm = str(e.name).replace("\x1b[36m","").replace("\x1b[0m","")
      dur[nm] += float(e.en - e.st); cnt[nm] += 1
  return dur, cnt

def main():
  model_path = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  assert os.environ.get("PREFILL_V2")
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  import extra.qk_tensile_inmodel as TI
  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(model_path, 2048)
  N = PREFILL_UBATCH; TI.install(Device[Device.DEFAULT])
  t = Tensor([5,6,7,8,9,10]*200 + [0]*(2048-1200), dtype="int32").reshape(1, 2048)
  chunk = t[:, 0:N].contiguous(); temp = Tensor([0.0]); saved = pr._WARMSTART_OPTS
  def run_eager():   # EAGER (no TinyJit) so each kernel emits a ProfileRangeEvent
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try: model.forward(chunk, 0, temp).realize(); Device[Device.DEFAULT].synchronize()
    finally: pr._WARMSTART_OPTS = saved
  res = {}
  for flag, lbl in [(False, "OFF"), (True, "ON")]:
    Mod.PREFILL_TENSILE_GEMM = flag
    for b in model.blk: b._use_flash, b._prefill_v2 = False, True
    for _ in range(2): run_eager()                 # compile/warm (1st build incl warmstart-search)
    Device[Device.DEFAULT]._at_profile_finalize()
    base = len(Compiled.profile_events)
    run_eager(); Device[Device.DEFAULT]._at_profile_finalize()   # clean: cached programs, no recompile
    evs = Compiled.profile_events[base:]
    dur = collections.defaultdict(float); cnt = collections.defaultdict(int)
    for e in evs:
      if type(e).__name__ == "ProfileRangeEvent" and getattr(e, "en", None) is not None:
        nm = str(e.name).replace("\x1b[36m","").replace("\x1b[0m","")
        dur[nm] += float(e.en - e.st); cnt[nm] += 1
    res[lbl] = (dur, cnt)
    tot = sum(dur.values())
    print(f"\n===== {lbl}: total {tot/1000:.1f}us across {sum(cnt.values())} kernels =====")
    for nm, d in sorted(dur.items(), key=lambda x: -x[1])[:10]:
      print(f"  {nm[:40]:<42} {d/1000:8.1f}us  ({cnt[nm]}x, {d/cnt[nm]/1000:.1f}us/ea)  {100*d/tot:.1f}%")
  # focused: the routed roles
  print("\n===== routed-role time OFF vs ON =====")
  off, on = res["OFF"][0], res["ON"][0]
  print(f"  OFF total {sum(off.values())/1000:.1f}us   ON total {sum(on.values())/1000:.1f}us")
  for key in ("tensile_gateup", "tensile_down", "tensile_qo"):
    print(f"  ON {key}: {on.get(key,0)/1000:.1f}us")

if __name__ == "__main__": main()
