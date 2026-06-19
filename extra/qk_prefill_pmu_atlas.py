#!/usr/bin/env python3
"""In-model PREFILL primitive PMU atlas — learn whether prefill is compute- or bandwidth-bound (measured).

Counterpart to qk_primitive_pmu_atlas.py (decode). Runs the real Qwen3-8B PREFILL forward eagerly on a
512-token chunk with _prefill_v2=True (fp16 TC/WMMA GEMMs) under native PMC, classifies each primitive from
hardware counters. Prefill reuses each weight across all 512 tokens (arithmetic intensity ~512x decode) -> if
the regime difference is real, prefill matmuls should be compute(WMMA)-bound with high L2 reuse, unlike the
bandwidth-bound decode GEMVs.

Run:  DEV=AMD PREFILL_V2=1 PMC=1 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_pmu_atlas.py
"""
from __future__ import annotations
import os, json, pathlib, collections
import numpy as np
from extra.qk_primitive_pmu_atlas import stats_of

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  from tinygrad import Tensor, Device, UOp
  from tinygrad.device import Compiled
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  Tensor.manual_seed(0)
  m, _ = Transformer.from_gguf(pathlib.Path(model).expanduser(), 2048)
  N = PREFILL_UBATCH
  for b in m.blk: b._use_flash, b._prefill_v2 = False, True
  temp = Tensor([0.0])
  toks = Tensor([[5,6,7,8,9,10]*200][0][:N], dtype="int32").reshape(1, N)
  # warm (compile + clock; counters are clock-independent so even cold is fine) then capture one eager forward
  for _ in range(2): m.forward(toks, 0, temp).realize(); Device['AMD'].synchronize()
  base = len(Compiled.profile_events)
  m.forward(toks, 0, temp).realize()
  Device['AMD'].synchronize(); Device['AMD']._at_profile_finalize()
  evs = Compiled.profile_events[base:]
  names = {e.tag: e.name for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent" and getattr(e,"tag",None) is not None}
  agg = collections.defaultdict(lambda: {"n":0, "active":0, "valu":[], "l2hit":[]})
  for ev in evs:
    if type(ev).__name__ != "ProfilePMCEvent": continue
    nm = names.get(ev.kern, f"kern{ev.kern}"); s = stats_of(ev)
    active = s.get("GRBM_GUI_ACTIVE",(0,0,1))[1]; a = agg[nm]; a["n"]+=1; a["active"]+=active
    if active>0:
      vsum,_,vcnt = s.get("SQ_INSTS_VALU",(0,0,1)); a["valu"].append(100*(vsum/max(vcnt,1))/(active*4))
    hit,miss = s.get("GL2C_HIT",(0,0,1))[0], s.get("GL2C_MISS",(0,0,1))[0]
    if hit+miss>0: a["l2hit"].append(100*hit/(hit+miss))
  rows=[]
  for nm,a in agg.items():
    vu=float(np.mean(a["valu"])) if a["valu"] else 0.0; l2=float(np.mean(a["l2hit"])) if a["l2hit"] else -1.0
    cls=("ALU/WMMA-bound" if vu>=40 else "bandwidth-bound" if 0<=l2<30 else "cache/compute-bound" if l2>=30 else "?")
    rows.append({"primitive":nm.replace("\x1b[36m","").replace("\x1b[0m",""),"launches":a["n"],"active":a["active"],
                 "VALU_util%":round(vu,1),"L2_hit%":round(l2,1),"class":cls})
  rows.sort(key=lambda r:-r["active"]); tot=sum(r["active"] for r in rows) or 1
  for r in rows: r["gpu%"]=round(100*r["active"]/tot,1)
  print(f"{'primitive':<32}{'launch':>7}{'gpu%':>6}{'VALU%':>7}{'L2hit%':>8}  class")
  for r in rows[:24]: print(f"{r['primitive'][:31]:<32}{r['launches']:>7}{r['gpu%']:>6}{r['VALU_util%']:>7}{r['L2_hit%']:>8}  {r['class']}")
  out=pathlib.Path("bench/qk-primitive-pmu-atlas"); out.mkdir(parents=True,exist_ok=True)
  (out/"prefill_result.json").write_text(json.dumps({"model":pathlib.Path(model).stem,"N":N,"rows":rows},indent=1))
  print(f"\nwrote {out/'prefill_result.json'} ({len(rows)} primitives)")

if __name__ == "__main__": main()
