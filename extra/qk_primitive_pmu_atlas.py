#!/usr/bin/env python3
"""In-model primitive PMU atlas — LEARN where decode time goes and WHY, from hardware counters.

Runs the REAL Qwen3-8B decode forward eagerly (so native PMC instruments every kernel), maps each
ProfilePMCEvent to its kernel name (PMC.kern == ProfileProgramEvent.tag -> .name), decodes the counter
blob, and aggregates per primitive. Output: a table sorted by GPU cycles with each primitive classified
bandwidth- / ALU- / latency-bound from MEASURED counters (L2 hit%, VALU utilization), not inference.

Run:  DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_primitive_pmu_atlas.py
Env:  QK_MODEL (default 8B), CTX (decode start_pos, default 512).
"""
from __future__ import annotations
import os, json, pathlib, collections
import numpy as np

def stats_of(ev):
  """viz-exact per-event stats: counter -> (sum_over_instances, max_over_instances, count)."""
  view = memoryview(ev.blob).cast('Q'); ptr = 0; st = {}
  for s in ev.sched:
    tot = mx = cnt = 0
    for _ in range(s.xcc*s.inst*s.se*s.sa):
      for _ in range(s.wgp):
        v = int(view[ptr]); tot += v; mx = max(mx, v); cnt += 1; ptr += 1
    st[s.name] = (tot, mx, cnt)
  return st

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  CTX = int(os.environ.get("CTX", "512")); MAXC = CTX + 64
  from tinygrad import Tensor, Device
  from tinygrad.device import Compiled
  from extra.llm_generate import load_model_and_tokenizer
  m, _ = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for b in m.blk: b._use_flash, b._prefill_v2 = (CTX >= 512), False
  tok = Tensor([[1]]); temp = Tensor(0.0)
  # warm: compile + clock ramp (eager, PMC on but we snapshot after)
  for i in range(6): m.forward(tok, CTX + i, temp).realize(); Device['AMD'].synchronize()
  base = len(Compiled.profile_events)
  m.forward(tok, CTX + 10, temp).realize()
  Device['AMD'].synchronize(); Device['AMD']._at_profile_finalize()
  evs = Compiled.profile_events[base:]
  # ProfileProgramEvents are emitted at COMPILE time (during warmup, before `base`) -> scan the FULL list for names.
  names = {e.tag: e.name for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent" and getattr(e, "tag", None) is not None}
  agg = collections.defaultdict(lambda: {"n":0, "active":0, "valu_util":[], "salu_util":[], "l2hit":[], "busy":0})
  for ev in evs:
    if type(ev).__name__ != "ProfilePMCEvent": continue
    nm = names.get(ev.kern, f"kern{ev.kern}")
    s = stats_of(ev)
    active = s.get("GRBM_GUI_ACTIVE", (0,0,1))[1]
    a = agg[nm]; a["n"] += 1; a["active"] += active; a["busy"] += s.get("SQ_BUSY_CYCLES",(0,0,1))[0]
    if active > 0:
      vsum,_,vcnt = s.get("SQ_INSTS_VALU",(0,0,1)); ssum,_,scnt = s.get("SQ_INSTS_SALU",(0,0,1))
      a["valu_util"].append(100*(vsum/max(vcnt,1))/(active*4)); a["salu_util"].append(100*(ssum/max(scnt,1))/(active*4))
    hit,miss = s.get("GL2C_HIT",(0,0,1))[0], s.get("GL2C_MISS",(0,0,1))[0]
    if hit+miss>0: a["l2hit"].append(100*hit/(hit+miss))
  rows = []
  for nm,a in agg.items():
    vu = float(np.mean(a["valu_util"])) if a["valu_util"] else 0.0
    l2 = float(np.mean(a["l2hit"])) if a["l2hit"] else -1.0
    cls = ("ALU-bound" if vu>=40 else "bandwidth-bound" if 0<=l2<30 else "cache/latency-bound" if l2>=30 else "?")
    rows.append({"primitive":nm.replace("\x1b[36m","").replace("\x1b[0m",""), "launches":a["n"],
                 "tot_active_cyc":a["active"], "VALU_util%":round(vu,1), "L2_hit%":round(l2,1), "class":cls})
  rows.sort(key=lambda r:-r["tot_active_cyc"]); tot=sum(r["tot_active_cyc"] for r in rows) or 1
  for r in rows: r["gpu_time%"]=round(100*r["tot_active_cyc"]/tot,1)
  print(f"{'primitive':<34}{'launch':>7}{'gpu%':>7}{'VALU%':>7}{'L2hit%':>8}  class")
  for r in rows[:30]:
    print(f"{r['primitive'][:33]:<34}{r['launches']:>7}{r['gpu_time%']:>7}{r['VALU_util%']:>7}{r['L2_hit%']:>8}  {r['class']}")
  out = pathlib.Path("bench/qk-primitive-pmu-atlas"); out.mkdir(parents=True, exist_ok=True)
  (out/"result.json").write_text(json.dumps({"model":pathlib.Path(model).stem,"ctx":CTX,"rows":rows}, indent=1))
  print(f"\nwrote {out/'result.json'} ({len(rows)} primitives)")

if __name__ == "__main__": main()
