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

def profile_ctx(m, CTX):
  from tinygrad import Tensor, Device
  from tinygrad.device import Compiled
  for b in m.blk: b._use_flash, b._prefill_v2 = (CTX >= 512), False
  tok = Tensor([[1]]); temp = Tensor(0.0)
  for i in range(6): m.forward(tok, CTX + i, temp).realize(); Device['AMD'].synchronize()
  base = len(Compiled.profile_events)
  m.forward(tok, CTX + 10, temp).realize()
  Device['AMD'].synchronize(); Device['AMD']._at_profile_finalize()
  evs = Compiled.profile_events[base:]
  names = {e.tag: e.name for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent" and getattr(e, "tag", None) is not None}
  agg = collections.defaultdict(lambda: {"n":0, "active":0, "valu_util":[], "l2hit":[]})
  for ev in evs:
    if type(ev).__name__ != "ProfilePMCEvent": continue
    nm = names.get(ev.kern, f"kern{ev.kern}"); s = stats_of(ev)
    active = s.get("GRBM_GUI_ACTIVE", (0,0,1))[1]; a = agg[nm]; a["n"] += 1; a["active"] += active
    if active > 0:
      vsum,_,vcnt = s.get("SQ_INSTS_VALU",(0,0,1)); a["valu_util"].append(100*(vsum/max(vcnt,1))/(active*4))
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
  bw = sum(r["gpu_time%"] for r in rows if r["class"]=="bandwidth-bound")
  cache = sum(r["gpu_time%"] for r in rows if r["class"]=="cache/latency-bound")
  return rows, bw, cache

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ctxs = [int(x) for x in os.environ.get("CTX", "512").split(",")]; MAXC = max(ctxs) + 64
  from extra.llm_generate import load_model_and_tokenizer
  m, _ = load_model_and_tokenizer(model, MAXC, seed=20260617)
  allres = {}
  for CTX in ctxs:
    rows, bw, cache = profile_ctx(m, CTX); allres[CTX] = {"bw_bound_gpu%":round(bw,1), "cache_served_gpu%":round(cache,1), "rows":rows}
    print(f"\n=== ctx {CTX}: bandwidth-bound {bw:.0f}% | cache-served {cache:.0f}% of decode GPU time ===")
    print(f"{'primitive':<30}{'launch':>6}{'gpu%':>6}{'VALU%':>6}{'L2hit%':>7}  class")
    for r in rows[:10]:
      print(f"{r['primitive'][:29]:<30}{r['launches']:>6}{r['gpu_time%']:>6}{r['VALU_util%']:>6}{r['L2_hit%']:>7}  {r['class']}")
  out = pathlib.Path("bench/qk-primitive-pmu-atlas"); out.mkdir(parents=True, exist_ok=True)
  (out/"result.json").write_text(json.dumps({"model":pathlib.Path(model).stem,"by_ctx":allres}, indent=1))
  print(f"\nwrote {out/'result.json'}")

if __name__ == "__main__": main()
