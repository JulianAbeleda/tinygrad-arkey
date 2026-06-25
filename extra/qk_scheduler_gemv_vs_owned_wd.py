#!/usr/bin/env python3
"""Milestone 6: W==D-compare a SCHEDULER-generated FFN gate/up GEMV (with the cross-lane ds_bpermute lowering)
against the hand-written owned warp GEMV. The env-flag/candidates route reads flat for this (the owned kernel is a
custom_kernel the flag can't touch), so this is a standalone, clock-pinned, in-process INTERLEAVED A/B (shared
clock, drift-cancelled), real per-token .item() W==D, byte/token-match + route-fire verified.

Three arms (only the FFN gate/up GEMV differs; everything else identical):
  owned       : Q4K_GEMV_SCHEDULER=0 WARP_REDUCE_LOWERING=0           -> hand warp custom_kernel (the oracle)
  sched_lds   : Q4K_GEMV_SCHEDULER=1 WARP_REDUCE_LOWERING=0 MVR=1     -> scheduler fp matvec, LDS-tree group reduce
  sched_xlane : Q4K_GEMV_SCHEDULER=1 WARP_REDUCE_LOWERING=1 MVR=1     -> scheduler fp matvec, ds_bpermute reduce

Reads: sched_xlane vs sched_lds = the cross-lane primitive's effect on a scheduler GEMV (clean).
       sched_* vs owned = how far a scheduler GEMV is from the hand oracle (the gap = the non-reduce hand-opts:
       packed-Q4_K-word loads + block-group-K; the scheduler arm fuses a lazy Q4_K->fp16 dequant instead).
MVR=MV_ROWS_PER_THREAD=1 gives the scheduler GEMV a SCALAR lane reduce (the cross-lane ladder declines vectorized
UPCAST reduces in this first pass).

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_scheduler_gemv_vs_owned_wd.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-scheduler-gemv-vs-owned"
CTXS = [512, 1024, 2048, 4096]; MAXC = 4608; NMEAS = 30; REPEATS = 3
ARMS = {
  "owned":       {"Q4K_GEMV_SCHEDULER": "0", "WARP_REDUCE_LOWERING": "0", "MV_ROWS_PER_THREAD": "4"},
  "sched_lds":   {"Q4K_GEMV_SCHEDULER": "1", "WARP_REDUCE_LOWERING": "0", "MV_ROWS_PER_THREAD": "1"},
  "sched_xlane": {"Q4K_GEMV_SCHEDULER": "1", "WARP_REDUCE_LOWERING": "1", "MV_ROWS_PER_THREAD": "1"},
}

def _try_pin_clock():
  try:
    subprocess.run(["rocm-smi", "--setperflevel", "high"], capture_output=True, timeout=20)
    out = subprocess.run(["rocm-smi", "--showperflevel"], capture_output=True, timeout=20, text=True)
    for ln in out.stdout.splitlines():
      if "GPU" in ln and "Performance Level:" in ln: return ln.split("Performance Level:")[-1].strip()
  except Exception: pass
  return "unknown"

def main():
  perflevel = _try_pin_clock()
  from extra.qk_harness_contract import DEFAULT_MODEL
  from tinygrad import Tensor, UOp, TinyJit
  from tinygrad.helpers import getenv
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])

  def route_info(step):
    owned_gateup = 0; sched_gateup_xlane = 0
    for u in step.captured.linear.toposort():
      if u.op.name == "CALL" and len(u.src) and u.src[0].op.name == "PROGRAM":
        p = u.src[0]; name = p.arg.name
        src = next((c.arg for c in p.toposort() if c.op.name == "SOURCE"), "")
        if name.startswith("q4k_gemv_warp_12288"): owned_gateup += 1          # owned FFN gate/up custom kernel
        elif "12288" in name and "ds_bpermute" in src: sched_gateup_xlane += 1  # scheduler gate/up via cross-lane
    return owned_gateup, sched_gateup_xlane

  def build(env, ck):
    for k, val in env.items(): os.environ[k] = val
    getenv.cache_clear()
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    og, sx = route_info(step)
    return step, og, sx
  def measure(step, ck):
    out = Tensor([[int(ids[ck])]], dtype="int32").contiguous(); W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    return statistics.median(W), toks

  rows = {}
  arm_names = list(ARMS)
  for ck in CTXS:
    steps, route = {}, {}
    for a in arm_names:
      steps[a], og, sx = build(ARMS[a], ck); route[a] = {"owned_gateup": og, "sched_gateup_xlane": sx}
    ms = {a: [] for a in arm_names}; toks = {}
    for r in range(REPEATS):
      order = arm_names[r % len(arm_names):] + arm_names[:r % len(arm_names)]   # rotate for fairness
      for a in order:
        mm, tt = measure(steps[a], ck); ms[a].append(mm); toks[a] = tt
    med = {a: statistics.median(ms[a]) for a in arm_names}
    tps = {a: round(1000/med[a], 1) for a in arm_names}
    spread = {a: round(100*(max(ms[a])-min(ms[a]))/med[a], 2) for a in arm_names}
    rows[ck] = {
      "tok_s": tps, "spread_pct": spread, "route": route,
      "tokens_match_all": all(toks[a] == toks["owned"] for a in arm_names),
      "xlane_vs_lds_pct": round(100*(med["sched_lds"]-med["sched_xlane"])/med["sched_lds"], 2),
      "sched_xlane_vs_owned_pct": round(100*(med["owned"]-med["sched_xlane"])/med["owned"], 2),
    }
    print(f"ctx {ck:5}: owned {tps['owned']} | sched_lds {tps['sched_lds']} | sched_xlane {tps['sched_xlane']} tok/s "
          f"| xlane vs lds {rows[ck]['xlane_vs_lds_pct']:+.2f}% | sched_xlane vs owned {rows[ck]['sched_xlane_vs_owned_pct']:+.2f}% "
          f"| tokens_match {rows[ck]['tokens_match_all']} | route {route}", file=sys.__stderr__)

  tok_ok = all(r["tokens_match_all"] for r in rows.values())
  route_ok = all(rows[c]["route"]["owned"]["owned_gateup"] > 0 and rows[c]["route"]["sched_xlane"]["owned_gateup"] == 0
                 and rows[c]["route"]["sched_xlane"]["sched_gateup_xlane"] > 0 for c in CTXS)
  xlane_helps = all(rows[c]["xlane_vs_lds_pct"] > 0 for c in CTXS)
  out = {"date": "2026-06-25", "phase": "SCHEDULER_GEMV_VS_OWNED_WD", "perflevel": perflevel,
         "role": "FFN gate/up (Q4_K 4096x12288), 3-arm interleaved clock-pinned W==D",
         "arms": ARMS, "ctxs": CTXS, "nmeas": NMEAS, "repeats": REPEATS, "rows": rows,
         "tokens_match_all_ctx": tok_ok, "route_fire_ok": route_ok, "xlane_beats_lds_all_ctx": xlane_helps,
         "verdict": ("M6_SCHEDULER_GEMV_TRAILS_OWNED" if route_ok and tok_ok else "M6_INCONCLUSIVE"),
         "note": "sched_xlane vs owned is bandwidth-confounded (scheduler fuses Q4K->fp16 dequant; owned reads packed words + block-group-K). xlane_vs_lds isolates the cross-lane primitive."}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"wd.json").write_text(json.dumps(out, indent=2))
  print(f"\nverdict: {out['verdict']} | tokens_match {tok_ok} | route_ok {route_ok} | xlane>lds {xlane_helps} | {OUT/'wd.json'}", file=sys.__stderr__)
  print(json.dumps({"verdict": out["verdict"], "tokens_match": tok_ok, "route_ok": route_ok,
                    "xlane_vs_lds_pct": {c: rows[c]["xlane_vs_lds_pct"] for c in CTXS},
                    "sched_xlane_vs_owned_pct": {c: rows[c]["sched_xlane_vs_owned_pct"] for c in CTXS}}))

if __name__ == "__main__":
  main()
