#!/usr/bin/env python3
"""W==D for the WORD-STRUCTURED packed-Q4_K scheduler GEMV (Q4K_GEMV_SCHEDULER=2, extra/qk_q4k_scheduler_gemv)
vs the fp16-logical scheduler GEMV (=1, the M6 arm) vs the owned warp kernel. FFN gate/up only. Clock-pinned,
in-process interleaved, real per-token .item() W==D, tokens_match. Tests whether reading packed uint32 WORDS
(vs the logical fp16 dequant) closes the ~2x gap to owned -- i.e. whether the scheduler coalesces the word loads.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_q4k_packed_gemv_wd.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-scheduler-gemv-vs-owned"
CTXS = [512, 1024, 2048, 4096]; MAXC = 4608; NMEAS = 30; REPEATS = 3
ARMS = {
  "owned":       {"Q4K_GEMV_SCHEDULER": "0"},
  "sched_fp16":  {"Q4K_GEMV_SCHEDULER": "1", "MV_ROWS_PER_THREAD": "1"},
  "sched_packed":{"Q4K_GEMV_SCHEDULER": "2"},
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

  def owned_gateup(step):
    return sum(1 for u in step.captured.linear.toposort()
               if u.op.name == "CALL" and len(u.src) and u.src[0].op.name == "PROGRAM"
               and u.src[0].arg.name.startswith("q4k_gemv_warp_12288"))
  def build(env, ck):
    for k in ("Q4K_GEMV_SCHEDULER", "MV_ROWS_PER_THREAD", "WARP_REDUCE_LOWERING"): os.environ.pop(k, None)
    for k, val in env.items(): os.environ[k] = val
    getenv.cache_clear()
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    return step, owned_gateup(step)
  def measure(step, ck):
    out = Tensor([[int(ids[ck])]], dtype="int32").contiguous(); W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    return statistics.median(W), toks

  rows = {}; arm_names = list(ARMS)
  for ck in CTXS:
    steps, og = {}, {}
    for a in arm_names: steps[a], og[a] = build(ARMS[a], ck)
    ms = {a: [] for a in arm_names}; toks = {}
    for r in range(REPEATS):
      order = arm_names[r % len(arm_names):] + arm_names[:r % len(arm_names)]
      for a in order:
        mm, tt = measure(steps[a], ck); ms[a].append(mm); toks[a] = tt
    med = {a: statistics.median(ms[a]) for a in arm_names}
    tps = {a: round(1000/med[a], 1) for a in arm_names}
    rows[ck] = {"tok_s": tps, "owned_gateup": og,
                "tokens_match_all": all(toks[a] == toks["owned"] for a in arm_names),
                "packed_vs_fp16_pct": round(100*(med["sched_fp16"]-med["sched_packed"])/med["sched_fp16"], 2),
                "packed_vs_owned_pct": round(100*(med["owned"]-med["sched_packed"])/med["owned"], 2),
                "spread_pct": {a: round(100*(max(ms[a])-min(ms[a]))/med[a], 2) for a in arm_names}}
    print(f"ctx {ck:5}: owned {tps['owned']} | sched_fp16 {tps['sched_fp16']} | sched_packed {tps['sched_packed']} tok/s "
          f"| packed vs fp16 {rows[ck]['packed_vs_fp16_pct']:+.2f}% | packed vs owned {rows[ck]['packed_vs_owned_pct']:+.2f}% "
          f"| tokens_match {rows[ck]['tokens_match_all']} | owned_gateup {og}", file=sys.__stderr__)

  tok_ok = all(r["tokens_match_all"] for r in rows.values())
  route_ok = all(rows[c]["owned_gateup"]["owned"] > 0 and rows[c]["owned_gateup"]["sched_packed"] == 0 for c in CTXS)
  out = {"date": "2026-06-25", "phase": "Q4K_PACKED_SCHEDULER_GEMV_WD", "perflevel": perflevel,
         "role": "FFN gate/up (Q4_K 4096x12288)", "arms": ARMS, "ctxs": CTXS, "nmeas": NMEAS, "repeats": REPEATS,
         "rows": rows, "tokens_match_all_ctx": tok_ok, "route_fire_ok": route_ok,
         "verdict": ("Q4K_PACKED_SCHEDULER_GEMV_TRAILS_OWNED" if route_ok and tok_ok else "INCONCLUSIVE")}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"packed_wd.json").write_text(json.dumps(out, indent=2))
  print(f"\nverdict: {out['verdict']} | tokens_match {tok_ok} | route_ok {route_ok} | {OUT/'packed_wd.json'}", file=sys.__stderr__)
  print(json.dumps({"verdict": out["verdict"], "tokens_match": tok_ok,
                    "packed_vs_fp16_pct": {c: rows[c]["packed_vs_fp16_pct"] for c in CTXS},
                    "packed_vs_owned_pct": {c: rows[c]["packed_vs_owned_pct"] for c in CTXS}}))

if __name__ == "__main__":
  main()
