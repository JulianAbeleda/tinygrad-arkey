#!/usr/bin/env python3
"""W==D for the attn q/o projection warp-GEMV lever (Q4K_GEMV_WARP_PROJ=1) vs the shipped default, in-process
INTERLEAVED A/B (alternating order each repeat = shared clock, drift-cancelled), real per-token .item() sync,
route-firing verified (the proj kernel is `q4k_gemv_warp_4096_4096`), byte-identical token check.

Resolves the conflict: the 2026-06-22 promotion-hardening audit marked PROJ "research-only / no W==D transfer",
but the 2026-06-24 aggressive-target-proof measured a clean +1.5%/ctx -- that proof ran the arms as SEPARATE
sequential blocks under `auto` clock (the documented clock-confound). This is the drift-cancelled re-test.

Both arms keep the SHIPPED default Q4K_GEMV_WARP=1 + Q4K_GEMV_WARP_DOWN=1 (FFN), toggling ONLY PROJ, so the
delta isolates the q/o projection lever on top of the real default route. Default unchanged either way.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/proj_gemv_warp_wd.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-proj-gemv-warp"
CTXS = [512, 1024, 2048, 4096]; MAXC = 4608; NMEAS = 40; REPEATS = 6
PROJ_KERNEL = "q4k_gemv_warp_4096_4096"   # the q/o projection warp kernel (out=4096, in=4096)

def _try_pin_clock():
  """Best-effort clock pin (rocm-smi --setperflevel high; no root needed per repo notes). Returns level used."""
  try:
    subprocess.run(["rocm-smi", "--setperflevel", "high"], capture_output=True, timeout=20)
    out = subprocess.run(["rocm-smi", "--showperflevel"], capture_output=True, timeout=20, text=True)
    for ln in out.stdout.splitlines():
      if "GPU" in ln and "Performance Level:" in ln: return ln.split("Performance Level:")[-1].strip()
  except Exception:
    pass
  return "unknown"

def main():
  perflevel = _try_pin_clock()
  from extra.qk.harness_contract import DEFAULT_MODEL
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  from tinygrad import Tensor, UOp, TinyJit, Device
  from tinygrad.helpers import getenv
  from extra.llm.generate import load_model_and_tokenizer
  dev = Device["AMD"]
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])

  # shipped FFN default in BOTH arms; isolate the PROJ toggle.
  os.environ["Q4K_GEMV_WARP"] = "1"; os.environ["Q4K_GEMV_WARP_DOWN"] = "1"

  def build(proj_flag, ck):
    os.environ["Q4K_GEMV_WARP_PROJ"] = str(proj_flag); getenv.cache_clear()
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    names = [u.src[0].arg.name for u in step.captured.linear.toposort()
             if u.op.name == "CALL" and len(u.src) and u.src[0].op.name == "PROGRAM"]
    return step, sum(PROJ_KERNEL in n for n in names)
  def measure(step, ck):
    out = Tensor([[int(ids[ck])]], dtype="int32").contiguous(); W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    return statistics.median(W), toks

  rows = {}
  for ck in CTXS:
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    base_step, base_proj = build(0, ck); proj_step, proj_n = build(1, ck)
    base_ms, proj_ms, bt, pt = [], [], None, None
    for r in range(REPEATS):
      if r % 2 == 0:
        bm, bt = measure(base_step, ck); base_ms.append(bm); pm, pt = measure(proj_step, ck); proj_ms.append(pm)
      else:
        pm, pt = measure(proj_step, ck); proj_ms.append(pm); bm, bt = measure(base_step, ck); base_ms.append(bm)
    bms, pms = statistics.median(base_ms), statistics.median(proj_ms)
    rows[ck] = {"base_tok_s": round(1000/bms, 1), "proj_tok_s": round(1000/pms, 1),
                "delta_pct": round(100*(bms-pms)/bms, 2), "tokens_match": bt == pt,
                "proj_kernels_fired": proj_n, "base_proj_kernels": base_proj,
                "base_spread_pct": round(100*(max(base_ms)-min(base_ms))/bms, 2),
                "proj_spread_pct": round(100*(max(proj_ms)-min(proj_ms))/pms, 2)}
    print(f"ctx {ck:5}: base {rows[ck]['base_tok_s']} vs +proj {rows[ck]['proj_tok_s']} tok/s -> "
          f"{rows[ck]['delta_pct']:+.2f}% | tokens_match {rows[ck]['tokens_match']} | proj_fired {proj_n} "
          f"(base {base_proj}) | spread b{rows[ck]['base_spread_pct']}/p{rows[ck]['proj_spread_pct']}%",
          file=sys.__stderr__)

  d = {c: rows[c]["delta_pct"] for c in CTXS}
  tok_ok = all(r["tokens_match"] for r in rows.values())
  fired_ok = all(rows[c]["proj_kernels_fired"] > 0 and rows[c]["base_proj_kernels"] == 0 for c in CTXS)
  # promotion gate: byte-identical + proj route fires + a real interleaved win above the worst per-arm spread floor
  worst_spread = max(max(rows[c]["base_spread_pct"], rows[c]["proj_spread_pct"]) for c in CTXS)
  min_delta = min(d.values())
  transfers = tok_ok and fired_ok and min_delta > 0 and min_delta >= 0.5 * worst_spread
  verdict = ("PROJ_GEMV_WARP_WD_TRANSFERS_PROMOTABLE" if transfers else
             "PROJ_GEMV_WARP_WD_WITHIN_NOISE_RESEARCH_ONLY" if (tok_ok and fired_ok) else
             "PROJ_GEMV_WARP_BLOCKED_IMPLEMENTATION")
  out = {"date": "2026-06-25", "phase": "PROJ_GEMV_WARP_WD_INTERLEAVED", "perflevel": perflevel,
         "comparator": "shipped default (Q4K_GEMV_WARP=1, DOWN=1, PROJ=0)",
         "route": "Q4K_GEMV_WARP_PROJ=1 (attn q/o 4096x4096, default-off/research-only)",
         "method": "in-process interleaved alternating A/B, real per-token .item() W==D, drift-cancelled",
         "ctxs": CTXS, "nmeas": NMEAS, "repeats": REPEATS, "rows": rows, "delta_pct": d,
         "all_tokens_match": tok_ok, "proj_route_fired": fired_ok, "worst_spread_pct": worst_spread,
         "min_delta_pct": min_delta, "transfers": transfers, "verdict": verdict, "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"wd.json").write_text(json.dumps(out, indent=2))
  print(f"\nverdict: {verdict} | perflevel {perflevel} | artifact: {OUT/'wd.json'}", file=sys.__stderr__)
  print(json.dumps({"verdict": verdict, "delta_pct": d, "tokens_match": tok_ok, "proj_route_fired": fired_ok,
                    "worst_spread_pct": worst_spread}))

if __name__ == "__main__":
  main()
