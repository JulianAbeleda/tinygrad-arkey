#!/usr/bin/env python3
"""W==D for the lossless q4k_gemv_warp FFN gate/up route (Q4K_GEMV_WARP=1) vs the default, in-process interleaved A/B
(alternating order, shared clock; real per-token .item() sync; route-firing verified). The diagnostic/scope gate:
>=+5%@ctx1024 OR >=+7%@ctx4096, no ctx512 regression, tokens match. Default stays OFF.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/ffn_gemv_warp_wd.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-ffn-gemv-warp"
CTXS = [512, 1024, 4096]; MAXC = 4608; NMEAS = 40; REPEATS = 6

def main():
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

  def build(flag, ck):
    os.environ["Q4K_GEMV_WARP"] = str(flag); getenv.cache_clear()
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    names = [u.src[0].arg.name for u in step.captured.linear.toposort()
             if u.op.name == "CALL" and len(u.src) and u.src[0].op.name == "PROGRAM"]
    return step, sum("q4k_gemv_warp" in n for n in names)
  def measure(step, ck):
    out = Tensor([[int(ids[ck])]], dtype="int32").contiguous(); W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    return statistics.median(W), toks

  rows = {}
  for ck in CTXS:
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    base_step, base_warp = build(0, ck); warp_step, warp_n = build(1, ck)
    base_ms, warp_ms, bt, wt = [], [], None, None
    for r in range(REPEATS):
      if r % 2 == 0:
        bm, bt = measure(base_step, ck); base_ms.append(bm); wm, wt = measure(warp_step, ck); warp_ms.append(wm)
      else:
        wm, wt = measure(warp_step, ck); warp_ms.append(wm); bm, bt = measure(base_step, ck); base_ms.append(bm)
    bms, wms = statistics.median(base_ms), statistics.median(warp_ms)
    rows[ck] = {"base_tok_s": round(1000/bms, 1), "warp_tok_s": round(1000/wms, 1),
                "delta_pct": round(100*(bms-wms)/bms, 2), "tokens_match": bt == wt,
                "warp_kernels_fired": warp_n, "base_warp_kernels": base_warp,
                "base_spread_pct": round(100*(max(base_ms)-min(base_ms))/bms, 2),
                "warp_spread_pct": round(100*(max(warp_ms)-min(warp_ms))/wms, 2)}
    print(f"ctx {ck:5}: base {rows[ck]['base_tok_s']} vs warp {rows[ck]['warp_tok_s']} tok/s -> {rows[ck]['delta_pct']:+.2f}% "
          f"| tokens_match {rows[ck]['tokens_match']} | warp_fired {warp_n}", file=sys.__stderr__)

  d512, d1024, d4096 = rows[512]["delta_pct"], rows[1024]["delta_pct"], rows[4096]["delta_pct"]
  tok_ok = all(r["tokens_match"] for r in rows.values())
  gate = tok_ok and d512 >= -1.0 and (d1024 >= 5.0 or d4096 >= 7.0)
  verdict = ("Q4K_GEMV_WARP_WD_PASS" if gate else
             "Q4K_GEMV_WARP_LOCAL_PASS_WD_FAIL" if tok_ok else "Q4K_GEMV_WARP_BLOCKED_IMPLEMENTATION")
  out = {"date": "2026-06-22", "phase": "Q4K_GEMV_WARP_WD", "comparator": "default q4k_gemv_partial gate/up",
         "route": "Q4K_GEMV_WARP=1 (FFN gate/up only, default-off)", "ctxs": CTXS, "rows": rows,
         "gate_rule": "(d1024>=+5% OR d4096>=+7%) AND no ctx512 regress AND tokens match", "all_tokens_match": tok_ok,
         "wd_gate_pass": gate, "verdict": verdict, "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"wd.json").write_text(json.dumps(out, indent=2))
  print(f"\nverdict: {verdict} | artifact: {OUT/'wd.json'}", file=sys.__stderr__)
  print(json.dumps({"verdict": verdict, "delta_pct": {c: rows[c]["delta_pct"] for c in CTXS}}))

if __name__ == "__main__":
  main()
