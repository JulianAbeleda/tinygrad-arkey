#!/usr/bin/env python3
"""Settle the Tensile "86% of llama" prefill claim: measure baseline WMMA, the dependency-free graph route, and
the external Tensile .co route on ONE synced axis (plus nosync, to expose which number the 86% was).

Each mode runs in a fresh subprocess (clean VRAM) via the existing synced worker
(extra/qk_prefill_graph_gemm_default_perf.py --worker), which reports both SYNCED arbiter (K forwards/one sync)
and NOSYNC (qk_prefill_v2_measure-style) ms/512. The routing is selected purely by env:
  baseline : PREFILL_V2=1
  graph    : PREFILL_V2=1 PREFILL_GRAPH_GEMM=1
  tensile  : PREFILL_V2=1 PREFILL_GRAPH_GEMM=0 PREFILL_TENSILE_GEMM=1

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_tensile_settle.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys

WORKER = "extra/qk_prefill_graph_gemm_default_perf.py"
LLAMA_PP512 = 3020.0  # synced llama-bench reference
MODES = {
  "baseline_wmma": {"PREFILL_V2": "1", "PREFILL_GRAPH_GEMM": "0"},
  "graph_gemm":    {"PREFILL_V2": "1", "PREFILL_GRAPH_GEMM": "1"},
  "tensile_co":    {"PREFILL_V2": "1", "PREFILL_GRAPH_GEMM": "0", "PREFILL_TENSILE_GEMM": "1"},
}


def run(model_path, envextra):
  env = dict(os.environ); env["DEV"] = "AMD"; env["PYTHONPATH"] = "."
  for k in ("PREFILL_GRAPH_GEMM", "PREFILL_TENSILE_GEMM"): env.pop(k, None)
  env.update(envextra)
  p = subprocess.run([sys.executable, WORKER, "--worker", model_path], env=env, capture_output=True, text=True, timeout=1200)
  if p.returncode != 0: return {"rc": p.returncode, "err": p.stderr[-400:]}
  ls = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
  if not ls: return {"rc": -1, "err": "no json: " + p.stdout[-200:] + " | " + p.stderr[-200:]}
  d = json.loads(ls[-1]); d["rc"] = 0; return d


def main() -> int:
  model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  rows = {}
  for name, envextra in MODES.items():
    r = run(model_path, envextra)
    if r.get("rc") == 0:
      r["pct_llama_synced"] = round(100 * r["toks_synced"] / LLAMA_PP512, 1)
      r["pct_llama_nosync"] = round(100 * r["toks_nosync"] / LLAMA_PP512, 1)
      print(f"  {name:14}: synced {r['ms512_synced']}ms ({r['toks_synced']} tok/s, {r['pct_llama_synced']}% llama) "
            f"| nosync {r['ms512_nosync']}ms ({r['toks_nosync']} tok/s, {r['pct_llama_nosync']}% llama)")
    else:
      print(f"  {name:14}: FAILED rc={r['rc']} {r.get('err','')[:160]}")
    rows[name] = r

  t = rows.get("tensile_co", {})
  verdict, why = "INCONCLUSIVE", ""
  if t.get("rc") == 0:
    ts, tn = t["pct_llama_synced"], t["pct_llama_nosync"]
    if ts >= 80:
      verdict = "TENSILE_86_REAL_SYNCED"; why = f"Tensile synced {ts}% llama (~86% holds) -> real win above the dep-free graph route ({rows['graph_gemm'].get('pct_llama_synced')}%); the .co integrates better in-model."
    elif tn >= 80 and ts < 70:
      verdict = "TENSILE_86_WAS_NOSYNC"; why = f"Tensile nosync {tn}% llama (~86%) but synced only {ts}% -> the 86% was a NOSYNC artifact; real synced Tensile = {ts}% llama."
    else:
      verdict = "TENSILE_PARTIAL"; why = f"Tensile synced {ts}% / nosync {tn}% llama -> 86% partly inflated; real synced = {ts}%."
  elif t.get("rc") is not None:
    verdict = "TENSILE_ROUTE_FAILED"; why = t.get("err", "")[:200]
  result = {"date": "2026-06-20", "schema": "prefill_tensile_settle_v1", "llama_pp512": LLAMA_PP512,
            "rows": rows, "verdict": verdict, "why": why}
  out = pathlib.Path("bench/amd-broad-backend-roadmap"); out.mkdir(parents=True, exist_ok=True)
  (out / "prefill_tensile_settle_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(f"\n{verdict}\n  {why}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
