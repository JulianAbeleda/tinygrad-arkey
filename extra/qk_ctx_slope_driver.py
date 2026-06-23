#!/usr/bin/env python3
"""Ctx-slope audit driver: interleaved-rep W==D for Config A (default whole-cache buffer-identity route) vs
Config B (DECODE_ATTN_KV_IDENTITY=0 slice/materialization route) at ctx 512/1024/2048/4096.

Authority: each rep subprocesses extra/qk_decode_runtime_overhead.py (W path = real decode .item()/token sync,
the only promotion-grade authority per bench/qk-decode-eval/HARNESS_GUIDE.md). We run REPS interleaved A,B,A,B,...
to share thermal/clock drift between configs, then report per-ctx median + spread band across reps.

  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_ctx_slope_driver.py
Env: QK_SLOPE_REPS (default 3), QK_CKPTS (default 512,1024,2048,4096).
Writes bench/qk-decode-ctx-slope-audit/wd_by_ctx.json (does NOT change any default).
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-ctx-slope-audit"
RUNTIME_ART = ROOT / "bench/qk-decode-runtime-overhead/result.json"
CKPTS = os.environ.get("QK_CKPTS", "512,1024,2048,4096")
REPS = int(os.environ.get("QK_SLOPE_REPS", "3"))
CONFIGS = {"A_whole_default": {}, "B_slice_identity0": {"DECODE_ATTN_KV_IDENTITY": "0"}}

def run_once(extra_env: dict) -> dict:
  env = dict(os.environ); env.update({"DEV": "AMD", "JIT": "1", "PYTHONPATH": ".", "QK_CKPTS": CKPTS}); env.update(extra_env)
  r = subprocess.run([sys.executable, "extra/qk_decode_runtime_overhead.py"], cwd=ROOT, env=env,
                     capture_output=True, text=True)
  if r.returncode != 0 or "@@DONE@@" not in r.stdout:
    sys.stderr.write(r.stdout + "\n" + r.stderr + "\n"); raise SystemExit(f"runtime_overhead failed (rc={r.returncode})")
  d = json.loads(RUNTIME_ART.read_text())
  return {str(row["ctx"]): {"tok_s_W": row["tok_s_W"], "wall_ms_W": row["wall_ms_W"],
                            "dispatch_ms_D": row["dispatch_ms_D"], "host_pct": row["host_sync_pct_of_wall"],
                            "progs": row["programs_per_token"], "flash": row["flash"]} for row in d["rows"]}

def band(xs):
  xs = sorted(xs); med = statistics.median(xs)
  return {"median": round(med, 3), "min": round(xs[0], 3), "max": round(xs[-1], 3),
          "mean": round(statistics.mean(xs), 3), "n": len(xs),
          "spread_pct": round(100 * (xs[-1] - xs[0]) / med, 2) if med else 0.0}

def main():
  ckpts = CKPTS.split(",")
  raw = {c: {ck: [] for ck in ckpts} for c in CONFIGS}
  raw_ms = {c: {ck: [] for ck in ckpts} for c in CONFIGS}
  meta = {c: {} for c in CONFIGS}
  for rep in range(REPS):
    for cname, cenv in CONFIGS.items():
      sys.stderr.write(f"[rep {rep+1}/{REPS}] {cname} ...\n"); sys.stderr.flush()
      res = run_once(cenv)
      for ck in ckpts:
        raw[cname][ck].append(res[ck]["tok_s_W"]); raw_ms[cname][ck].append(res[ck]["wall_ms_W"])
        meta[cname][ck] = {"progs": res[ck]["progs"], "flash": res[ck]["flash"], "host_pct": res[ck]["host_pct"]}
  out = {"ckpts": ckpts, "reps": REPS, "interleave": "A,B per rep (shared thermal/clock drift)",
         "authority": "W path of qk_decode_runtime_overhead (.item()/token sync); per-ctx median over reps",
         "configs": {}, "raw_tok_s": raw}
  for cname in CONFIGS:
    out["configs"][cname] = {ck: {"tok_s": band(raw[cname][ck]), "ms_token": band(raw_ms[cname][ck]),
                                  **meta[cname][ck]} for ck in ckpts}
  # delta A vs B
  delta = {}
  for ck in ckpts:
    a = statistics.median(raw[ "A_whole_default"][ck]); b = statistics.median(raw["B_slice_identity0"][ck])
    a_ms = statistics.median(raw_ms["A_whole_default"][ck]); b_ms = statistics.median(raw_ms["B_slice_identity0"][ck])
    delta[ck] = {"A_tok_s": round(a, 2), "B_tok_s": round(b, 2), "delta_pct_tok_s": round(100 * (a - b) / b, 2),
                 "A_ms": round(a_ms, 3), "B_ms": round(b_ms, 3), "saved_ms": round(b_ms - a_ms, 3),
                 "saved_pct_of_B": round(100 * (b_ms - a_ms) / b_ms, 2)}
  out["delta_A_vs_B"] = delta
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "wd_by_ctx.json").write_text(json.dumps(out, indent=2))
  sys.stderr.write("\n=== delta A(whole) vs B(slice) ===\n")
  for ck in ckpts:
    d = delta[ck]
    sys.stderr.write(f"ctx {ck:>4}: A {d['A_tok_s']:.1f} | B {d['B_tok_s']:.1f} | +{d['delta_pct_tok_s']:.1f}% | "
                     f"saved {d['saved_ms']:.3f}ms ({d['saved_pct_of_B']:.1f}% of B)\n")
  sys.stderr.write(f"\nartifact: {OUT/'wd_by_ctx.json'}\n")
  print("@@DONE@@")

if __name__ == "__main__":
  main()
