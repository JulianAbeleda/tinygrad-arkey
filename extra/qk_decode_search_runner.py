"""Decode machine-search RUNNER — the safe constrained candidate runner (generate -> evaluate -> prune -> rank ->
remember). Spawns the gate (extra/qk_decode_search_gate.py) once per candidate in a subprocess with the candidate's
bounded knob env vars set, applies the hard reject rules (in the gate, cost-ordered, short-circuiting), ranks the
PASSing candidates by W==D delta vs the frozen oracle, checks ctx512 regression, and remembers results.

THIS DOES NOT RUN A REAL SEARCH. It freezes the oracle and smoke-tests the framework on a tiny grid (the oracle +
deliberately-bad candidates) to prove the gates distinguish good from bad. A real search is a separate authorized step.

  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_search_runner.py [--smoke]
See docs/decode-machine-search-readiness-package-scope-20260623.md (P0,P7)."""
from __future__ import annotations
import os, sys, json, subprocess, pathlib

OUT = pathlib.Path("bench/qk-decode-search-readiness"); OUT.mkdir(parents=True, exist_ok=True)
ORACLE_FILE = OUT / "baseline_oracle.json"
GATE = "extra/qk_decode_search_gate.py"

# The ONLY allowed knobs map to these env vars (bounded; see scope). A candidate = {id, env:{...}}.
def run_candidate(cand, oracle_tokens_file=None):
  env = {**os.environ, "QK_CAND_ID": cand["id"], "DEV": "AMD", "JIT": "1", **{k: str(v) for k, v in cand["env"].items()}}
  argv = [sys.executable, GATE]
  if oracle_tokens_file: argv += ["--oracle-tokens", str(oracle_tokens_file)]
  r = subprocess.run(argv, capture_output=True, text=True, env=env)
  for line in r.stdout.splitlines():
    if line.startswith("RESULT "): return json.loads(line[len("RESULT "):])
  return {"id": cand["id"], "verdict": "GATE_CRASH", "reject_reason": "gate_crash", "stderr_tail": r.stderr[-400:]}

def freeze_oracle():
  print("[P0] freezing oracle (current buffer-identity default)...", file=sys.stderr)
  oracle_cand = {"id": "oracle", "env": {"DECODE_ATTN_KV_IDENTITY": 1}}
  res = run_candidate(oracle_cand)  # no oracle file yet -> records tokens, no correctness compare
  res["role"] = "frozen_oracle"
  json.dump(res, open(ORACLE_FILE, "w"), indent=2)
  print(f"[P0] oracle: verdict={res['verdict']} tokens={res.get('tokens')} wd={res.get('wd')}", file=sys.stderr)
  return res

def main():
  smoke = "--smoke" in sys.argv or True  # this task is smoke-only by design
  oracle = freeze_oracle()
  ok = oracle["verdict"] == "PASS"
  # smoke grid: the oracle (re-run vs frozen tokens) + deliberately-bad candidates that MUST be rejected
  grid = [
    {"id": "oracle_replay", "env": {"DECODE_ATTN_KV_IDENTITY": 1}, "expect": "PASS"},
    {"id": "bad_slice_route", "env": {"DECODE_ATTN_KV_IDENTITY": 0}, "expect": "REJECTED:E_49152_returned"},
    {"id": "bad_no_route", "env": {"DECODE_ATTN_AMDGCN_TILE": 0}, "expect": "REJECTED:route_not_firing"},
  ]
  results = []
  for c in grid:
    res = run_candidate(c, oracle_tokens_file=ORACLE_FILE)
    res["expect"] = c["expect"]; res["smoke_ok"] = res["verdict"].split(":")[0] == c["expect"].split(":")[0]
    results.append(res)
    print(f"[P7] {c['id']:18} verdict={res['verdict']:38} expect={c['expect']:30} {'OK' if res['smoke_ok'] else 'SMOKE-FAIL'}", file=sys.stderr)
  # rank PASS candidates by W==D delta vs oracle (smoke: only oracle_replay should PASS)
  passing = [r for r in results if r["verdict"] == "PASS"]
  o512 = (oracle.get("wd") or {}).get("512", {}).get("tok_s"); o1024 = (oracle.get("wd") or {}).get("1024", {}).get("tok_s")
  for r in passing:
    w = r.get("wd", {}); r["delta_vs_oracle_pct"] = (round(100*(w.get("1024",{}).get("tok_s",0)-o1024)/o1024, 1) if o1024 else None)
    r["ctx512_regression"] = (w.get("512",{}).get("tok_s", 0) < (o512 or 0) * 0.98) if o512 else None
  leaderboard = sorted(passing, key=lambda r: -(r.get("wd",{}).get("1024",{}).get("tok_s") or 0))
  smoke_pass = all(r["smoke_ok"] for r in results)
  summary = {"date": "2026-06-23", "phase": "SEARCH_RUNNER_SMOKE", "oracle_verdict": oracle["verdict"],
             "oracle_wd": oracle.get("wd"), "smoke_results": [{k: r.get(k) for k in ["id","verdict","expect","smoke_ok","delta_vs_oracle_pct"]} for r in results],
             "leaderboard_top": [r["id"] for r in leaderboard],
             "verdict": "SEARCH_RUNNER_READY" if (smoke_pass and ok) else "SEARCH_RUNNER_SMOKE_FAIL_STOP"}
  json.dump(summary, open(OUT / "search_runner_smoke.json", "w"), indent=2)
  print("RUNNER " + json.dumps(summary))

if __name__ == "__main__":
  main()
