"""Role-selective prefill pipe: harden vs the SHIPPED global pipe. 3 arms via the recovered synced harness
(extra/qk/prefill_pipe_hardening.py), one subprocess per arm (arm chosen by env). Decides whether excluding ffn_gate_up
(out_f==12288) from the pipe beats the global pipe, honestly accounting for run noise.

  old_lds2      : PREFILL_GEMM_PIPELINE=0
  global_pipe   : (defaults -> PIPELINE=1 TM=2 TN=2)
  role_selective: PREFILL_PIPE_ROLE_SELECTIVE=1  (gate_up -> lds2, rest -> pipe)

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/prefill_pipe_role_selective.py
"""
import os, sys, json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-prefill-pipe-role-selective"; OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"old_lds2": {"PREFILL_GEMM_PIPELINE": "0"},
        "global_pipe": {},
        "role_selective": {"PREFILL_PIPE_ROLE_SELECTIVE": "1"}}
REPS = {"old_lds2": 1, "global_pipe": 2, "role_selective": 2}   # repeat the decisive arms for run-to-run spread

def run_arm(env_extra, fingerprint=False):
  env = {**os.environ, "DEV": "AMD", "PYTHONPATH": str(ROOT), "PREFILL_V2": "1",
         "PIPE_SPS": "0,1024,2048,4096,7680", "PIPE_LS": "512,1024,2048,4096,8192", **env_extra}
  if fingerprint: env["PIPE_FINGERPRINT"] = "1"
  out = subprocess.run([sys.executable, "extra/qk/prefill_pipe_hardening.py"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=1800).stdout
  ln = [l.strip().lstrip("@") for l in out.splitlines() if l.strip().lstrip("@").startswith("{")]
  if not ln: raise RuntimeError("arm failed: " + out[-1500:])
  return json.loads(ln[-1])

def main():
  runs = {a: [run_arm(ARMS[a]) for _ in range(REPS[a])] for a in ARMS}
  # whole-prefill tok/s: take MAX across reps (min chunk time -> max tok/s) per ctx
  def best(a, L):
    return max(r["whole_prefill_tok_s"][str(L)] for r in runs[a])
  def spread(a, L):
    vs = [r["whole_prefill_tok_s"][str(L)] for r in runs[a]]
    return round(100*(max(vs)-min(vs))/max(vs), 1) if len(vs) > 1 else 0.0
  ctxs = [int(x) for x in runs["global_pipe"][0]["whole_prefill_tok_s"]]
  table = {}
  for L in ctxs:
    o, g, r = best("old_lds2", L), best("global_pipe", L), best("role_selective", L)
    table[str(L)] = {"old_lds2": round(o), "global_pipe": round(g), "role_selective": round(r),
                     "rs_vs_old_pct": round(100*(r-o)/o, 1), "rs_vs_global_pct": round(100*(r-g)/g, 1),
                     "global_spread_pct": spread("global_pipe", L), "rs_spread_pct": spread("role_selective", L)}
  # correctness: m.logits fingerprint role_selective vs global (global==old_lds2 already proven by hardening H2)
  fp = {a: run_arm(ARMS[a], fingerprint=True).get("fingerprint") for a in ("global_pipe", "role_selective")}
  equivalent = fp["role_selective"] == fp["global_pipe"] and fp["role_selective"] is not None
  # verdict: rs vs global, accounting for noise (max observed spread as the bar)
  noise_bar = max(max(table[str(L)]["global_spread_pct"], table[str(L)]["rs_spread_pct"]) for L in ctxs)
  rs_deltas = [table[str(L)]["rs_vs_global_pct"] for L in ctxs]
  best_rs_vs_g = max(rs_deltas); worst_rs_vs_g = min(rs_deltas)
  if not equivalent:
    verdict = "ROLE_SELECTIVE_BLOCKED_CORRECTNESS"
  elif worst_rs_vs_g < -max(2.0, noise_bar):
    verdict = "ROLE_SELECTIVE_WORSE"
  elif best_rs_vs_g > max(2.0, noise_bar) and worst_rs_vs_g > -2.0:
    verdict = "ROLE_SELECTIVE_PASS_BEATS_GLOBAL"
  else:
    verdict = "ROLE_SELECTIVE_EQUIVALENT_TO_GLOBAL"
  rec = {"verdict": verdict, "table": table, "noise_bar_pct": noise_bar, "fingerprints": fp,
         "correct_equivalent": equivalent, "rs_vs_global_by_ctx": {str(L): table[str(L)]["rs_vs_global_pct"] for L in ctxs},
         "recommendation": {
           "ROLE_SELECTIVE_PASS_BEATS_GLOBAL": "promote role-selective as the new default (beats global above noise, correct, no regression).",
           "ROLE_SELECTIVE_EQUIVALENT_TO_GLOBAL": "KEEP global pipe (simpler, already shipped). role-selective delta is within run noise -- not worth the per-role complexity.",
           "ROLE_SELECTIVE_WORSE": "keep global pipe; role-selective regresses (per-role split overhead outweighs the gate_up recovery).",
           "ROLE_SELECTIVE_BLOCKED_CORRECTNESS": "blocked: role-selective output not equivalent."}[verdict]}
  json.dump(rec, open(OUT/"latest.json", "w"), indent=2)
  json.dump(table, open(OUT/"wd_table.json", "w"), indent=2)
  (OUT/"summary.md").write_text(
    f"# Role-selective prefill pipe vs global\n\n**Verdict:** {verdict}\n\n{rec['recommendation']}\n\n"
    f"noise bar (max run spread) = {noise_bar}%; correctness equivalent = {equivalent}\n\n"
    "| ctx | old_lds2 | global_pipe | role_selective | rs vs old | rs vs global |\n|---|---|---|---|---|---|\n" +
    "\n".join(f"| {L} | {table[str(L)]['old_lds2']} | {table[str(L)]['global_pipe']} | {table[str(L)]['role_selective']} | "
              f"+{table[str(L)]['rs_vs_old_pct']}% | {table[str(L)]['rs_vs_global_pct']:+}% |" for L in ctxs))
  print(json.dumps(rec, indent=2)); print("\nROLE_SELECTIVE", verdict)
  return rec

if __name__ == "__main__": main()
