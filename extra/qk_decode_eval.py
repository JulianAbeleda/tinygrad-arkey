#!/usr/bin/env python3
"""Decode evaluation harness — the project's first machine-search evaluator (infrastructure, NOT kernel work).

Turns a registered decode candidate into a machine-readable verdict by running the lifecycle ladder
(correctness -> local A/B -> whole-decode W==D -> policy) as ISOLATED SUBPROCESSES (env set per child only;
`getenv` is @cache'd so a same-process loop would leak/cache), wrapping the existing benchmark scripts. It NEVER
edits tinygrad/ or changes model defaults; it only MEASURES and classifies.

Measurement authority (do not mix): clean W==D (PROFILE off, AUTO clock) = promotion authority; clock-pinned local
= diagnostic only; PROFILE timestamps = attribution only. Promotion is reported, never applied.

CLI:
  python extra/qk_decode_eval.py --list
  python extra/qk_decode_eval.py --candidate flash_l_64 [--dry-run] [--repeats N] [--out DIR]
  python extra/qk_decode_eval.py --suite historical [--out DIR]
  python extra/qk_decode_eval.py --validate bench/qk-decode-eval/runs/<file>.json

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --suite historical
"""
from __future__ import annotations
import argparse, json, os, pathlib, shutil, statistics, subprocess, sys, time
from typing import Any
from extra.qk_modes import Verdict, VERDICTS  # verdict states SSOT

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = ROOT / "bench/qk-decode-eval/candidates.json"
SCHEMA = ROOT / "bench/qk-decode-eval/schema.json"
RUNS = ROOT / "bench/qk-decode-eval/runs"
SUMMARIES = ROOT / "bench/qk-decode-eval/summaries"
WD_RESULT = ROOT / "bench/qk-decode-runtime-overhead/result.json"  # fixed path the W==D script overwrites
HEARTBEAT = ROOT / "bench/qk-decode-eval/heartbeat.json"
# ---- provenance + child-env + model default: single source of truth is extra/qk_harness_contract.py (centralized) -
from extra.qk_harness_contract import git_commit, dirty_tree, perf_state, hardware, child_env, DEFAULT_MODEL  # noqa: E402
MODEL = os.environ.get("QK_MODEL", DEFAULT_MODEL)
def now_ts(): return time.strftime("%Y%m%dT%H%M%S")
def now_date(): return time.strftime("%Y-%m-%d")

# ---- runners (each spawns a subprocess, returns parsed numbers + the exact command) -----------------------------
def run_wd(env: dict, repeats: int) -> dict:
  """N repeats of the clean W==D script with `env`; returns per-ctx tok_s_W samples + median/band. AUTO clock."""
  cmd = [sys.executable, "extra/qk_decode_runtime_overhead.py"]
  samples: dict[int, list[float]] = {}
  log_dir = ROOT / "bench/qk-decode-eval/wd-logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  for rep in range(repeats):
    started = time.time()
    log_path = log_dir / f"{now_ts()}-wd-repeat{rep + 1}.log"
    if WD_RESULT.exists(): WD_RESULT.unlink()
    log_f = log_path.open("w")
    p = subprocess.Popen(cmd, cwd=ROOT, env=child_env(env), text=True, stdout=log_f, stderr=subprocess.STDOUT)
    last_hb, out = 0.0, ""
    timeout_s, timed_out = int(os.environ.get("QK_DECODE_EVAL_WD_TIMEOUT_S", "1800")), False
    try:
      while p.poll() is None:
        time.sleep(5)
        elapsed = time.time() - started
        if elapsed - last_hb >= 30:
          last_hb = elapsed
          hb = {"status": "running", "pid": p.pid, "elapsed_s": round(elapsed, 1), "repeat": rep + 1,
                "repeats": repeats, "command": " ".join(cmd), "env": env, "result_json": str(WD_RESULT),
                "output_log": str(log_path)}
          HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
          HEARTBEAT.write_text(json.dumps(hb, indent=2, sort_keys=True) + "\n")
          print(f"[wd heartbeat] pid={p.pid} repeat={rep + 1}/{repeats} elapsed={elapsed:.0f}s env={env}", file=sys.stderr, flush=True)
        if elapsed >= timeout_s:
          timed_out = True
          p.terminate()
          try: p.wait(timeout=10)
          except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
          break
      log_f.close()
      out = log_path.read_text(errors="replace")
    except KeyboardInterrupt:
      p.terminate()
      log_f.close()
      hb = {"status": "interrupted", "pid": p.pid, "elapsed_s": round(time.time() - started, 1),
            "repeat": rep + 1, "repeats": repeats, "command": " ".join(cmd), "env": env,
            "result_json": str(WD_RESULT), "output_log": str(log_path)}
      HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
      HEARTBEAT.write_text(json.dumps(hb, indent=2, sort_keys=True) + "\n")
      raise
    finally:
      if not log_f.closed: log_f.close()
    hb = {"status": "timeout" if timed_out else "completed", "pid": p.pid, "elapsed_s": round(time.time() - started, 1),
          "repeat": rep + 1, "repeats": repeats, "returncode": p.returncode, "command": " ".join(cmd),
          "env": env, "result_json": str(WD_RESULT), "output_log": str(log_path), "timeout_s": timeout_s}
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(json.dumps(hb, indent=2, sort_keys=True) + "\n")
    if not WD_RESULT.exists():
      raise RuntimeError(f"W==D run {'timed out' if timed_out else 'failed'} without result.json: {(out or '')[-500:]}")
    rows = json.loads(WD_RESULT.read_text())["rows"]
    expected_ctxs = [128, 512, 1024, 4096]
    if sorted(int(r["ctx"]) for r in rows) != expected_ctxs:
      raise RuntimeError(f"W==D incomplete rows: got {[r.get('ctx') for r in rows]}, expected {expected_ctxs}; {(out or '')[-500:]}")
    if "@@DONE@@" not in (out or ""):
      print(f"[wd warning] pid={p.pid} wrote complete result without @@DONE@@; accepting complete result_json={WD_RESULT}", file=sys.stderr, flush=True)
    for r in rows: samples.setdefault(int(r["ctx"]), []).append(float(r["tok_s_W"]))
  out = {"command": "DEV=AMD JIT=1 " + " ".join(f"{k}={v}" for k, v in env.items()) + f" python3 {cmd[1]} (x{repeats})",
         "repeats": repeats, "samples": samples, "median": {}, "band_pct": {}}
  for ctx, xs in samples.items():
    med = statistics.median(xs); spread = (max(xs) - min(xs)) / med * 100 if med else 0.0
    out["median"][ctx] = round(med, 2); out["band_pct"][ctx] = round(spread, 2)
    out.setdefault("min", {})[ctx] = round(min(xs), 2); out.setdefault("max", {})[ctx] = round(max(xs), 2)
    out.setdefault("mad", {})[ctx] = round(statistics.median([abs(x - med) for x in xs]), 3)
  return out

def run_q8_audit() -> dict:
  # The opt-in q8 speedup uses the CLOCK-CONTROLLED manual_peak lane: the audit measures baseline and q8 in
  # SEPARATE child processes, so the auto lane is clock-confounded at the ~1.06x signal level (ctx512 can read
  # q8<baseline purely from per-process clock variance). manual_peak isolates the q8 effect (its designed purpose).
  out_json = ROOT / "bench/qk-decode-eval/_q8_audit.json"
  cmd = [sys.executable, "extra/qk_decode_q8_model_route_timing_audit.py", "--modes", "baseline,q8",
         "--lanes", "auto,manual_peak", "--ckpts", "512", "1024", "--nmeas", "20", "--warmups", "8", "--out", str(out_json)]
  p = subprocess.run(cmd, cwd=ROOT, env=child_env({}), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if not out_json.exists(): raise RuntimeError(f"q8 audit failed: {(p.stdout or '')[-600:]}")
  d = json.loads(out_json.read_text()); summ = d.get("summary", {})
  spd_mp = {r["ctx"]: r["speedup_W"] for r in d.get("rows", []) if r.get("lane") == "manual_peak"}
  spd_auto = {r["ctx"]: r["speedup_W"] for r in d.get("rows", []) if r.get("lane") == "auto"}
  return {"command": "DEV=AMD JIT=1 " + " ".join(cmd[1:]),
          "speedup_W_per_ctx": spd_mp, "speedup_W_per_ctx_auto": spd_auto,
          "median_speedup_W": summ.get("manual_peak", {}).get("median_speedup_W"),       # controlled lane = opt-in authority
          "median_speedup_W_auto": summ.get("auto", {}).get("median_speedup_W"),
          "q8_median_tok_s_W": summ.get("manual_peak", {}).get("q8_median_tok_s_W"), "verdict": d.get("verdict"),
          "lane": "manual_peak (clock-controlled; isolates q8 from auto-clock separate-process confound)"}

def run_ab_script(script: str, result_json: str) -> dict:
  cmd = [sys.executable, script]
  p = subprocess.run(cmd, cwd=ROOT, env=child_env({}), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  rj = ROOT / result_json
  if not rj.exists(): raise RuntimeError(f"A/B script failed: {(p.stdout or '')[-500:]}")
  d = json.loads(rj.read_text())
  # both split-sweep and fused styles: pull ctx1024 best speedup + first_gate_pass + max err
  best, err = None, 0.0
  for row in d.get("results", d.get("rows", [])):
    if row.get("ctx") == 1024:
      best = row.get("best_speedup_vs_coop") or row.get("fused_lds_speedup_vs_coop")
      for s in row.get("splits", []): err = max(err, s.get("err", 0.0))
      err = max(err, row.get("lds_err", 0.0))
  # CONTRACT ENFORCEMENT (Harnesses Are Performance Primitives Too): audit the consumed child artifact against the
  # 13-field evaluator contract and surface any gap, so a non-conforming ab_script harness is flagged centrally.
  try:
    from extra.qk_harness_contract import contract_audit
    child_contract = contract_audit(d)
  except Exception as e:
    child_contract = {"conformance": "UNKNOWN", "missing": [f"audit_error:{e}"], "present": [], "n_present": 0, "n_total": 13}
  return {"command": "DEV=AMD JIT=1 " + " ".join(cmd[1:]), "first_gate_pass": d.get("first_gate_pass"),
          "speedup_ctx1024": best, "max_err": err, "phase": d.get("phase"), "child_contract": child_contract}

def run_route_gate_script(script: str, result_json: str, env: dict) -> dict:
  cmd = [sys.executable, script]
  p = subprocess.run(cmd, cwd=ROOT, env=child_env(env), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  rj = ROOT / result_json
  if not rj.exists(): raise RuntimeError(f"route gate script failed before artifact: {(p.stdout or '')[-800:]}")
  d = json.loads(rj.read_text())
  return {"command": "DEV=AMD JIT=1 " + " ".join(f"{k}={v}" for k, v in env.items()) + " " + " ".join(cmd[1:]),
          "artifact_path": result_json, "returncode": p.returncode, "passed": p.returncode == 0,
          "verdict": d.get("verdict"), "output_tail": (p.stdout or "")[-1200:], "artifact_json": d}

def run_flash_l_local() -> dict:
  """Self-spawned child: flash_decode_attention L=64 vs L=128 at decode shape, clock-pinned, byte-exact vs numpy."""
  cmd = [sys.executable, "extra/qk_decode_eval.py", "--_child", "flash_l_local"]
  p = subprocess.run(cmd, cwd=ROOT, env=child_env({}), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  line = next((l for l in (p.stdout or "").splitlines() if l.startswith("@@FLASHL@@")), None)
  if not line: raise RuntimeError(f"flash_l local A/B failed: {(p.stdout or '')[-600:]}")
  d = json.loads(line[len("@@FLASHL@@"):])
  d["command"] = "DEV=AMD JIT=1 python3 extra/qk_decode_eval.py --_child flash_l_local (clock-pinned diagnostic)"
  return d

def _child_flash_l_local() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, TinyJit
  from tinygrad.uop.ops import UOp
  from extra.qk_flash_decode import flash_decode_attention, FLASH_DECODE_DEFAULT_VARIANT
  from extra.qk_clock_pin import pinned_peak
  dev = Device["AMD"]; Hd, Hq, Hkv, MAXC = 128, 32, 8, 4608; G = Hq // Hkv; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16); k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); qn, kn, vn = Tensor(q), Tensor(k), Tensor(v); Tc = 1024
  ref = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kv = h // G; sc = (q[h].astype(np.float32) @ k[kv, :Tc].astype(np.float32).T) / np.sqrt(Hd)
    pw = np.exp(sc - sc.max()); pw /= pw.sum(); ref[h] = pw @ v[kv, :Tc].astype(np.float32)
  def tfn(fn, n=150):
    dev.synchronize(); ts = []
    for _ in range(n):
      t0 = time.perf_counter(); fn(); dev.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6
  res = {}
  with pinned_peak():
    time.sleep(0.4)
    for L in (128, 64):
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      j = TinyJit(lambda spb, LL=L: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, LL, variant=FLASH_DECODE_DEFAULT_VARIANT).realize())
      for _ in range(8): j(vsp.bind(Tc - 1))
      err = float(np.abs(j(vsp.bind(Tc - 1)).numpy() - ref).max()); res[L] = (tfn(lambda: j(vsp.bind(Tc - 1))), err)
  spd = res[128][0] / res[64][0] if res[64][0] else 0.0
  print("@@FLASHL@@" + json.dumps({"speedup_ctx1024": round(spd, 3), "us_L128": round(res[128][0], 1),
        "us_L64": round(res[64][0], 1), "max_err": round(max(res[128][1], res[64][1]), 4),
        "authority": "clock-pinned local diagnostic (NOT promotion)"}))
  return 0

# ---- verdict logic ----------------------------------------------------------------------------------------------
def classify(cand: dict, th: dict, corr: dict, local: dict, wd: dict) -> tuple[Verdict, str]:
  if cand["family"] == "binding_selftest":
    return Verdict.SELFTEST_PASS, "evaluator-binding plumbing selftest; no performance measured (not a perf pass)"
  if cand["family"] == "reference_oracle":
    if local["checked"] and local["passed"]:
      return Verdict.PASS_ORACLE_LOCAL_AB, f"reference oracle beats comparator local A/B ({local.get('speedup_ctx1024')}x); a TARGET that informs codegen; NON-promotable (vendored reference, not a tinygrad primitive)"
    return Verdict.FAIL_ORACLE_LOCAL_AB, f"reference oracle does not beat comparator local A/B ({local.get('speedup_ctx1024')}x)"
  if corr["checked"] and corr["passed"] is False:
    return Verdict.FAIL_CORRECTNESS, f"correctness {corr.get('metric')}={corr.get('value')} > {corr.get('threshold')}"
  if local["checked"] and local["passed"] is False:
    return Verdict.FAIL_LOCAL_AB, f"local speedup {local.get('speedup_ctx1024')} < {local.get('threshold_min')}"
  if wd["checked"]:
    if wd.get("repro_band_ok") is False:
      return Verdict.NEEDS_GPU_STATE_TOOLING, f"W==D repro band {wd.get('repro_band_pct')} > {th['repro_band_max_pct']}% (auto-clock variance exceeds the promotion margin)"
    if wd.get("promotion_gate_passed"):
      return Verdict.PASS_PROMOTE, f"W==D delta {wd.get('delta_pct')} clears the promotion gate"
    if local["checked"] and local["passed"]:
      return Verdict.LOCAL_PASS_WD_FAIL, f"local passed ({local.get('speedup_ctx1024')}x) but W==D {wd.get('delta_pct')} below the >=5%@1024/>=7%@4096 gate"
  if cand["family"] == "q8_route":
    spd = (wd.get("median_speedup_W") or 0)
    if spd >= th["opt_in_min_speedup"] and corr.get("passed"):
      return Verdict.PASS_OPT_IN, f"q8 whole-decode {spd:.3f}x (opt-in, default-off) + dNLL {corr.get('value')} <= {th['dnll_max']}"
  if cand["family"] == "baseline":
    return Verdict.REST, "baseline curve; not a promotion candidate"
  return Verdict.REST, "no rung cleared a promotion gate"

# ---- evaluate one candidate -------------------------------------------------------------------------------------
def evaluate(cand: dict, reg: dict, cache: dict, repeats_override: int | None, dry: bool, dirty_tree_at_start: bool | None = None) -> dict:
  th = {**reg["thresholds_default"]}
  res = {"schema": "decode_eval_run_v1", "candidate_id": cand["id"], "candidate_family": cand["family"],
         "candidate_description": cand["description"], "date": now_date(), "git_commit": git_commit(),
         "dirty_tree": dirty_tree() if dirty_tree_at_start is None else dirty_tree_at_start, "hardware": hardware(), "perf_state_before": perf_state(),
         "clock_pin_mode": "auto", "env": cand.get("env", {}), "commands": [], "contexts": cand["contexts"],
         "repeats": 0, "warmups": 8, "thresholds": th, "verdict_expected": cand.get("historical_expected_verdict"),
         "linked_ledger_entry": cand.get("compare_artifact"), "default_behavior_changed": False,
         "source_files": ["extra/qk_decode_eval.py", "bench/qk-decode-eval/candidates.json"], "notes": [],
           "correctness": {"checked": False, "passed": None, "metric": None, "value": None, "threshold": None, "note": None},
           "local_ab": {"checked": False, "passed": None, "speedup_ctx1024": None, "threshold_min": None, "authority": None, "note": None},
           "route_gate": {"checked": False, "passed": None, "artifact_path": None, "artifact_verdict": None, "returncode": None, "note": None},
           "wd": {"checked": False, "authority": "clean W==D PROFILE-off AUTO-clock (promotion authority)"}}
  if dry:
    res["verdict"] = Verdict.REST; res["stop_reason"] = "dry-run (no GPU)"; res["perf_state_after"] = perf_state()
    res["verdict_matches_expected"] = None
    res["commands"] = [f"[dry] would run rungs: {[r['rung'] for r in cand['rungs']]}"]
    return res
  for rung in cand["rungs"]:
    r, runner = rung["rung"], rung["runner"]
    if r == "local_ab" and runner == "flash_l_local":
      d = run_flash_l_local(); res["commands"].append(d["command"]); res["clock_pin_mode"] = "manual_peak_diagnostic_only" if res["clock_pin_mode"] == "auto" else "mixed"
      res["local_ab"] = {"checked": True, "speedup_ctx1024": d["speedup_ctx1024"], "threshold_min": th["local_min_speedup"],
                         "passed": d["speedup_ctx1024"] >= th["local_min_speedup"], "authority": d["authority"], "note": f"L64 {d['us_L64']}us vs L128 {d['us_L128']}us"}
      if cand.get("correctness_req") == "byte_exact":
        res["correctness"] = {"checked": True, "metric": "max_err", "value": d["max_err"], "threshold": th["correctness_tol"], "passed": d["max_err"] <= th["correctness_tol"], "note": "byte-exact vs numpy ref"}
    elif r == "local_ab" and runner == "ab_script":
      d = run_ab_script(rung["script"], rung["result_json"]); res["commands"].append(d["command"]); res["source_files"].append(rung["script"])
      res["local_ab"] = {"checked": True, "speedup_ctx1024": d["speedup_ctx1024"], "threshold_min": th["local_min_speedup"],
                         "passed": bool(d.get("first_gate_pass")), "authority": "clock-pinned local diagnostic", "note": d.get("phase"),
                         "child_artifact_contract": d.get("child_contract")}
      cc = d.get("child_contract") or {}
      if cc.get("conformance") not in (None, "CONFORMS"):
        res["notes"].append(f"HARNESS-CONTRACT: child artifact {rung['result_json']} is {cc.get('conformance')} "
                            f"({cc.get('n_present')}/{cc.get('n_total')} fields); missing: {', '.join(cc.get('missing', [])[:6])}")
      if cand.get("correctness_req") == "byte_exact":
        res["correctness"] = {"checked": True, "metric": "max_err", "value": d["max_err"], "threshold": th["correctness_tol"], "passed": d["max_err"] <= th["correctness_tol"], "note": "byte-exact vs numpy ref"}
    elif r == "route_gate" and runner == "script":
      d = run_route_gate_script(rung["script"], rung["result_json"], cand.get("env", {})); res["commands"].append(d["command"])
      res["source_files"].append(rung["script"])
      res["route_gate"] = {"checked": True, "passed": bool(d["passed"]), "artifact_path": rung["result_json"],
                           "artifact_verdict": d.get("verdict"), "returncode": d.get("returncode"),
                           "note": "route gate passed; W==D is allowed to run" if d["passed"] else "route gate failed; W==D blocked"}
      if cand.get("correctness_req") in ("route_gate", "route_gate_and_score_numeric", "lifecycle_gate_and_route_gate"):
        res["correctness"] = {"checked": True, "metric": "route_gate", "value": 1.0 if d["passed"] else 0.0,
                              "threshold": 1.0, "passed": bool(d["passed"]), "note": d.get("verdict")}
      if not d["passed"]:
        res["notes"].append(f"route_gate blocked W==D: {rung['result_json']} verdict={d.get('verdict')} returncode={d.get('returncode')}")
        break
    elif r == "wd" and runner == "runtime_overhead":
      reps = repeats_override or rung.get("repeats", 3); res["repeats"] = reps
      wd = run_wd(cand.get("env", {}), reps); res["commands"].append(wd["command"])
      cur = {"checked": True, "authority": res["wd"]["authority"], "per_ctx": wd["median"], "min": wd.get("min"),
             "max": wd.get("max"), "mad": wd.get("mad"), "repro_band_pct": wd["band_pct"]}
      cur["repro_band_ok"] = all(b <= th["repro_band_max_pct"] for b in wd["band_pct"].values())
      if rung.get("is_baseline"):
        cache["baseline_wd"] = wd["median"]; cur["note"] = "baseline curve + reproducibility band (the falsifier)"
      else:
        base = cache.get("baseline_wd")
        if base is None:  # run baseline inline if not cached
          b = run_wd({}, max(2, reps // 2)); base = b["median"]; res["commands"].append(b["command"] + " [baseline]")
        cur["baseline_per_ctx"] = base
        cur["delta_pct"] = {c: round((wd["median"][c] - base[c]) / base[c] * 100, 2) for c in wd["median"] if c in base}
        d1024 = cur["delta_pct"].get(1024, 0); d4096 = cur["delta_pct"].get(4096, 0); d512 = cur["delta_pct"].get(512, 0)
        cur["promotion_gate_passed"] = (d1024 >= th["wd_min_pct_ctx1024"] or d4096 >= th["wd_min_pct_ctx4096"]) and d512 >= -th["ctx512_regress_max_pct"]
      res["wd"] = cur
    elif r == "wd_q8" and runner == "q8_audit":
      d = run_q8_audit(); res["commands"].append(d["command"]); res["repeats"] = 1
      res["wd"] = {"checked": True, "authority": f"q8 audit baseline-vs-q8 W==D, {d.get('lane','')}", "median_speedup_W": d["median_speedup_W"],
                   "median_speedup_W_auto": d.get("median_speedup_W_auto"), "per_ctx_speedup": d["speedup_W_per_ctx"],
                   "per_ctx_speedup_auto": d.get("speedup_W_per_ctx_auto"), "q8_median_tok_s_W": d["q8_median_tok_s_W"],
                   "promotion_gate_passed": False, "repro_band_ok": True, "note": "opt-in route; not a default promotion; speedup read on the clock-controlled lane"}
      res["clock_pin_mode"] = "manual_peak_diagnostic_only" if res["clock_pin_mode"] == "auto" else "mixed"
      res["notes"].append("q8 opt-in speedup uses the clock-controlled manual_peak lane (auto lane is clock-confounded at the ~1.06x signal: baseline/q8 run in separate processes).")
    elif r == "correctness" and runner == "q8_dnll_historical":
      bn = json.loads((ROOT / rung["baseline_nll"]).read_text())["nll"]; qn = json.loads((ROOT / rung["q8_nll"]).read_text())["nll"]
      dnll = qn - bn; res["correctness"] = {"checked": True, "metric": "dNLL", "value": round(dnll, 5), "threshold": th["dnll_max"], "passed": dnll <= th["dnll_max"], "note": "historical teacher-forced dNLL artifact"}
      res["source_files"] += [rung["baseline_nll"], rung["q8_nll"]]
      res["notes"].append("q8 dNLL is from the historical bench/q8-ffn-handwritten-oracle artifact (the q8 audit script does not compute dNLL).")
    elif r == "selftest" and runner == "binding_selftest":
      # plumbing only: NO GPU, NO benchmark. Validates the binding -> candidate -> decode_eval -> artifact path.
      res["commands"].append(f"[binding selftest: no GPU; binding_template_id={rung.get('binding_template_id')}]")
      res["notes"].append("binding-plumbing selftest; SELFTEST_PASS is not a performance pass")
      res["source_files"].append("bench/qk-decode-eval/binding_templates.json")
    else:
      raise RuntimeError(f"unknown rung/runner pair: rung={r!r} runner={runner!r}")
  v, reason = classify(cand, th, res["correctness"], res["local_ab"], res["wd"])
  res["verdict"], res["stop_reason"] = v, reason
  res["verdict_matches_expected"] = (cand.get("historical_expected_verdict") in (None, v)) or (v == cand.get("historical_expected_verdict"))
  res["perf_state_after"] = perf_state()
  res["notes"].append(cand.get("historical_note", ""))
  return res

# ---- artifacts / CLI --------------------------------------------------------------------------------------------
def validate(path: pathlib.Path) -> int:
  import jsonschema
  schema = json.loads(SCHEMA.read_text())
  try: jsonschema.validate(json.loads(path.read_text()), schema); print(f"VALID: {path}"); return 0
  except jsonschema.ValidationError as e: print(f"INVALID: {path}\n  {e.message}"); return 1

def emit(res: dict, outdir: pathlib.Path) -> pathlib.Path:
  assert res["verdict"] in VERDICTS, f"invalid verdict {res['verdict']!r} not in the Verdict SSOT"  # encode the invariant
  outdir.mkdir(parents=True, exist_ok=True)
  f = outdir / f"{now_ts()}-{res['candidate_id']}.json"; f.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
  return f

def main() -> int:
  ap = argparse.ArgumentParser(description="Decode evaluation harness (measurement only; no defaults/kernels changed)")
  ap.add_argument("--list", action="store_true"); ap.add_argument("--candidate"); ap.add_argument("--suite")
  ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--repeats", type=int)
  ap.add_argument("--out", type=pathlib.Path, default=RUNS); ap.add_argument("--validate", type=pathlib.Path)
  ap.add_argument("--_child")  # internal self-spawn
  args = ap.parse_args()
  if args._child == "flash_l_local": return _child_flash_l_local()
  if args.validate: return validate(args.validate)
  reg = json.loads(REG.read_text())
  by_id = {c["id"]: c for c in reg["candidates"]}
  if args.list:
    print(f"{'id':22}{'family':16}{'expected':22} description")
    for c in reg["candidates"]: print(f"{c['id']:22}{c['family']:16}{str(c.get('historical_expected_verdict')):22}{c['description'][:70]}")
    return 0
  cands = ([by_id[i] for i in reg["suites"][args.suite]] if args.suite else [by_id[args.candidate]] if args.candidate else [])
  if not cands: print("specify --list, --candidate <id>, --suite <name>, or --validate <file>"); return 2
  cache, results = {}, []
  dirty_tree_at_start = dirty_tree()
  for c in cands:
    print(f"=== evaluating {c['id']} ({c['family']}) ===", file=sys.stderr)
    try: res = evaluate(c, reg, cache, args.repeats, args.dry_run, dirty_tree_at_start)
    except Exception as e:
      res = {"schema": "decode_eval_run_v1", "candidate_id": c["id"], "verdict": Verdict.REST, "stop_reason": f"runner error: {str(e)[:200]}",
             "verdict_expected": c.get("historical_expected_verdict"), "verdict_matches_expected": False, "default_behavior_changed": False, "notes": [str(e)[:300]]}
    f = emit(res, args.out); results.append(res)
    rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
    vstr = res['verdict'].value if isinstance(res['verdict'], Verdict) else res['verdict']  # .value: str(enum) is the repr
    print(f"  -> {vstr} (expected {res.get('verdict_expected')}, match={res.get('verdict_matches_expected')}) | {res.get('stop_reason','')[:90]}\n  artifact: {rel}", file=sys.stderr)
  SUMMARIES.mkdir(parents=True, exist_ok=True)
  summ = {"date": now_date(), "git_commit": git_commit(), "rows": [{"candidate": r["candidate_id"], "verdict": r["verdict"],
          "expected": r.get("verdict_expected"), "match": r.get("verdict_matches_expected"),
          "route_gate_verdict": r.get("route_gate", {}).get("artifact_verdict"),
          "wd_repro_band_pct": r.get("wd", {}).get("repro_band_pct"), "stop_reason": r.get("stop_reason")} for r in results]}
  (SUMMARIES / "latest.json").write_text(json.dumps(summ, indent=2) + "\n")
  print(json.dumps(summ, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
