#!/usr/bin/env python3
"""Phase M0a: re-base the candidate metric on device bandwidth and re-audit the 4.x wins.

The 3F-4.x line scored gains on wall-clock q4_eff (~28-35 GB/s, dominated by the ~0.27 ms
launch overhead). This re-runs a representative sample of the 4.x raw_accept schedules with
device timing (device_q4_eff via DEBUG=2), compares each to v1_partial on the DEVICE metric,
expresses everything as a fraction of measured peak HBM bandwidth, and reports whether any
"win" was real on device or wall-clock noise.
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, statistics, subprocess
from typing import Any

DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_ARTIFACT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/metric-audit-m0")
# Measured achievable streaming bandwidth on this GPU (warm a+1 copy, stable ~858-863 GB/s),
# 89% of the RX7900XTX 960 GB/s datasheet peak. The roofline denominator.
PEAK_GBS_MEASURED = 859.0
# A device gain within this band of zero is a tie, not a real improvement.
NOISE_BAND_PCT = 2.0

DEVICE_RE = re.compile(r"q4k_primitive_gemv:.*device_q4_eff=(?P<dev>[0-9.]+) GB/s")
CORRECT_RE = re.compile(r"primitive_gemv_correctness: (?P<status>PASS|FAIL)")

# Modes that realize each 4.x mechanism in q4_k_bench (mirrors qk_semantic_schedule_bench).
MECHANISM_SPEC = {
  "v1_partial":    {"mode": "partial", "opts": ["LOCAL:0:64"], "parts": 1},
  "row_upcast":    {"mode": "partial", "opts": ["LOCAL:0:64", "UPCAST:0:2"], "parts": 1},
  "direct_output": {"mode": "serial",  "opts": ["LOCAL:0:64"], "parts": 1},
}
# Representative sample of the 22 distinct 4.x raw_accept (mechanism, tensor) configs.
RE_AUDIT = [
  ("row_upcast", "blk.1.attn_q.weight"), ("row_upcast", "blk.2.attn_q.weight"),
  ("row_upcast", "blk.6.attn_q.weight"), ("row_upcast", "blk.13.ffn_gate.weight"),
  ("direct_output", "blk.11.attn_q.weight"), ("direct_output", "blk.6.attn_q.weight"),
  ("direct_output", "blk.18.ffn_gate.weight"),
]

def _measure(repo:pathlib.Path, model:pathlib.Path, tensor:str, spec:dict[str, Any], *,
             device:str, iters:int, seed:int, timeout:float, runs:int) -> dict[str, Any]:
  cmd = [".venv/bin/python", "extra/q4_k_bench.py", str(model), "--device", device, "--tensor", tensor,
         "--iters", str(iters), "--format", "text", "--activation", "random", "--seed", str(seed),
         "--primitive", "--primitive-mode", spec["mode"], "--primitive-parts", str(spec["parts"])]
  for opt in spec["opts"]: cmd += ["--primitive-opt", opt]
  env = {**os.environ, "DEV": device, "DEBUG": "2", "PYTHONPATH": "."}
  devs = []
  for _ in range(runs):
    try:
      proc = subprocess.run(cmd, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
      out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
      out, rc = (exc.stdout or "") + "\nTIMEOUT", 124
    c, m = CORRECT_RE.search(out), DEVICE_RE.search(out)
    if rc != 0 or c is None or c["status"] != "PASS" or m is None:
      return {"status": "fail", "tail": "\n".join(out.strip().splitlines()[-4:])}
    devs.append(float(m["dev"]))
  return {"status": "pass", "device_q4_gbs": round(statistics.median(devs), 3)}

def run_audit(repo:pathlib.Path, model:pathlib.Path, artifact:pathlib.Path, *,
              device:str="AMD", iters:int=3, seed:int=1337, timeout:float=180.0, runs:int=3,
              peak_gbs:float=PEAK_GBS_MEASURED) -> dict[str, Any]:
  repo, model = repo.resolve(), model.expanduser().resolve()
  rows, baseline_cache = [], {}
  for mech, tensor in RE_AUDIT:
    if tensor not in baseline_cache:
      baseline_cache[tensor] = _measure(repo, model, tensor, MECHANISM_SPEC["v1_partial"],
                                         device=device, iters=iters, seed=seed, timeout=timeout, runs=runs)
    base, cand = baseline_cache[tensor], _measure(repo, model, tensor, MECHANISM_SPEC[mech],
                                                  device=device, iters=iters, seed=seed, timeout=timeout, runs=runs)
    row = {"mechanism": mech, "tensor": tensor, "baseline_status": base["status"], "candidate_status": cand["status"]}
    if base["status"] == "pass" and cand["status"] == "pass":
      b, c = base["device_q4_gbs"], cand["device_q4_gbs"]
      row.update({
        "baseline_device_gbs": b, "candidate_device_gbs": c,
        "device_gain_pct": round((c / b - 1) * 100, 3),
        "baseline_roofline_pct": round(b / peak_gbs * 100, 2),
        "candidate_roofline_pct": round(c / peak_gbs * 100, 2),
        # A "win" must clear a noise band; +-NOISE_BAND_PCT is a tie, not an improvement.
        "real_device_win": (c / b - 1) * 100 > NOISE_BAND_PCT,
      })
    rows.append(row)

  audited = [r for r in rows if "device_gain_pct" in r]
  real_wins = [r for r in audited if r["real_device_win"]]
  # Baseline roofline fraction per tensor shape (headroom is shape-dependent).
  baseline_roofline = {}
  for r in audited:
    baseline_roofline.setdefault(r["tensor"], r["baseline_roofline_pct"])
  # Re-score the G0 grid on the roofline-relative metric.
  g0_path = repo / "bench/amd-decode-flywheel-proof-20260614/generation-g0/candidates.jsonl"
  g0_best = {}
  if g0_path.exists():
    for line in g0_path.read_text().splitlines():
      r = json.loads(line)
      if r.get("device_q4_gbs_median"):
        g0_best[r["tensor"]] = max(g0_best.get(r["tensor"], 0.0), r["device_q4_gbs_median"])
  g0_roofline = {t: round(g / peak_gbs * 100, 2) for t, g in g0_best.items()}

  median_gain = round(statistics.median([r["device_gain_pct"] for r in audited]), 3) if audited else None
  conclusion = (f"4x_wins_confirmed_noise_no_raw_accept_beats_v1_partial_on_device_by_{NOISE_BAND_PCT:.0f}pct" if not real_wins
                else "some_4x_raw_accepts_are_real_device_wins_re_examine")
  roofline_by_shape = dict(sorted(baseline_roofline.items()))
  summary = {
    "kind": "qk_flywheel_metric_audit_m0a", "phase": "Phase M0a", "conclusion": conclusion,
    "measured_peak_gbs": peak_gbs, "peak_provenance": "warm a+1 streaming copy, this GPU; 89% of 960 datasheet",
    "noise_band_pct": NOISE_BAND_PCT,
    "audited_configs": len(audited), "real_device_wins": len(real_wins),
    "median_device_gain_pct_of_4x_raw_accepts": median_gain,
    "v1_partial_roofline_pct_by_tensor": roofline_by_shape,
    "headroom_x_by_tensor": {t: round(100 / p, 2) for t, p in roofline_by_shape.items()},
    "best_kernel_roofline_pct": g0_roofline,
    "headroom_x_vs_best": {t: round(100 / p, 2) for t, p in g0_roofline.items()},
    "re_audit": rows,
    "metric": "device_q4_eff (DEBUG=2) as a fraction of measured peak HBM bandwidth",
    "note": ("Root cause: qk_semantic_schedule_bench Q4_RESULT_RE captures q4_eff (wall), not "
             "device_q4_eff. The 4.x gains were measured on the overhead-dominated wall metric."),
  }
  artifact.mkdir(parents=True, exist_ok=True)
  (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def main() -> int:
  p = argparse.ArgumentParser(description="Phase M0a metric re-base and 4.x win re-audit")
  p.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  p.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  p.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  p.add_argument("--device", default="AMD")
  p.add_argument("--peak-gbs", type=float, default=PEAK_GBS_MEASURED)
  args = p.parse_args()
  print(json.dumps(run_audit(args.repo, args.model, args.artifact, device=args.device, peak_gbs=args.peak_gbs), indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
