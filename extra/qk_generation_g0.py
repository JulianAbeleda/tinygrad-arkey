#!/usr/bin/env python3
"""Phase G0: search-space headroom probe for the AMD decode flywheel generation track.

Brute-force expand the parametric schedule space (LOCAL / UPCAST / UNROLL / parts and
compositions the hardcoded grid never tries) on live-bearing attn_q tensors, run every
candidate through the existing q4_k_bench primitive correctness + microbench gate, and
ask the precondition question before any model: is there a candidate that beats both the
v1_partial baseline and the best hardcoded mechanism? No model here -- this is the
deterministic baseline the later model-guided search (G1) must beat.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, statistics
from typing import Any

DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_ARTIFACT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/generation-g0")
DEFAULT_TENSORS = ("blk.20.attn_q.weight", "blk.21.attn_q.weight")

from extra.q4k_bench_metrics import CORRECT_RE, DEVICE_RE

# The current runtime (v1_partial) and the four hardcoded mechanisms are the bar to beat.
BASELINE = {"label": "v1_partial", "opts": ["LOCAL:0:64"], "parts": 1, "group": "baseline"}
HARDCODED = [
  {"label": "row_upcast2", "opts": ["LOCAL:0:64", "UPCAST:0:2"], "parts": 1, "group": "hardcoded"},
  {"label": "reduce_unroll4", "opts": ["LOCAL:0:64", "UNROLL:2:4"], "parts": 1, "group": "hardcoded"},
  {"label": "two_dim_local4", "opts": ["LOCAL:0:32", "LOCAL:1:4"], "parts": 1, "group": "hardcoded"},
]

def _expanded() -> list[dict[str, Any]]:
  cands = []
  # single-axis LOCAL sweep beyond {32,64}
  for lx in (16, 32, 64, 128, 256):
    cands.append({"label": f"local{lx}", "opts": [f"LOCAL:0:{lx}"], "parts": 1})
  # UPCAST arg sweep beyond the hardcoded :2
  for lx in (32, 64):
    for u in (2, 4, 8):
      cands.append({"label": f"local{lx}_upcast{u}", "opts": [f"LOCAL:0:{lx}", f"UPCAST:0:{u}"], "parts": 1})
  # UNROLL arg sweep beyond the hardcoded :4
  for lx in (32, 64):
    for u in (2, 4, 8, 16):
      cands.append({"label": f"local{lx}_unroll{u}", "opts": [f"LOCAL:0:{lx}", f"UNROLL:2:{u}"], "parts": 1})
  # two-dim LOCAL beyond the hardcoded :4
  for lx in (32, 64):
    for l1 in (2, 4):
      cands.append({"label": f"local{lx}_l1_{l1}", "opts": [f"LOCAL:0:{lx}", f"LOCAL:1:{l1}"], "parts": 1})
  # compositions the grid never tries
  cands.append({"label": "local64_upcast2_unroll2", "opts": ["LOCAL:0:64", "UPCAST:0:2", "UNROLL:2:2"], "parts": 1})
  cands.append({"label": "local64_upcast2_unroll4", "opts": ["LOCAL:0:64", "UPCAST:0:2", "UNROLL:2:4"], "parts": 1})
  cands.append({"label": "local32_upcast2_l1_4", "opts": ["LOCAL:0:32", "UPCAST:0:2", "LOCAL:1:4"], "parts": 1})
  # parts sweep
  for p in (2, 4):
    cands.append({"label": f"local64_parts{p}", "opts": ["LOCAL:0:64"], "parts": p})
  # Drop any combo that coincides with the baseline or a hardcoded mechanism, so the
  # expanded group genuinely means "beyond the grid" (e.g. local64_upcast2 == row_upcast2).
  known = {(frozenset(c["opts"]), c["parts"]) for c in [BASELINE] + HARDCODED}
  cands = [c for c in cands if (frozenset(c["opts"]), c["parts"]) not in known]
  for c in cands: c["group"] = "expanded"
  return cands

def _classify(rc:int, out:str) -> str:
  if "KernelOptError" in out: return "illegal_opt"
  if "CompileError" in out or "compile failed" in out: return "compile_error"
  if rc == 0: return "pass"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"

def _run_candidate(repo:pathlib.Path, model:pathlib.Path, tensor:str, cand:dict[str, Any], *,
                   device:str, iters:int, seed:int, timeout:float, runs:int) -> dict[str, Any]:
  cmd = [".venv/bin/python", "extra/q4_k_bench.py", str(model), "--device", device, "--tensor", tensor,
         "--iters", str(iters), "--format", "text", "--activation", "random", "--seed", str(seed),
         "--primitive", "--primitive-mode", "partial", "--primitive-parts", str(cand["parts"])]
  for opt in cand["opts"]: cmd += ["--primitive-opt", opt]
  env = {**os.environ, "DEV": device, "DEBUG": "2", "PYTHONPATH": "."}
  devs, status, last = [], "pass", ""
  for _ in range(runs):
    try:
      proc = subprocess.run(cmd, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
      out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
      out, rc = (exc.stdout or "") + "\nTIMEOUT", 124
    st = _classify(rc, out)
    if st != "pass":
      status, last = st, "\n".join(out.strip().splitlines()[-4:])
      break
    m, c = DEVICE_RE.search(out), CORRECT_RE.search(out)
    if c is None or c["status"] != "PASS": status, last = "wrong", "\n".join(out.strip().splitlines()[-4:]); break
    if m is None: status, last = "no_device_timing", "\n".join(out.strip().splitlines()[-4:]); break
    devs.append(float(m["dev"]))
  row = {"tensor": tensor, "label": cand["label"], "group": cand["group"], "opts": cand["opts"],
         "parts": cand["parts"], "status": status, "command": " ".join(cmd[1:])}
  if devs: row["device_q4_gbs_median"] = round(statistics.median(devs), 3)
  else: row["failure_tail"] = last
  return row

def run_g0(repo:pathlib.Path, model:pathlib.Path, tensors:tuple[str, ...], artifact:pathlib.Path, *,
           device:str="AMD", iters:int=3, seed:int=1337, timeout:float=180.0, runs:int=3) -> dict[str, Any]:
  repo, model = repo.resolve(), model.expanduser().resolve()
  candidates = [BASELINE] + HARDCODED + _expanded()
  rows = []
  for tensor in tensors:
    for cand in candidates:
      rows.append(_run_candidate(repo, model, tensor, cand, device=device, iters=iters, seed=seed, timeout=timeout, runs=runs))

  per_tensor = {}
  any_headroom = False
  best_gain_vs_baseline = best_gain_vs_hardcoded = None
  for tensor in tensors:
    trows = [r for r in rows if r["tensor"] == tensor]
    def gbs(group_or_label, by_label=False):
      vals = [r["device_q4_gbs_median"] for r in trows if r.get("device_q4_gbs_median") is not None and
              (r["label"] == group_or_label if by_label else r["group"] == group_or_label)]
      return max(vals) if vals else None
    baseline = gbs("v1_partial", by_label=True)
    hardcoded_best = gbs("hardcoded")
    expanded = [r for r in trows if r["group"] == "expanded" and r.get("device_q4_gbs_median") is not None]
    expanded_best = max((r["device_q4_gbs_median"] for r in expanded), default=None)
    winner = max(expanded, key=lambda r: r["device_q4_gbs_median"], default=None)
    bar = max(x for x in (baseline, hardcoded_best) if x is not None) if (baseline or hardcoded_best) else None
    has_headroom = bool(winner and bar and winner["device_q4_gbs_median"] > bar)
    any_headroom = any_headroom or has_headroom
    entry = {
      "baseline_gbs": baseline, "hardcoded_best_gbs": hardcoded_best, "expanded_best_gbs": expanded_best,
      "expanded_best_label": winner["label"] if winner else None,
      "expanded_best_opts": winner["opts"] if winner else None,
      "headroom_over_bar": has_headroom,
      "gain_vs_baseline_pct": round((expanded_best / baseline - 1) * 100, 3) if (expanded_best and baseline) else None,
      "gain_vs_hardcoded_pct": round((expanded_best / hardcoded_best - 1) * 100, 3) if (expanded_best and hardcoded_best) else None,
    }
    per_tensor[tensor] = entry
    if entry["gain_vs_baseline_pct"] is not None:
      best_gain_vs_baseline = entry["gain_vs_baseline_pct"] if best_gain_vs_baseline is None else max(best_gain_vs_baseline, entry["gain_vs_baseline_pct"])
    if entry["gain_vs_hardcoded_pct"] is not None:
      best_gain_vs_hardcoded = entry["gain_vs_hardcoded_pct"] if best_gain_vs_hardcoded is None else max(best_gain_vs_hardcoded, entry["gain_vs_hardcoded_pct"])

  conclusion = ("parametric_headroom_found_proceed_to_g1_model_guided_search" if any_headroom
                else "no_parametric_headroom_hardcoded_grid_near_optimal_stop_or_escalate_to_g2")
  summary = {
    "kind": "qk_flywheel_generation_g0_headroom", "phase": "Phase G0", "conclusion": conclusion,
    "tensors": list(tensors), "candidates_per_tensor": len(candidates),
    "gpu_experiments": sum(1 for r in rows) * runs, "distinct_candidate_runs": len(rows),
    "any_parametric_headroom": any_headroom,
    "best_gain_vs_baseline_pct": best_gain_vs_baseline, "best_gain_vs_hardcoded_pct": best_gain_vs_hardcoded,
    "per_tensor": per_tensor,
    "metric": "median device_q4_eff GB/s over repeated runs, correctness-gated",
    "note": ("Deterministic brute-force search; the GPU cost here is the baseline G1 model-guided "
             "search must beat on sample-efficiency. No model involved in G0."),
  }
  artifact.mkdir(parents=True, exist_ok=True)
  with (artifact / "candidates.jsonl").open("w") as f:
    for r in rows: f.write(json.dumps(r, sort_keys=True) + "\n")
  (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Phase G0 search-space headroom probe")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--tensor", action="append", default=None, help="repeatable; defaults to two fresh attn_q")
  parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--runs", type=int, default=3)
  args = parser.parse_args()
  tensors = tuple(args.tensor) if args.tensor else DEFAULT_TENSORS
  summary = run_g0(args.repo, args.model, tensors, args.artifact, device=args.device, iters=args.iters, runs=args.runs)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
