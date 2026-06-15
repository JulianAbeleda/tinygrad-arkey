#!/usr/bin/env python3
"""Phase G0': device-metric search over existing kernel strategies and occupancy.

M0 showed the bottleneck is Q4_K dequant compute + occupancy, not bandwidth/load-width, and
that the opt knobs (UPCAST/UNROLL/LOCAL) don't touch it. But q4_k_bench has several primitive
MODES that are genuinely different dequant/load kernels. This sweeps modes x occupancy
(parts) on the device metric (device_q4_eff as a fraction of measured peak), correctness-gated,
to test whether any EXISTING kernel beats v1_partial before committing to new dequant codegen.
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, statistics, subprocess
from typing import Any

DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_ARTIFACT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/generation-g0prime")
DEFAULT_TENSORS = ("blk.20.attn_q.weight", "blk.13.ffn_gate.weight")  # ~20% and ~47% of peak
PEAK_GBS_MEASURED = 859.0
NOISE_BAND_PCT = 2.0

MODES = ("partial", "serial", "packed_load", "vector_load", "grouped", "tile_custom")
PARTS = (1, 2, 4)
BASELINE = {"mode": "partial", "opts": ["LOCAL:0:64"], "parts": 1, "row_group": 1, "label": "v1_partial"}

DEVICE_RE = re.compile(r"q4k_primitive_gemv:.*device_q4_eff=(?P<dev>[0-9.]+) GB/s")
CORRECT_RE = re.compile(r"primitive_gemv_correctness: (?P<status>PASS|FAIL)")


def _mode_opts(mode:str) -> list[str]:
  # tile_custom takes no opts (mirrors qk_threeway_load_microbench); others get a LOCAL.
  return [] if mode == "tile_custom" else ["LOCAL:0:64"]

def _candidates() -> list[dict[str, Any]]:
  cands = [BASELINE]
  for mode in MODES:
    for parts in PARTS:
      cands.append({"mode": mode, "opts": _mode_opts(mode), "parts": parts, "row_group": 2 if mode == "grouped" else 1,
                    "label": f"{mode}_parts{parts}"})
  # dedup the (mode, opts, parts, row_group) that equals the baseline
  seen, out = set(), []
  for c in cands:
    key = (c["mode"], tuple(c["opts"]), c["parts"], c["row_group"])
    if key in seen: continue
    seen.add(key); out.append(c)
  return out

def _classify(rc:int, out:str) -> str:
  if "KernelOptError" in out: return "illegal_opt"
  if "CompileError" in out or "compile failed" in out: return "compile_error"
  if rc != 0: return "error"
  return "pass"

def _measure(repo:pathlib.Path, model:pathlib.Path, tensor:str, cand:dict[str, Any], *,
             device:str, iters:int, seed:int, timeout:float, runs:int) -> dict[str, Any]:
  cmd = [".venv/bin/python", "extra/q4_k_bench.py", str(model), "--device", device, "--tensor", tensor,
         "--iters", str(iters), "--format", "text", "--activation", "random", "--seed", str(seed),
         "--primitive", "--primitive-mode", cand["mode"], "--primitive-parts", str(cand["parts"])]
  if cand["mode"] == "grouped": cmd += ["--primitive-row-group", str(cand["row_group"])]
  for opt in cand["opts"]: cmd += ["--primitive-opt", opt]
  env = {**os.environ, "DEV": device, "DEBUG": "2", "PYTHONPATH": "."}
  devs = []
  for _ in range(runs):
    try:
      proc = subprocess.run(cmd, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
      out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
      out, rc = (exc.stdout or "") + "\nTIMEOUT", 124
    st = _classify(rc, out)
    c, m = CORRECT_RE.search(out), DEVICE_RE.search(out)
    if st != "pass" or c is None or c["status"] != "PASS" or m is None:
      return {"status": "wrong" if (st == "pass") else st, "tail": "\n".join(out.strip().splitlines()[-3:])}
    devs.append(float(m["dev"]))
  return {"status": "pass", "device_q4_gbs": round(statistics.median(devs), 3)}

def _build_summary(rows:list[dict[str, Any]], tensors:tuple[str, ...], peak_gbs:float) -> dict[str, Any]:
  per_tensor, any_real_headroom, best_roofline_after = {}, False, 0.0
  for tensor in tensors:
    res = {r["label"]: r for r in rows if r["tensor"] == tensor}
    base = res.get("v1_partial", {}).get("device_q4_gbs")
    passing = [r for r in res.values() if r.get("device_q4_gbs") is not None and r["label"] != "v1_partial"]
    best = max(passing, key=lambda r: r["device_q4_gbs"], default=None)
    win = bool(best and base and (best["device_q4_gbs"] / base - 1) * 100 > NOISE_BAND_PCT)
    any_real_headroom = any_real_headroom or win
    winner_gbs = best["device_q4_gbs"] if (win and best) else base
    if winner_gbs: best_roofline_after = max(best_roofline_after, winner_gbs / peak_gbs * 100)
    per_tensor[tensor] = {
      "baseline_v1_partial_gbs": base, "baseline_roofline_pct": round(base / peak_gbs * 100, 2) if base else None,
      "best_other_label": best["label"] if best else None,
      "best_other_gbs": best["device_q4_gbs"] if best else None,
      "best_gain_vs_baseline_pct": round((best["device_q4_gbs"] / base - 1) * 100, 3) if (best and base) else None,
      "beats_baseline": win,
      "best_roofline_pct_after": round(winner_gbs / peak_gbs * 100, 2) if winner_gbs else None,
    }
  # Honest verdict: a win exists only if a kernel clears the noise band, but if even the best
  # existing kernel stays far below peak the residual headroom needs new dequant codegen, not a
  # model search over the (already fully enumerated) existing mode space.
  if not any_real_headroom:
    conclusion = "no_existing_kernel_beats_v1_partial_on_device_bottleneck_needs_new_dequant_codegen_g0pp"
  elif best_roofline_after < 60.0:
    conclusion = "small_real_device_win_via_existing_kernel_residual_headroom_needs_new_dequant_codegen_g0pp"
  else:
    conclusion = "existing_kernel_strategy_closes_most_of_the_gap_hand_to_g1"
  return {
    "kind": "qk_flywheel_generation_g0prime", "phase": "Phase G0'", "conclusion": conclusion,
    "measured_peak_gbs": peak_gbs, "noise_band_pct": NOISE_BAND_PCT,
    "modes_swept": list(MODES), "parts_swept": list(PARTS),
    "any_real_headroom": any_real_headroom,
    "best_roofline_pct_after_win": round(best_roofline_after, 2),
    "candidates_enumerated": len({r["label"] for r in rows}),
    "per_tensor": per_tensor,
    "metric": "median device_q4_eff as a fraction of measured peak HBM bandwidth, correctness-gated",
    "note": ("The existing mode x parts space is small and fully enumerated, so a marginal win leaves no "
             "role for model-guided search (G1) -- the residual headroom needs new dequant codegen (G0'')."),
  }

def run_g0prime(repo:pathlib.Path, model:pathlib.Path, tensors:tuple[str, ...], artifact:pathlib.Path, *,
                device:str="AMD", iters:int=3, seed:int=1337, timeout:float=180.0, runs:int=3,
                peak_gbs:float=PEAK_GBS_MEASURED) -> dict[str, Any]:
  repo, model = repo.resolve(), model.expanduser().resolve()
  rows = []
  for tensor in tensors:
    for cand in _candidates():
      r = _measure(repo, model, tensor, cand, device=device, iters=iters, seed=seed, timeout=timeout, runs=runs)
      row = {"tensor": tensor, "label": cand["label"], "mode": cand["mode"], "parts": cand["parts"], "status": r["status"]}
      if r["status"] == "pass":
        row["device_q4_gbs"] = r["device_q4_gbs"]
        row["roofline_pct"] = round(r["device_q4_gbs"] / peak_gbs * 100, 2)
      rows.append(row)
  summary = _build_summary(rows, tensors, peak_gbs)
  artifact.mkdir(parents=True, exist_ok=True)
  with (artifact / "candidates.jsonl").open("w") as f:
    for r in rows: f.write(json.dumps(r, sort_keys=True) + "\n")
  (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def rescore_g0prime(artifact:pathlib.Path, tensors:tuple[str, ...]=DEFAULT_TENSORS, peak_gbs:float=PEAK_GBS_MEASURED) -> dict[str, Any]:
  rows = [json.loads(l) for l in (artifact / "candidates.jsonl").read_text().splitlines() if l.strip()]
  summary = _build_summary(rows, tuple(sorted({r["tensor"] for r in rows})), peak_gbs)
  (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def main() -> int:
  p = argparse.ArgumentParser(description="Phase G0' device-metric kernel-strategy + occupancy search")
  p.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  p.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  p.add_argument("--tensor", action="append", default=None)
  p.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  p.add_argument("--device", default="AMD")
  p.add_argument("--rescore", action="store_true", help="rebuild summary.json from existing candidates.jsonl, no GPU")
  args = p.parse_args()
  tensors = tuple(args.tensor) if args.tensor else DEFAULT_TENSORS
  if args.rescore:
    print(json.dumps(rescore_g0prime(args.artifact), indent=2, sort_keys=True))
  else:
    print(json.dumps(run_g0prime(args.repo, args.model, tensors, args.artifact, device=args.device), indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
