#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, statistics, subprocess, sys, time
from typing import Any

from extra.q4_k_safety import assert_q4k_native_sweep_allowed

DEFAULT_ARTIFACT = pathlib.Path("bench/qk-threeway-load-microbench-20260613")
DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_TENSORS = ("blk.0.ffn_gate.weight",)
MODES = ("v1_partial", "vector_load", "tile_custom")
MEANINGFUL_GAIN_PCT = 5.0
TIE_BAND_PCT = 3.0

Q4_RESULT_RE = re.compile(
  r"^(?P<tensor>\S+) (?P<shape>\S+) q4k_primitive_gemv: .*?q4_eff=(?P<wall>[0-9.]+) GB/s "
  r"device_q4_eff=(?P<device>[0-9.]+|n/a)(?: GB/s)? .*?kernels=(?P<kernels>[0-9.]+)",
  re.MULTILINE,
)
GEMV_RE = re.compile(r"^primitive_gemv_correctness: PASS \S+ max_abs=(?P<max_abs>[0-9.eE+-]+)", re.MULTILINE)


from extra.qk_paths import portable_path as _portable


def _display_path(path:pathlib.Path) -> str:
  text = str(path)
  home = str(pathlib.Path.home())
  return text.replace(home + "/", "~/")


def _sanitize(text:str, *, repo:pathlib.Path, model:pathlib.Path) -> str:
  out = text.replace(str(repo.resolve()), ".")
  if model.exists():
    home = pathlib.Path.home().resolve()
    try:
      out = out.replace(str(model.resolve()), "~/" + str(model.resolve().relative_to(home)))
    except ValueError:
      out = out.replace(str(model.resolve()), model.name)
  return "\n".join(line.rstrip() for line in out.splitlines()) + ("\n" if text else "")


def _write(path:pathlib.Path, text:str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)


def _stats(values:list[float]) -> dict[str, float|int]:
  if not values: raise ValueError("cannot summarize empty metric list")
  return {
    "n": len(values),
    "median": statistics.median(values),
    "min": min(values),
    "max": max(values),
    "mean": statistics.fmean(values),
    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
  }


def _primitive_mode(mode:str) -> str:
  if mode == "v1_partial": return "partial"
  if mode in ("vector_load", "tile_custom"): return mode
  raise ValueError(f"unknown mode {mode!r}")


def _mode_opts(mode:str, opts:str) -> list[str]:
  if mode == "tile_custom": return []
  return ["--primitive-opt", opts]


def _command(py:pathlib.Path, model:pathlib.Path, tensor:str, mode:str, *, device:str, iters:int, seed:int, opts:str) -> list[str]:
  return [
    str(py), "extra/q4_k_bench.py", str(model), "--device", device, "--tensor", tensor,
    "--iters", str(iters), "--format", "text", "--activation", "random", "--seed", str(seed),
    "--primitive", "--primitive-mode", _primitive_mode(mode), "--primitive-parts", "1",
    *_mode_opts(mode, opts),
  ]


def _classify(returncode:int, output:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if returncode == 0: return "pass"
  if "shape mismatch" in output or "RuntimeError: shape mismatch" in output: return "construction_error"
  if "KernelOptError" in output: return "illegal_opt"
  if "CompileError" in output or "compile failed" in output: return "compile_error"
  if "correctness failed" in output or "AssertionError" in output: return "wrong"
  return "error"


def _parse_pass(output:str) -> dict[str, Any]:
  result = Q4_RESULT_RE.search(output)
  if result is None: raise ValueError("pass run did not contain q4k_primitive_gemv result line")
  corr = GEMV_RE.search(output)
  if corr is None: raise ValueError("pass run did not contain primitive GEMV correctness line")
  if result["device"] == "n/a": raise ValueError("device_q4_eff is n/a; rerun with DEBUG=2")
  return {
    "tensor": result["tensor"],
    "shape": result["shape"],
    "wall_q4_gbs": float(result["wall"]),
    "device_q4_gbs": float(result["device"]),
    "kernels": float(result["kernels"]),
    "primitive_gemv_max_abs": float(corr["max_abs"]),
  }


def _run_one(repo:pathlib.Path, model:pathlib.Path, py:pathlib.Path, tensor:str, mode:str, run_idx:int, *,
             outdir:pathlib.Path, device:str, iters:int, seed:int, timeout:float, opts:str, vary_seed:bool=False) -> dict[str, Any]:
  run_seed = seed + run_idx if vary_seed else seed
  cmd = _command(py, model, tensor, mode, device=device, iters=iters, seed=run_seed, opts=opts)
  env = {**os.environ, "DEV": device, "DEBUG": "2", "PYTHONPATH": "."}
  raw_base = outdir / "raw" / tensor.replace(".", "_") / mode / f"run-{run_idx:02d}"
  raw_json = raw_base.with_suffix(".json")
  raw_stdout = raw_base.with_suffix(".stdout")
  st = time.perf_counter()
  timed_out = False
  try:
    proc = subprocess.run(cmd, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    output, returncode = proc.stdout, proc.returncode
  except subprocess.TimeoutExpired as exc:
    timed_out = True
    partial = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    output, returncode = partial + "\nTIMEOUT\n", 124
  elapsed = time.perf_counter() - st
  sanitized = _sanitize(output, repo=repo, model=model)
  _write(raw_stdout, sanitized)
  status = _classify(returncode, output, timed_out)
  row: dict[str, Any] = {
    "tensor": tensor,
    "mode": mode,
    "run": run_idx,
    "status": status,
    "seed": run_seed,
    "seed_policy": "vary_by_run" if vary_seed else "fixed",
    "elapsed_s": round(elapsed, 3),
    "returncode": returncode,
    "command": " ".join(_sanitize(" ".join(cmd), repo=repo, model=model).splitlines()),
    "stdout": _portable(raw_stdout, repo),
    "raw_file": _portable(raw_json, repo),
    "tail": "\n".join(sanitized.strip().splitlines()[-12:]),
  }
  if status == "pass":
    row.update(_parse_pass(output))
  _write(raw_json, json.dumps(row, indent=2, sort_keys=True) + "\n")
  return row


def _mode_summary(rows:list[dict[str, Any]], mode:str) -> dict[str, Any]:
  mode_rows = [row for row in rows if row["mode"] == mode]
  pass_rows = [row for row in mode_rows if row["status"] == "pass"]
  status_counts = {status: sum(1 for row in mode_rows if row["status"] == status) for status in sorted({row["status"] for row in mode_rows})}
  out: dict[str, Any] = {
    "mode": mode,
    "runs": len(mode_rows),
    "status": "pass" if len(pass_rows) == len(mode_rows) and mode_rows else "invalid",
    "status_counts": status_counts,
    "raw_files": [row["raw_file"] for row in mode_rows],
  }
  if pass_rows:
    out.update({
      "device_q4_gbs": _stats([float(row["device_q4_gbs"]) for row in pass_rows]),
      "wall_q4_gbs": _stats([float(row["wall_q4_gbs"]) for row in pass_rows]),
      "primitive_gemv_max_abs": _stats([float(row["primitive_gemv_max_abs"]) for row in pass_rows]),
      "kernels": _stats([float(row["kernels"]) for row in pass_rows]),
    })
  else:
    out["failure_tail"] = mode_rows[0]["tail"] if mode_rows else "no runs"
  return out


def _gain(candidate:dict[str, Any], baseline:dict[str, Any]) -> float|None:
  if candidate["status"] != "pass" or baseline["status"] != "pass": return None
  return (float(candidate["device_q4_gbs"]["median"]) / float(baseline["device_q4_gbs"]["median"]) - 1.0) * 100.0


def summarize_runs(raw_runs:list[dict[str, Any]], *, meaningful_gain_pct:float=MEANINGFUL_GAIN_PCT,
                   tie_band_pct:float=TIE_BAND_PCT) -> dict[str, Any]:
  tensors = sorted({row["tensor"] for row in raw_runs})
  tensor_reports = []
  for tensor in tensors:
    rows = [row for row in raw_runs if row["tensor"] == tensor]
    modes = {mode: _mode_summary(rows, mode) for mode in MODES}
    v1, vector, tile = modes["v1_partial"], modes["vector_load"], modes["tile_custom"]
    vector_gain = _gain(vector, v1)
    tile_gain = _gain(tile, v1)
    vector_vs_tile = None
    if vector["status"] == "pass" and tile["status"] == "pass":
      vector_vs_tile = (float(vector["device_q4_gbs"]["median"]) / float(tile["device_q4_gbs"]["median"]) - 1.0) * 100.0

    if v1["status"] != "pass":
      decision = "blocked_invalid_measurement"
      reason = "v1 baseline failed"
    elif vector["status"] == "pass" and (vector_gain or 0.0) >= meaningful_gain_pct:
      decision = "vector_load_already_sufficient"
      reason = "schedulable vector_load beats v1 by the meaningful-gain threshold"
    elif vector["status"] == "pass" and (vector_gain or 0.0) < meaningful_gain_pct:
      decision = "wide_load_not_sufficient"
      reason = "schedulable vector_load does not beat v1 by the meaningful-gain threshold"
    elif tile["status"] == "pass" and (tile_gain or 0.0) >= meaningful_gain_pct:
      decision = "schedulable_vector_load_blocked"
      reason = "tile_custom has a meaningful gain but schedulable vector_load did not execute"
    else:
      decision = "inconclusive_threeway"
      reason = "schedulable vector_load did not execute, and tile_custom is an opaque no-LOCAL control"

    tensor_reports.append({
      "tensor": tensor,
      "decision": decision,
      "reason": reason,
      "meaningful_gain_pct": meaningful_gain_pct,
      "tie_band_pct": tie_band_pct,
      "gains_pct": {
        "vector_load_vs_v1": vector_gain,
        "tile_custom_vs_v1": tile_gain,
        "vector_load_vs_tile_custom": vector_vs_tile,
      },
      "modes": [modes[mode] for mode in MODES],
    })

  decisions = [row["decision"] for row in tensor_reports]
  if not decisions:
    overall = "blocked_invalid_measurement"
  elif all(decision == "vector_load_already_sufficient" for decision in decisions):
    overall = "vector_load_already_sufficient"
  elif any(decision == "schedulable_vector_load_blocked" for decision in decisions):
    overall = "schedulable_vector_load_blocked"
  elif all(decision == "wide_load_not_sufficient" for decision in decisions):
    overall = "wide_load_not_sufficient"
  elif any(decision == "blocked_invalid_measurement" for decision in decisions):
    overall = "blocked_invalid_measurement"
  else:
    overall = "inconclusive_threeway"

  return {
    "kind": "qk_threeway_load_microbench",
    "schema_version": 1,
    "summary": {
      "overall_decision": overall,
      "tensors": len(tensor_reports),
      "meaningful_gain_pct": meaningful_gain_pct,
      "tie_band_pct": tie_band_pct,
      "run_full_decode": False,
      "next_allowed_gate": (
        "harden_vector_load" if overall == "vector_load_already_sufficient" else
        "fix_schedulable_vector_consumption" if overall == "schedulable_vector_load_blocked" else
        "stop_wide_load_only_branch" if overall == "wide_load_not_sufficient" else
        "fix_measurement_or_scope"
      ),
    },
    "tensors": tensor_reports,
  }


def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Three-Way Load Microbench",
    "",
    f"Decision: `{report['summary']['overall_decision']}`",
    "",
    "Compares the current schedulable v1 Q4_K partial kernel, the schedulable",
    "`vector_load` kernel, and the opaque `tile_custom` wide-load kernel. This is",
    "a diagnostic artifact only: no runtime integration or full decode follows",
    "from this result.",
    "",
    "Method note: v1 and `vector_load` are the apples-to-apples comparison. Both",
    "use the schedulable primitive path and `LOCAL:0:32`. `tile_custom` is an",
    "opaque no-LOCAL control, so it can show construction feasibility but cannot",
    "by itself prove load-width performance.",
    "",
    "## Summary",
    "",
    f"- tensors: `{report['summary']['tensors']}`",
    f"- meaningful gain threshold: `{report['summary']['meaningful_gain_pct']:.2f}%`",
    f"- tie band: `{report['summary']['tie_band_pct']:.2f}%`",
    f"- run full decode next: `{report['summary']['run_full_decode']}`",
    f"- next allowed gate: `{report['summary']['next_allowed_gate']}`",
    "",
    "## Tensor Results",
    "",
    "| tensor | decision | v1 GB/s | vector GB/s | tile GB/s | vector vs v1 | tile vs v1 | vector status | tile status |",
    "|---|---|---:|---:|---:|---:|---:|---|---|",
  ]
  for row in report["tensors"]:
    modes = {mode["mode"]: mode for mode in row["modes"]}
    def med(mode:str) -> str:
      m = modes[mode]
      return "n/a" if m["status"] != "pass" else f"{m['device_q4_gbs']['median']:.2f}"
    def pct(value:float|None) -> str:
      return "n/a" if value is None else f"{value:.2f}%"
    lines.append(
      f"| `{row['tensor']}` | `{row['decision']}` | {med('v1_partial')} | {med('vector_load')} | {med('tile_custom')} | "
      f"{pct(row['gains_pct']['vector_load_vs_v1'])} | {pct(row['gains_pct']['tile_custom_vs_v1'])} | "
      f"`{modes['vector_load']['status']}` | `{modes['tile_custom']['status']}` |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
  ]
  decision = report["summary"]["overall_decision"]
  if decision == "vector_load_already_sufficient":
    lines.append("The schedulable vector path is good enough to harden. Do not build a new semantic op first.")
  elif decision == "schedulable_vector_load_blocked":
    lines.append("The opaque wide-load path beats v1 but the schedulable vector path does not. The target is vector consumption/codegen.")
  elif decision == "wide_load_not_sufficient":
    lines.append("The schedulable vector path does not beat v1 meaningfully. Do not chase load width alone; diagnose instruction mix or downstream dot/dequant cost.")
  else:
    lines.append("The measurement did not produce a decisive branch; fix the invalid schedulable mode or narrow the scope before continuing.")
  lines.append("")
  return "\n".join(lines)


def run(args:argparse.Namespace) -> dict[str, Any]:
  repo = args.repo.resolve()
  model = args.model.expanduser().resolve()
  if not model.exists(): raise FileNotFoundError(f"model not found: {model}")
  assert_q4k_native_sweep_allowed(args.device, "QK three-way load microbench")
  artifact = args.artifact
  raw_runs = []
  for tensor in args.tensor:
    for run_idx in range(args.runs):
      for mode in MODES:
        raw_runs.append(_run_one(repo, model, args.python, tensor, mode, run_idx, outdir=artifact, device=args.device,
                                 iters=args.iters, seed=args.seed, timeout=args.timeout, opts=args.opt, vary_seed=args.vary_seed))
  report = summarize_runs(raw_runs, meaningful_gain_pct=args.meaningful_gain_pct, tie_band_pct=args.tie_band_pct)
  report["artifact"] = _portable(artifact, repo)
  report["config"] = {
    "model": _display_path(model),
    "device": args.device,
    "tensors": list(args.tensor),
    "runs": args.runs,
    "iters": args.iters,
    "seed": args.seed,
    "seed_policy": "vary_by_run" if args.vary_seed else "fixed",
    "opt": args.opt,
    "modes": list(MODES),
    "debug": 2,
  }
  artifact.mkdir(parents=True, exist_ok=True)
  (artifact / "microbench.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  md = report_markdown(report)
  (artifact / "microbench.md").write_text(md)
  (artifact / "README.md").write_text(md)
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description="Three-way Q4_K wide-load diagnostic: v1 vs vector_load vs tile_custom")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  parser.add_argument("--python", type=pathlib.Path, default=pathlib.Path(".venv/bin/python"))
  parser.add_argument("--tensor", action="append", default=list(DEFAULT_TENSORS))
  parser.add_argument("--runs", type=int, default=5)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--vary-seed", action="store_true", help="vary activation seed by run; default keeps repeat inputs fixed")
  parser.add_argument("--timeout", type=float, default=240.0)
  parser.add_argument("--opt", default="LOCAL:0:32")
  parser.add_argument("--meaningful-gain-pct", type=float, default=MEANINGFUL_GAIN_PCT)
  parser.add_argument("--tie-band-pct", type=float, default=TIE_BAND_PCT)
  args = parser.parse_args()
  if args.runs < 1: raise ValueError("--runs must be >= 1")
  if args.iters < 1: raise ValueError("--iters must be >= 1")
  args.repo = args.repo.resolve()
  args.artifact = (args.repo / args.artifact).resolve() if not args.artifact.is_absolute() else args.artifact.resolve()
  report = run(args)
  print(report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
