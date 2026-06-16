#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from typing import Any

from extra.qk_load_width_report import build_report as build_load_width_report, report_markdown as load_width_report_markdown

DEFAULT_TENSORS = (
  "blk.0.ffn_gate.weight",
  "blk.0.ffn_up.weight",
  "blk.0.attn_output.weight",
  "blk.0.attn_q.weight",
  "blk.0.attn_k.weight",
)

MODE_SPECS = {
  "v1_partial": {
    "primitive_mode": "partial",
    "primitive_parts": 1,
    "extra_args": ("--primitive-opt", "LOCAL:0:32"),
  },
  "tile_custom": {
    "primitive_mode": "tile_custom",
    "primitive_parts": 1,
    "extra_args": (),
  },
}

def _portable(path:pathlib.Path, repo:pathlib.Path) -> str:
  resolved = str(repo.resolve())
  text = str(path)
  if text == resolved: return "."
  return text.replace(resolved + "/", "")

def _display_path(path:pathlib.Path) -> str:
  text = str(path)
  home = str(pathlib.Path.home())
  return text.replace(home + "/", "~/")

def _write_text(path:pathlib.Path, text:str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  # DEBUG logs contain padded profiler rows; strip line-end whitespace so
  # committed evidence still passes git whitespace checks.
  path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + ("\n" if text else ""))

def _load_bench_json(path:pathlib.Path) -> list[dict[str, Any]]:
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"failed to parse benchmark JSON at {path}: {exc}") from exc

def _primitive_row(rows:list[dict[str, Any]]) -> dict[str, Any]:
  matches = [row for row in rows if row.get("name") == "q4k_primitive_gemv"]
  if len(matches) != 1:
    raise ValueError(f"expected exactly one q4k_primitive_gemv row, got {len(matches)}")
  return matches[0]

from extra.llm_eval_common import value_stats as _stats

def summarize_runs(raw_runs:list[dict[str, Any]]) -> dict[str, Any]:
  grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for run in raw_runs:
    key = (str(run["tensor"]), str(run["mode"]))
    grouped.setdefault(key, []).append(run)

  modes = []
  for (tensor, mode), rows in sorted(grouped.items()):
    gbs = [float(row["q4_eff_gbs"]) for row in rows]
    ms = [float(row["ms"]) for row in rows]
    max_abs = [float(row["primitive_gemv_max_abs"]) for row in rows]
    modes.append({
      "tensor": tensor,
      "mode": mode,
      "runs": len(rows),
      "q4_eff_gbs": _stats(gbs),
      "ms": _stats(ms),
      "primitive_gemv_max_abs": _stats(max_abs),
      "raw_files": [row["raw_file"] for row in rows],
    })

  by_tensor_mode = {(row["tensor"], row["mode"]): row for row in modes}
  comparisons = []
  for tensor in sorted({row["tensor"] for row in modes}):
    v1 = by_tensor_mode.get((tensor, "v1_partial"))
    tile = by_tensor_mode.get((tensor, "tile_custom"))
    if v1 is None or tile is None:
      comparisons.append({"tensor": tensor, "status": "missing_mode"})
      continue
    v1_gbs = float(v1["q4_eff_gbs"]["median"])
    tile_gbs = float(tile["q4_eff_gbs"]["median"])
    gain_pct = (tile_gbs / v1_gbs - 1.0) * 100.0
    comparisons.append({
      "tensor": tensor,
      "status": "compared",
      "v1_median_q4_eff_gbs": v1_gbs,
      "tile_custom_median_q4_eff_gbs": tile_gbs,
      "gain_pct": gain_pct,
      "v1_median_ms": float(v1["ms"]["median"]),
      "tile_custom_median_ms": float(tile["ms"]["median"]),
      "tile_custom_max_abs_median": float(tile["primitive_gemv_max_abs"]["median"]),
    })

  compared = [row for row in comparisons if row.get("status") == "compared"]
  if not compared:
    decision = "blocked_no_comparisons"
  elif all(float(row["gain_pct"]) >= 10.0 for row in compared):
    decision = "promote_to_8b_full_decode_candidate"
  elif any(float(row["gain_pct"]) >= 5.0 for row in compared):
    decision = "diagnose_only_not_promoted"
  else:
    decision = "stop_raw_custom_path"

  return {
    "kind": "qk_packed_tile_lowering_analysis",
    "schema_version": 1,
    "modes": modes,
    "comparisons": comparisons,
    "summary": {
      "decision": decision,
      "promotion_gate": "median tile_custom gain >=10% on every measured Q4_K tensor before full decode",
      "measured_tensors": len(compared),
      "min_gain_pct": min((float(row["gain_pct"]) for row in compared), default=None),
      "max_gain_pct": max((float(row["gain_pct"]) for row in compared), default=None),
      "median_gain_pct": statistics.median([float(row["gain_pct"]) for row in compared]) if compared else None,
    },
  }

def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# Packed QK Tile Lowering Analysis",
    "",
    "Repeated 8B Q4_K microbench comparison of the current v1 partial kernel",
    "against the raw custom `PackedQKTile` consumer. This is a diagnostic gate,",
    "not a runtime-promotion artifact.",
    "",
    "## Decision",
    "",
    f"- decision: `{report['summary']['decision']}`",
    f"- promotion gate: {report['summary']['promotion_gate']}",
    f"- measured tensors: `{report['summary']['measured_tensors']}`",
    f"- gain range: `{report['summary']['min_gain_pct']:.2f}%` to `{report['summary']['max_gain_pct']:.2f}%`"
    if report["summary"]["min_gain_pct"] is not None else "- gain range: `n/a`",
    "",
    "## Comparison",
    "",
    "| tensor | v1 median Q4 GB/s | tile median Q4 GB/s | gain % | v1 median ms | tile median ms | tile max_abs median |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for row in report["comparisons"]:
    if row.get("status") != "compared":
      lines.append(f"| `{row['tensor']}` | n/a | n/a | n/a | n/a | n/a | n/a |")
      continue
    lines.append(
      f"| `{row['tensor']}` | {row['v1_median_q4_eff_gbs']:.2f} | {row['tile_custom_median_q4_eff_gbs']:.2f} | "
      f"{row['gain_pct']:.2f} | {row['v1_median_ms']:.6f} | {row['tile_custom_median_ms']:.6f} | "
      f"{row['tile_custom_max_abs_median']:.6g} |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
  ]
  decision = report["summary"]["decision"]
  if decision == "promote_to_8b_full_decode_candidate":
    lines += [
      "The raw custom tile path clears the pre-registered microbench bar on every",
      "measured Q4_K tensor. The next gate is an 8B full-decode confirmation run;",
      "14B should wait until 8B confirms.",
    ]
  elif decision == "diagnose_only_not_promoted":
    lines += [
      "The raw custom tile path has a positive signal on at least one tensor, but",
      "does not clear the full-decode promotion bar across the measured Q4_K set.",
      "Do not integrate it into runtime from this result. The next step is source,",
      "assembly, or counter analysis to explain why vector-source loads produce",
      "only a partial bandwidth improvement.",
    ]
  elif decision == "stop_raw_custom_path":
    lines += [
      "The raw custom tile path does not show a useful repeated microbench gain.",
      "Stop this raw custom-C path and move only if the packed-tile idea is",
      "promoted into a core lowering/search surface for a different reason.",
    ]
  else:
    lines.append("No valid comparison was produced; fix the harness before drawing a performance conclusion.")
  lines.append("")
  return "\n".join(lines)

def _run(cmd:list[str], *, cwd:pathlib.Path, env:dict[str, str], stdout_path:pathlib.Path, stderr_path:pathlib.Path) -> None:
  stdout_path.parent.mkdir(parents=True, exist_ok=True)
  stderr_path.parent.mkdir(parents=True, exist_ok=True)
  proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
  _write_text(stdout_path, proc.stdout)
  _write_text(stderr_path, proc.stderr)
  if proc.returncode != 0:
    raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}\nstderr: {proc.stderr[-2000:]}")

def _bench_command(py:str, model:pathlib.Path, device:str, tensor:str, mode:str, iters:int) -> list[str]:
  spec = MODE_SPECS[mode]
  cmd = [
    py, "extra/q4_k_bench.py", str(model), "--device", device, "--tensor", tensor,
    "--iters", str(iters), "--primitive", "--primitive-mode", str(spec["primitive_mode"]),
    "--primitive-parts", str(spec["primitive_parts"]), "--format", "json",
  ]
  cmd += list(spec["extra_args"])
  return cmd

def _source_command(py:str, model:pathlib.Path, device:str, tensor:str, mode:str, rows:int) -> list[str]:
  spec = MODE_SPECS[mode]
  cmd = [
    py, "extra/q4_k_gemv_primitive.py", str(model), "--device", device, "--tensor", tensor,
    "--rows", str(rows), "--iters", "1", "--mode", str(spec["primitive_mode"]),
    "--parts", str(spec["primitive_parts"]),
  ]
  if mode == "v1_partial": cmd += ["--opt", "LOCAL:0:32"]
  return cmd

def run_analysis(args:argparse.Namespace) -> dict[str, Any]:
  repo = args.repo.resolve()
  outdir = args.outdir
  py = str(args.python)
  env = os.environ.copy()
  env["PYTHONPATH"] = "."
  env["DEV"] = args.device

  source_logs: list[pathlib.Path] = []
  for mode in ("v1_partial", "tile_custom"):
    log = outdir / "source" / f"{mode}-debug4.log"
    err = outdir / "source" / f"{mode}-debug4.err"
    cmd = _source_command(py, args.model.expanduser(), args.device, args.source_tensor, mode, args.source_rows)
    source_env = {**env, "DEBUG": "4"}
    _run(cmd, cwd=repo, env=source_env, stdout_path=log, stderr_path=err)
    source_logs.append(log)

  load_report = build_load_width_report(source_logs, repo=repo)
  (outdir / "source").mkdir(parents=True, exist_ok=True)
  (outdir / "source" / "load-width-report.json").write_text(json.dumps(load_report, indent=2, sort_keys=True))
  (outdir / "source" / "load-width-report.md").write_text(load_width_report_markdown(load_report))

  raw_runs: list[dict[str, Any]] = []
  for tensor in args.tensor:
    safe_tensor = tensor.replace(".", "_")
    for mode in ("v1_partial", "tile_custom"):
      for run_idx in range(args.runs):
        raw = outdir / "raw" / safe_tensor / mode / f"run-{run_idx:02d}.json"
        err = outdir / "raw" / safe_tensor / mode / f"run-{run_idx:02d}.err"
        cmd = _bench_command(py, args.model.expanduser(), args.device, tensor, mode, args.iters)
        _run(cmd, cwd=repo, env=env, stdout_path=raw, stderr_path=err)
        row = _primitive_row(_load_bench_json(raw))
        raw_runs.append({
          "tensor": tensor,
          "mode": mode,
          "run": run_idx,
          "raw_file": _portable(raw, repo),
          "ms": row["ms"],
          "q4_eff_gbs": row["q4_eff_gbs"],
          "primitive_gemv_max_abs": row["primitive_gemv_max_abs"],
          "kernels": row["kernels"],
        })

  report = summarize_runs(raw_runs)
  report["source"] = {
    "source_tensor": args.source_tensor,
    "source_rows": args.source_rows,
    "load_width_report": _portable(outdir / "source" / "load-width-report.json", repo),
    "logs": [_portable(path, repo) for path in source_logs],
  }
  report["config"] = {
    "model": _display_path(args.model),
    "device": args.device,
    "runs": args.runs,
    "iters": args.iters,
    "tensors": list(args.tensor),
  }
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True))
  (outdir / "analysis.md").write_text(report_markdown(report))
  return report

def main() -> int:
  parser = argparse.ArgumentParser(description="Repeated v1-vs-tile-custom Q4_K packed tile lowering analysis")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf").expanduser())
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bench/qk-packed-tile-lowering-analysis-20260613"))
  parser.add_argument("--python", type=pathlib.Path, default=pathlib.Path(".venv/bin/python"))
  parser.add_argument("--runs", type=int, default=5)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--tensor", action="append", default=list(DEFAULT_TENSORS))
  parser.add_argument("--source-tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--source-rows", type=int, default=64)
  args = parser.parse_args()
  if args.runs < 1: raise ValueError("--runs must be >= 1")
  if args.iters < 1: raise ValueError("--iters must be >= 1")
  if args.source_rows < 1: raise ValueError("--source-rows must be >= 1")
  report = run_analysis(args)
  print(report_markdown(report))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
