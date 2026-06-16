#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, time
from typing import Any, Callable

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt
from tinygrad.helpers import GlobalCounters

from extra.q4_k_gemv_primitive import parse_opt, q4k_gemv_partial_kernel
from extra.qk_block_dot_compile_gate import DEFAULT_TENSOR, q4k_block_dot_partial_kernel
from extra.qk_layout import (
  GGML_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, pick_tensor, q4_k_reference, read_metadata, tensor_shape,
)

DEFAULT_ARTIFACT = pathlib.Path("bench/qk-block-dot-microbench-20260613")
DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
PROMOTION_GAIN_PCT = 10.0
TOLERANCE = 1e-2


from extra.qk_paths import portable_path as _portable


def _display_path(path:pathlib.Path) -> str:
  text = str(path)
  home = str(pathlib.Path.home())
  return text.replace(home + "/", "~/")


from extra.llm_eval_common import value_stats as _stats


def _mode_kernel(mode:str, rows:int, k:int, parts:int, opts:tuple[Opt, ...]):
  if mode == "v1_partial": return q4k_gemv_partial_kernel(rows, k, parts, "none", opts)
  if mode == "qk_block_dot": return q4k_block_dot_partial_kernel(rows, k, parts, opts)
  raise ValueError(f"unknown mode {mode!r}")


def _measure(fn:Callable[[], Tensor], *, iters:int, q4_bytes:int) -> dict[str, float|int|None]:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  wall_s = (time.perf_counter() - st) / iters
  device_s = GlobalCounters.time_sum_s / iters
  return {
    "iters": iters,
    "wall_ms": wall_s * 1000.0,
    "wall_q4_gbs": q4_bytes / wall_s / 1e9,
    "device_ms": device_s * 1000.0 if device_s > 0 else None,
    "device_q4_gbs": q4_bytes / device_s / 1e9 if device_s > 0 else None,
    "kernels": GlobalCounters.kernel_count / iters,
    "global_mem_mb": GlobalCounters.global_mem / iters / 1e6,
  }


def _mode_result(mode:str, partials:Tensor, words:Tensor, x:Tensor, ref:Tensor, rows:int, k:int, parts:int,
                 opts:tuple[Opt, ...], *, q4_bytes:int, iters:int) -> dict[str, Any]:
  def fn() -> Tensor:
    partial = partials.custom_kernel(words, x, fxn=_mode_kernel(mode, rows, k, parts, opts))[0]
    return partial.sum(axis=1)

  got = fn().realize()
  max_abs = (got - ref).abs().max().item()
  if max_abs > TOLERANCE:
    raise AssertionError(f"{mode} GEMV correctness failed: max_abs={max_abs}")
  return {
    "mode": mode,
    "primitive_gemv_max_abs": max_abs,
    **_measure(fn, iters=iters, q4_bytes=q4_bytes),
  }


def summarize_runs(raw_runs:list[dict[str, Any]]) -> dict[str, Any]:
  grouped: dict[str, list[dict[str, Any]]] = {}
  for row in raw_runs:
    grouped.setdefault(str(row["mode"]), []).append(row)
  required = {"v1_partial", "qk_block_dot"}
  missing = sorted(required - set(grouped))
  if missing: raise ValueError(f"missing benchmark modes: {missing}")

  modes = []
  for mode in ("v1_partial", "qk_block_dot"):
    rows = grouped[mode]
    device = [float(row["device_q4_gbs"]) for row in rows if row.get("device_q4_gbs") is not None]
    if len(device) != len(rows):
      raise ValueError(f"{mode}: device timing missing for {len(rows)-len(device)} run(s); rerun with DEBUG=2")
    wall = [float(row["wall_q4_gbs"]) for row in rows]
    ms = [float(row["device_ms"]) for row in rows if row.get("device_ms") is not None]
    max_abs = [float(row["primitive_gemv_max_abs"]) for row in rows]
    modes.append({
      "mode": mode,
      "runs": len(rows),
      "device_q4_gbs": _stats(device),
      "wall_q4_gbs": _stats(wall),
      "device_ms": _stats(ms),
      "primitive_gemv_max_abs": _stats(max_abs),
      "raw_files": [row["raw_file"] for row in rows],
    })

  by_mode = {row["mode"]: row for row in modes}
  v1 = float(by_mode["v1_partial"]["device_q4_gbs"]["median"])
  qk = float(by_mode["qk_block_dot"]["device_q4_gbs"]["median"])
  gain_pct = (qk / v1 - 1.0) * 100.0
  correctness_ok = max(float(row["primitive_gemv_max_abs"]["max"]) for row in modes) <= TOLERANCE
  if not correctness_ok:
    decision = "qk_block_dot_microbench_invalid_correctness"
  elif gain_pct >= PROMOTION_GAIN_PCT:
    decision = "qk_block_dot_microbench_raw_accept_unconfirmed"
  else:
    decision = "qk_block_dot_microbench_rejected"

  return {
    "kind": "qk_block_dot_microbench",
    "schema_version": 1,
    "modes": modes,
    "comparison": {
      "metric": "median device_q4_gbs",
      "v1_median_device_q4_gbs": v1,
      "qk_block_dot_median_device_q4_gbs": qk,
      "gain_pct": gain_pct,
      "promotion_gain_pct": PROMOTION_GAIN_PCT,
    },
    "summary": {
      "decision": decision,
      "correctness_ok": correctness_ok,
      "raw_accept": decision == "qk_block_dot_microbench_raw_accept_unconfirmed",
      "run_full_decode": False,
      "next_allowed_gate": (
        "full_decode_confirmation_after_explicit_scope"
        if decision == "qk_block_dot_microbench_raw_accept_unconfirmed" else
        "stop_qk_block_dot_runtime_integration"
      ),
      "reason": (
        f"median device Q4 GB/s gain {gain_pct:.2f}% vs required {PROMOTION_GAIN_PCT:.2f}%"
      ),
    },
  }


def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK_BLOCK_DOT Microbench",
    "",
    f"Decision: `{report['summary']['decision']}`",
    "",
    "Repeated dominant-shape microbench for the first `QK_BLOCK_DOT` lowering.",
    "This is still not a runtime integration or full-decode artifact.",
    "",
    "## Summary",
    "",
    f"- metric: `{report['comparison']['metric']}`",
    f"- v1 median device Q4 GB/s: `{report['comparison']['v1_median_device_q4_gbs']:.2f}`",
    f"- QK_BLOCK_DOT median device Q4 GB/s: `{report['comparison']['qk_block_dot_median_device_q4_gbs']:.2f}`",
    f"- gain: `{report['comparison']['gain_pct']:.2f}%`",
    f"- promotion bar: `>={report['comparison']['promotion_gain_pct']:.2f}%`",
    f"- correctness ok: `{report['summary']['correctness_ok']}`",
    f"- run full decode next: `{report['summary']['run_full_decode']}`",
    "",
    "Device timing is the gate metric. Wall timing is recorded only as secondary",
    "diagnostic data and is noisy when the run is executed with `DEBUG=2` to",
    "collect AMD device times.",
    "",
    "## Modes",
    "",
    "| mode | runs | median device GB/s | median device ms | median wall GB/s | max_abs max |",
    "|---|---:|---:|---:|---:|---:|",
  ]
  for row in report["modes"]:
    lines.append(
      f"| `{row['mode']}` | {row['runs']} | {row['device_q4_gbs']['median']:.2f} | "
      f"{row['device_ms']['median']:.6f} | {row['wall_q4_gbs']['median']:.2f} | "
      f"{row['primitive_gemv_max_abs']['max']:.6g} |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
  ]
  if report["summary"]["decision"] == "qk_block_dot_microbench_raw_accept_unconfirmed":
    lines += [
      "The repeated microbench clears the pre-registered `>=10%` bar. This is a",
      "raw accept only: the next step must be scoped as a separate full-decode",
      "confirmation gate, with greedy A/B, before any runtime promotion.",
    ]
  elif report["summary"]["decision"] == "qk_block_dot_microbench_rejected":
    lines += [
      "The compile-shape win did not translate into enough repeated microbench",
      "speedup. Do not integrate `QK_BLOCK_DOT` into runtime or run full decode",
      "from this result. The next research step should inspect why the wider",
      "loads are not paying off at this shape.",
    ]
  else:
    lines.append("The correctness gate failed; fix correctness before reading any timing.")
  lines.append("")
  return "\n".join(lines)


def run_benchmark(args:argparse.Namespace) -> dict[str, Any]:
  repo = args.repo.resolve()
  model = args.model.expanduser().resolve()
  meta = read_metadata(model)
  info = pick_tensor(meta.infos, args.tensor)
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  shape = tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, k = (shape[0] if args.rows is None else args.rows), shape[1]
  if rows < 1 or rows > shape[0]: raise ValueError(f"--rows must be in [1,{shape[0]}], got {rows}")
  if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"K={k} is not Q4_K block aligned")
  if (k // Q4_K_BLOCK_ELEMS) % args.parts != 0: raise ValueError("parts must divide Q4_K blocks exactly")
  opts = tuple(parse_opt(x) for x in args.opt)

  byte_start = meta.data_start + info.off
  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  q4_bytes = rows * row_bytes
  nwords = q4_bytes // 4

  raw_words = Tensor(model, dtype=dtypes.uint32)
  words = raw_words[byte_start//4:byte_start//4+nwords].to(args.device).contiguous().realize()
  raw_u8 = Tensor(model)[byte_start:byte_start+q4_bytes].to(args.device)
  Tensor.manual_seed(args.seed)
  x = Tensor.randn(k, dtype=dtypes.float16, device=args.device).realize()
  decoded = q4_k_reference(raw_u8, rows*k).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()
  partials = {
    mode: Tensor.empty(rows, args.parts, dtype=dtypes.float32, device=args.device) for mode in ("v1_partial", "qk_block_dot")
  }

  # Compile both kernels before the timed paired runs.
  for mode in ("v1_partial", "qk_block_dot"):
    partial = partials[mode].custom_kernel(words, x, fxn=_mode_kernel(mode, rows, k, args.parts, opts))[0]
    partial.sum(axis=1).realize()

  raw_runs = []
  raw_dir = args.artifact / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  for run_idx in range(args.runs):
    for mode in ("v1_partial", "qk_block_dot"):
      row = _mode_result(mode, partials[mode], words, x, ref, rows, k, args.parts, opts, q4_bytes=q4_bytes, iters=args.iters)
      row.update({
        "run": run_idx,
        "tensor": info.name,
        "shape": [rows, k],
        "parts": args.parts,
        "opts": [str(x) for x in opts],
        "q4_bytes": q4_bytes,
        "raw_file": _portable(raw_dir / f"run-{run_idx:02d}-{mode}.json", repo),
      })
      (raw_dir / f"run-{run_idx:02d}-{mode}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
      raw_runs.append(row)

  report = summarize_runs(raw_runs)
  report["artifact"] = _portable(args.artifact, repo)
  report["source_compile_gate"] = "bench/qk-block-dot-compile-gate-20260613/compile-gate.json"
  report["config"] = {
    "model": _display_path(model),
    "device": args.device,
    "tensor": info.name,
    "shape": [rows, k],
    "full_tensor_shape": list(shape),
    "parts": args.parts,
    "opts": [str(x) for x in opts],
    "runs": args.runs,
    "iters": args.iters,
    "seed": args.seed,
    "q4_bytes": q4_bytes,
    "q4_words_per_block": Q4K_WORDS_PER_BLOCK,
  }
  args.artifact.mkdir(parents=True, exist_ok=True)
  (args.artifact / "microbench.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  md = report_markdown(report)
  (args.artifact / "microbench.md").write_text(md)
  (args.artifact / "README.md").write_text(md)
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description="Repeated QK_BLOCK_DOT dominant-shape microbench")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  parser.add_argument("--tensor", default=DEFAULT_TENSOR)
  parser.add_argument("--rows", type=int, default=None, help="rows to benchmark; default is the full tensor")
  parser.add_argument("--parts", type=int, default=1)
  parser.add_argument("--opt", action="append", default=["LOCAL:0:32"])
  parser.add_argument("--runs", type=int, default=5)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--seed", type=int, default=1337)
  args = parser.parse_args()
  if args.runs < 1: raise ValueError("--runs must be >= 1")
  if args.iters < 1: raise ValueError("--iters must be >= 1")
  args.repo = args.repo.resolve()
  args.artifact = (args.repo / args.artifact).resolve() if not args.artifact.is_absolute() else args.artifact.resolve()
  report = run_benchmark(args)
  print(report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
