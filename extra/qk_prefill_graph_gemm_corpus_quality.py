#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, pathlib, platform, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_gemm_corpus_quality_result.json"


def _git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def _parse_offsets(s: str) -> list[int]:
  vals = [int(x) for x in s.split(",") if x.strip()]
  if not vals: raise ValueError("--score-offsets must contain at least one integer")
  for v in vals:
    if v < 0 or v >= 511: raise ValueError(f"score offset {v} is invalid for target offset+1 in a 512-token window")
  return vals


def _logsumexp(xs) -> float:
  import numpy as np
  m = float(np.max(xs))
  return m + math.log(float(np.exp(xs - m).sum()))


def _score_prefill_v2_position(model: Any, win: list[int], offset: int) -> tuple[float, int, bool]:
  import numpy as np
  from tinygrad import Tensor
  import tinygrad.codegen.opt.postrange as pr

  for lin in (getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []):
    lin.decode_enabled = False
  for block in model.blk:
    block._prefill_v2 = True
    block._use_flash = False

  t = Tensor([win], dtype="int32").contiguous()
  saved = pr._WARMSTART_OPTS
  pr._WARMSTART_OPTS = model._pf16_warmstart
  try:
    logits = model.logits(t, 0)[:, offset, :].realize()[0].numpy()
  finally:
    pr._WARMSTART_OPTS = saved

  target = int(win[offset + 1])
  nll = _logsumexp(logits) - float(logits[target])
  return float(nll), int(np.argmax(logits)), bool(np.isfinite(logits).all())


def _child_eval(args: argparse.Namespace) -> dict[str, Any]:
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_prefill_v2_nll_eval import CALIB_TEXT

  offsets = _parse_offsets(args.score_offsets)
  Tensor.manual_seed(args.seed)
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  if not getattr(model, "_pf16_warmstart", None):
    raise RuntimeError("PREFILL_V2 warmstart missing; run with PREFILL_V2=1")

  prefix = tok.prefix() if hasattr(tok, "prefix") else []
  body = tok.encode(CALIB_TEXT)
  need = args.stride * (args.windows - 1) + args.ubatch
  ids = prefix + body
  while len(ids) < need:
    ids += body

  rows = []
  for w in range(args.windows):
    start = w * args.stride
    win = ids[start:start + args.ubatch]
    for offset in offsets:
      nll, pred, finite = _score_prefill_v2_position(model, win, offset)
      target = int(win[offset + 1])
      rows.append({
        "window": w, "start": start, "ubatch": args.ubatch, "score_offset": offset,
        "target": target, "argmax": pred, "hit": pred == target, "nll": round(nll, 6),
        "finite": finite,
      })
      print(f"{args.child_label} window {w} offset {offset}: nll={nll:.6f} argmax={pred} target={target}",
            file=sys.__stdout__)

  return {
    "label": args.child_label,
    "prefill_graph_gemm": bool(int(os.environ.get("PREFILL_GRAPH_GEMM", "0"))),
    "rows": rows,
    "mean_nll": round(sum(r["nll"] for r in rows) / len(rows), 6),
    "scored_positions": len(rows),
  }


def _run_child(model: str, label: str, graph: bool, args: argparse.Namespace) -> dict[str, Any]:
  env = {**os.environ, "DEV": os.environ.get("DEV", "AMD"), "PREFILL_V2": "1", "PYTHONPATH": "."}
  if graph: env["PREFILL_GRAPH_GEMM"] = "1"
  else: env.pop("PREFILL_GRAPH_GEMM", None)
  cmd = [
    sys.executable, str(pathlib.Path(__file__).resolve()), "--child", "--child-label", label,
    "--model", model, "--max-context", str(args.max_context), "--ubatch", str(args.ubatch),
    "--windows", str(args.windows), "--stride", str(args.stride), "--score-offsets", args.score_offsets,
    "--seed", str(args.seed),
  ]
  last: subprocess.CompletedProcess[str] | None = None
  failures = 0
  for attempt in range(args.retries + 1):
    last = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if last.returncode == 0: break
    failures += 1
    print(last.stdout, file=sys.__stdout__)
    if attempt < args.retries:
      time.sleep(2.0)
      print(f"retrying {label} child after failure ({attempt + 1}/{args.retries})", file=sys.__stdout__)
  assert last is not None
  if last.returncode != 0:
    raise RuntimeError(f"{label} child failed with code {last.returncode}")
  print(last.stdout, file=sys.__stdout__)
  for line in reversed(last.stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      ret = json.loads(line)
      ret["retry_failures"] = failures
      return ret
  raise RuntimeError(f"{label} child produced no JSON result")


def main() -> int:
  ap = argparse.ArgumentParser(description="VRAM-safe corpus-style quality gate for PREFILL_GRAPH_GEMM")
  ap.add_argument("pos_model", nargs="?")
  ap.add_argument("--model", default=None)
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--ubatch", type=int, default=512)
  ap.add_argument("--windows", type=int, default=8)
  ap.add_argument("--stride", type=int, default=64)
  ap.add_argument("--score-offsets", default="128,256,384,510")
  ap.add_argument("--eps-mean", type=float, default=0.002)
  ap.add_argument("--eps-max", type=float, default=0.01)
  ap.add_argument("--argmax-mismatch-max", type=int, default=0)
  ap.add_argument("--seed", type=int, default=20260620)
  ap.add_argument("--retries", type=int, default=1)
  ap.add_argument("--artifact", default=str(OUT))
  ap.add_argument("--child", action="store_true")
  ap.add_argument("--child-label", default="child")
  args = ap.parse_args()
  args.model = args.model or args.pos_model or os.environ.get("QK_MODEL") or os.environ.get("MODEL")
  if not args.model:
    print("ERROR: pass a model gguf path or set QK_MODEL / MODEL", file=sys.__stdout__)
    return 2
  if args.ubatch != 512:
    print("ERROR: this probe intentionally keeps ubatch=512 so PREFILL_GRAPH_GEMM routing is exercised",
          file=sys.__stdout__)
    return 2
  _parse_offsets(args.score_offsets)
  if args.child:
    res = _child_eval(args)
    print(json.dumps(res, sort_keys=True), file=sys.__stdout__)
    return 0

  baseline = _run_child(args.model, "baseline_prefill_v2", False, args)
  graph = _run_child(args.model, "graph_gemm", True, args)
  if len(baseline["rows"]) != len(graph["rows"]):
    raise RuntimeError(f"row count mismatch: baseline={len(baseline['rows'])} graph={len(graph['rows'])}")

  rows = []
  for brow, grow in zip(baseline["rows"], graph["rows"]):
    key_b = (brow["window"], brow["score_offset"])
    key_g = (grow["window"], grow["score_offset"])
    if key_b != key_g: raise RuntimeError(f"row key mismatch: {key_b} vs {key_g}")
    dnll = round(grow["nll"] - brow["nll"], 6)
    rows.append({**grow, "baseline_nll": brow["nll"], "graph_nll": grow["nll"], "dNLL": dnll,
                 "baseline_argmax": brow["argmax"], "argmax_match_baseline": grow["argmax"] == brow["argmax"]})

  mean_dnll = round(sum(r["dNLL"] for r in rows) / len(rows), 6)
  max_abs_dnll = max(abs(r["dNLL"]) for r in rows)
  max_pos_dnll = max(r["dNLL"] for r in rows)
  argmax_mismatches = sum(0 if r["argmax_match_baseline"] else 1 for r in rows)
  retry_failures = int(baseline.get("retry_failures", 0)) + int(graph.get("retry_failures", 0))
  offsets = _parse_offsets(args.score_offsets)
  expected_positions = args.windows * len(offsets)
  gates = {
    "baseline_finite": all(r["finite"] for r in baseline["rows"]),
    "graph_finite": all(r["finite"] for r in graph["rows"]),
    "scored_positions_expected": len(rows) == expected_positions,
    "mean_dNLL_lte_eps": mean_dnll <= args.eps_mean,
    "max_positive_dNLL_lte_eps": max_pos_dnll <= args.eps_max,
    "max_abs_dNLL_lte_eps": max_abs_dnll <= args.eps_max,
    "argmax_mismatches_lte_limit": argmax_mismatches <= args.argmax_mismatch_max,
    "child_retry_failures_zero": retry_failures == 0,
  }
  verdict = "PASS_PREFILL_GRAPH_GEMM_CORPUS_QUALITY" if all(gates.values()) else "BLOCKED_PREFILL_GRAPH_GEMM_CORPUS_QUALITY"
  result = {
    "date": "2026-06-20", "phase": "PREFILL_GRAPH_GEMM_CORPUS_QUALITY",
    "schema": "prefill_graph_gemm_corpus_quality_v1", "verdict": verdict,
    "gate_pass": all(gates.values()), "model_id": pathlib.Path(args.model).name,
    "hardware": platform.node(), "commit": _git_sha(), "ubatch": args.ubatch, "windows": args.windows,
    "stride": args.stride, "score_offsets": offsets, "scored_positions": len(rows),
    "eps_mean": args.eps_mean, "eps_max": args.eps_max, "argmax_mismatch_max": args.argmax_mismatch_max,
    "baseline_mean_nll": baseline["mean_nll"], "graph_mean_nll": graph["mean_nll"], "mean_dNLL": mean_dnll,
    "max_abs_dNLL": max_abs_dnll, "max_positive_dNLL": max_pos_dnll,
    "argmax_mismatches": argmax_mismatches, "retry_failures": retry_failures, "rows": rows, "gates": gates,
    "quality_boundary": "corpus-style sampled positions only; full retained prompt-logits perplexity remains out of scope",
  }
  out = pathlib.Path(args.artifact)
  if not out.is_absolute(): out = ROOT / out
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  try: out_name = str(out.relative_to(ROOT))
  except ValueError: out_name = str(out)
  print(json.dumps({"verdict": verdict, "gate_pass": result["gate_pass"], "scored_positions": len(rows),
                    "mean_dNLL": mean_dnll, "max_abs_dNLL": max_abs_dnll,
                    "argmax_mismatches": argmax_mismatches, "gates": gates,
                    "out": out_name}, indent=2), file=sys.__stdout__)
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
