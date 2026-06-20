#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, pathlib, platform, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_gemm_quality_sampled_result.json"


def _git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def _logsumexp(xs) -> float:
  import numpy as np
  m = float(np.max(xs))
  return m + math.log(float(np.exp(xs - m).sum()))


def _child_eval(args: argparse.Namespace) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_prefill_v2_nll_eval import CALIB_TEXT

  Tensor.manual_seed(args.seed)
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  if not getattr(model, "_pf16_warmstart", None):
    raise RuntimeError("PREFILL_V2 warmstart missing; run with PREFILL_V2=1")
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  need = args.stride * (args.windows - 1) + args.ubatch
  if len(ids) < need:
    raise ValueError(f"calibration too short: have {len(ids)} tokens, need {need}")

  rows = []
  for w in range(args.windows):
    start = w * args.stride
    win = ids[start:start + args.ubatch]
    # Keep T=512 so the graph GEMM route is exercised, but slice before realize so only one vocab vector is
    # materialized. Position -2 predicts the final token in the 512-token window.
    t = Tensor([win], dtype="int32").contiguous()
    logits = model.logits(t, 0)[:, -2, :].realize()[0].numpy()
    target = int(win[-1])
    nll = _logsumexp(logits) - float(logits[target])
    pred = int(np.argmax(logits))
    rows.append({
      "window": w, "start": start, "ubatch": args.ubatch, "scored_position": args.ubatch - 2,
      "target": target, "argmax": pred, "hit": pred == target, "nll": round(float(nll), 6),
      "finite": bool(np.isfinite(logits).all()),
    })
    print(f"{args.child_label} window {w}: nll={nll:.6f} argmax={pred} target={target}", file=sys.__stdout__)
  return {
    "label": args.child_label,
    "prefill_graph_gemm": bool(int(os.environ.get("PREFILL_GRAPH_GEMM", "0"))),
    "rows": rows,
    "mean_nll": round(sum(r["nll"] for r in rows) / len(rows), 6),
  }


def _run_child(model: str, label: str, graph: bool, args: argparse.Namespace) -> dict[str, Any]:
  env = {**os.environ, "DEV": os.environ.get("DEV", "AMD"), "PREFILL_V2": "1", "PYTHONPATH": "."}
  if graph: env["PREFILL_GRAPH_GEMM"] = "1"
  else: env.pop("PREFILL_GRAPH_GEMM", None)
  cmd = [
    sys.executable, str(pathlib.Path(__file__).resolve()), "--child", "--child-label", label,
    "--model", model, "--max-context", str(args.max_context), "--ubatch", str(args.ubatch),
    "--windows", str(args.windows), "--stride", str(args.stride), "--seed", str(args.seed),
  ]
  last: subprocess.CompletedProcess[str] | None = None
  for attempt in range(args.retries + 1):
    last = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if last.returncode == 0: break
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
    if line.startswith("{") and line.endswith("}"): return json.loads(line)
  raise RuntimeError(f"{label} child produced no JSON result")


def main() -> int:
  ap = argparse.ArgumentParser(description="VRAM-safe sampled quality gate for PREFILL_GRAPH_GEMM")
  ap.add_argument("pos_model", nargs="?")
  ap.add_argument("--model", default=None)
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--ubatch", type=int, default=512)
  ap.add_argument("--windows", type=int, default=1)
  ap.add_argument("--stride", type=int, default=256)
  ap.add_argument("--eps", type=float, default=0.01)
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
    print("ERROR: this probe intentionally keeps ubatch=512 so PREFILL_GRAPH_GEMM routing is exercised", file=sys.__stdout__)
    return 2
  if args.child:
    res = _child_eval(args)
    print(json.dumps(res, sort_keys=True), file=sys.__stdout__)
    return 0

  baseline = _run_child(args.model, "baseline_prefill_v2", False, args)
  graph = _run_child(args.model, "graph_gemm", True, args)
  rows = []
  for brow, grow in zip(baseline["rows"], graph["rows"]):
    dnll = round(grow["nll"] - brow["nll"], 6)
    rows.append({**grow, "baseline_nll": brow["nll"], "graph_nll": grow["nll"], "dNLL": dnll,
                 "argmax_match_baseline": grow["argmax"] == brow["argmax"]})
  max_abs_dnll = max(abs(r["dNLL"]) for r in rows)
  max_pos_dnll = max(r["dNLL"] for r in rows)
  gates = {
    "baseline_finite": all(r["finite"] for r in baseline["rows"]),
    "graph_finite": all(r["finite"] for r in graph["rows"]),
    "abs_dNLL_lte_eps": max_abs_dnll <= args.eps,
  }
  verdict = "PASS_PREFILL_GRAPH_GEMM_SAMPLED_QUALITY" if all(gates.values()) else "BLOCKED_PREFILL_GRAPH_GEMM_SAMPLED_QUALITY"
  result = {
    "date": "2026-06-20", "phase": "PREFILL_GRAPH_GEMM_SAMPLED_QUALITY",
    "schema": "prefill_graph_gemm_sampled_quality_v1", "verdict": verdict,
    "gate_pass": all(gates.values()), "model_id": pathlib.Path(args.model).name,
    "hardware": platform.node(), "commit": _git_sha(),
    "ubatch": args.ubatch, "windows": args.windows, "stride": args.stride, "eps": args.eps,
    "baseline_mean_nll": baseline["mean_nll"], "graph_mean_nll": graph["mean_nll"],
    "mean_dNLL": round(graph["mean_nll"] - baseline["mean_nll"], 6),
    "max_abs_dNLL": max_abs_dnll, "max_positive_dNLL": max_pos_dnll,
    "rows": rows, "gates": gates,
    "quality_boundary": "sampled final-token NLL only; full-window NLL remains VRAM-heavy in the older harness",
  }
  out = pathlib.Path(args.artifact)
  if not out.is_absolute(): out = ROOT / out
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": verdict, "gate_pass": result["gate_pass"], "mean_dNLL": result["mean_dNLL"],
                    "max_abs_dNLL": max_abs_dnll, "gates": gates, "out": str(out.relative_to(ROOT))}, indent=2),
        file=sys.__stdout__)
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
