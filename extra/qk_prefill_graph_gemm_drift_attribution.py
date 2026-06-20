#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, pathlib, platform, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_gemm_drift_attribution_result.json"
ALL_ROLES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
VARIANTS = {
  "attn_qkv": ("attn_q", "attn_k", "attn_v"),
  "attn_output": ("attn_output",),
  "attention_all": ("attn_q", "attn_k", "attn_v", "attn_output"),
  "ffn_gateup": ("ffn_gate", "ffn_up"),
  "ffn_down": ("ffn_down",),
  "ffn_all": ("ffn_gate", "ffn_up", "ffn_down"),
  "all": ALL_ROLES,
}


def _git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def _parse_pairs(s: str) -> list[tuple[int, int]]:
  out: list[tuple[int, int]] = []
  for part in s.split(","):
    if not part.strip(): continue
    w, o = part.split(":")
    wi, oi = int(w), int(o)
    if wi < 0 or oi < 0 or oi >= 511: raise ValueError(f"bad pair {part!r}")
    out.append((wi, oi))
  if not out: raise ValueError("--pairs must include at least one window:offset pair")
  return out


def _logsumexp(xs) -> float:
  import numpy as np
  m = float(np.max(xs))
  return m + math.log(float(np.exp(xs - m).sum()))


def _tag_roles(model: Any) -> None:
  for block in model.blk:
    for role in ALL_ROLES:
      lin = getattr(block, role, None)
      if lin is not None: lin._prefill_graph_role = role


def _score_position(model: Any, win: list[int], offset: int) -> dict[str, Any]:
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
  pred = int(np.argmax(logits))
  sorted_logits = np.sort(logits)
  margin = float(sorted_logits[-1] - sorted_logits[-2]) if sorted_logits.size >= 2 else 0.0
  return {"target": target, "argmax": pred, "hit": pred == target, "nll": round(float(nll), 6),
          "finite": bool(np.isfinite(logits).all()), "argmax_margin": round(margin, 6)}


def _child_eval(args: argparse.Namespace) -> dict[str, Any]:
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_prefill_v2_nll_eval import CALIB_TEXT

  pairs = _parse_pairs(args.pairs)
  Tensor.manual_seed(args.seed)
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  if not getattr(model, "_pf16_warmstart", None):
    raise RuntimeError("PREFILL_V2 warmstart missing; run with PREFILL_V2=1")
  _tag_roles(model)

  prefix = tok.prefix() if hasattr(tok, "prefix") else []
  body = tok.encode(CALIB_TEXT)
  need = max(w for w, _ in pairs) * args.stride + args.ubatch
  ids = prefix + body
  while len(ids) < need: ids += body

  rows = []
  for window, offset in pairs:
    start = window * args.stride
    row = _score_position(model, ids[start:start + args.ubatch], offset)
    row.update({"window": window, "start": start, "score_offset": offset})
    rows.append(row)
    print(f"{args.child_label} window {window} offset {offset}: nll={row['nll']:.6f} "
          f"argmax={row['argmax']} target={row['target']} margin={row['argmax_margin']:.6f}", file=sys.__stdout__)
  return {"label": args.child_label, "roles": os.environ.get("PREFILL_GRAPH_GEMM_ROLES", ""), "rows": rows}


def _run_child(model: str, label: str, roles: tuple[str, ...], args: argparse.Namespace) -> dict[str, Any]:
  env = {**os.environ, "DEV": os.environ.get("DEV", "AMD"), "PREFILL_V2": "1", "PYTHONPATH": "."}
  if roles:
    env["PREFILL_GRAPH_GEMM"] = "1"
    env["PREFILL_GRAPH_GEMM_ROLES"] = ",".join(roles)
  else:
    env.pop("PREFILL_GRAPH_GEMM", None)
    env.pop("PREFILL_GRAPH_GEMM_ROLES", None)
  cmd = [
    sys.executable, str(pathlib.Path(__file__).resolve()), "--child", "--child-label", label,
    "--model", model, "--max-context", str(args.max_context), "--ubatch", str(args.ubatch),
    "--stride", str(args.stride), "--pairs", args.pairs, "--seed", str(args.seed),
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
  if last.returncode != 0: raise RuntimeError(f"{label} child failed with code {last.returncode}")
  print(last.stdout, file=sys.__stdout__)
  for line in reversed(last.stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"): return json.loads(line)
  raise RuntimeError(f"{label} child produced no JSON result")


def _summarize_variant(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
  mean = round(sum(r["dNLL"] for r in rows) / len(rows), 6)
  max_abs = max(abs(r["dNLL"]) for r in rows)
  max_pos = max(r["dNLL"] for r in rows)
  mismatches = sum(0 if r["argmax_match_baseline"] else 1 for r in rows)
  return {"variant": name, "mean_dNLL": mean, "max_abs_dNLL": max_abs,
          "max_positive_dNLL": max_pos, "argmax_mismatches": mismatches}


def main() -> int:
  ap = argparse.ArgumentParser(description="Role/confidence attribution for PREFILL_GRAPH_GEMM NLL drift")
  ap.add_argument("pos_model", nargs="?")
  ap.add_argument("--model", default=None)
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--ubatch", type=int, default=512)
  ap.add_argument("--stride", type=int, default=64)
  ap.add_argument("--pairs", default="6:384,3:256,3:510,2:256,5:128,0:510")
  ap.add_argument("--variants", default="attn_qkv,attn_output,attention_all,ffn_gateup,ffn_down,ffn_all,all")
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
  _parse_pairs(args.pairs)
  if args.ubatch != 512:
    print("ERROR: ubatch must be 512 so graph route is exercised", file=sys.__stdout__)
    return 2
  if args.child:
    res = _child_eval(args)
    print(json.dumps(res, sort_keys=True), file=sys.__stdout__)
    return 0

  variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
  for v in variant_names:
    if v not in VARIANTS: raise ValueError(f"unknown variant {v!r}; known={sorted(VARIANTS)}")

  baseline = _run_child(args.model, "baseline_prefill_v2", (), args)
  results = []
  for v in variant_names:
    child = _run_child(args.model, v, VARIANTS[v], args)
    rows = []
    for brow, grow in zip(baseline["rows"], child["rows"]):
      if (brow["window"], brow["score_offset"]) != (grow["window"], grow["score_offset"]):
        raise RuntimeError("row mismatch")
      dnll = round(grow["nll"] - brow["nll"], 6)
      rows.append({**grow, "baseline_nll": brow["nll"], "graph_nll": grow["nll"], "dNLL": dnll,
                   "baseline_argmax": brow["argmax"], "argmax_match_baseline": grow["argmax"] == brow["argmax"],
                   "baseline_argmax_margin": brow["argmax_margin"], "graph_argmax_margin": grow["argmax_margin"]})
    results.append({"variant": v, "roles": list(VARIANTS[v]), "summary": _summarize_variant(v, rows), "rows": rows})

  verdict = "PASS_PREFILL_GRAPH_GEMM_DRIFT_ATTRIBUTION"
  result = {
    "date": "2026-06-20", "phase": "PREFILL_GRAPH_GEMM_DRIFT_ATTRIBUTION",
    "schema": "prefill_graph_gemm_drift_attribution_v1", "verdict": verdict,
    "model_id": pathlib.Path(args.model).name, "hardware": platform.node(), "commit": _git_sha(),
    "pairs": [{"window": w, "score_offset": o} for w, o in _parse_pairs(args.pairs)],
    "baseline_rows": baseline["rows"], "variants": results,
    "interpretation_boundary": "role-attribution over selected high-drift corpus rows, not a promotion gate",
  }
  out = pathlib.Path(args.artifact)
  if not out.is_absolute(): out = ROOT / out
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  try: out_name = str(out.relative_to(ROOT))
  except ValueError: out_name = str(out)
  print(json.dumps({"verdict": verdict, "summaries": [r["summary"] for r in results], "out": out_name}, indent=2),
        file=sys.__stdout__)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
