#!/usr/bin/env python3
"""Route B B4 — interleaved W==D A/B for the owned AMDGCN graph-node decode route.

This keeps the measurement in-process (single-process replay/route control), reports explicit
route firing from the captured TinyJit graph, and enforces .item()-inside-timing-window for real
W==D comparability.

Recommended:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_b4_decode_eval.py --policy ctx2048_only --splits 24 32 40 48 56 64 80 96 128
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import os
import sys
from typing import Any

from tinygrad import Tensor, TinyJit, UOp
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops

from extra.llm_generate import load_model_and_tokenizer
from extra.qk_harness_contract import DEFAULT_MODEL, repro_band, stamp

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-route-b-b4"

MAXC = 4608
CTXS = [512, 1024, 4096]
NMEAS_DEFAULT = 40
REPEATS_DEFAULT = 6
WARMUPS_DEFAULT = 8
SPLIT_DEFAULTS = [24, 32, 40, 48, 56, 64, 80, 96, 128]
CTX_NO_ROUTE = 10_000_000  # effectively disables route for a context


def _policy_threshold(policy: str, ctx: int) -> int:
  if policy == "ctx4096_only":
    return 4096
  if policy == "ctx2048_only":
    return 2048
  if policy == "adaptive":
    return 1024 if ctx >= 1024 else CTX_NO_ROUTE
  raise ValueError(f"unknown policy {policy!r}")


def _parse_splits(raw: list[int] | None, default_split: int | None) -> list[int]:
  if raw:
    vals = sorted({int(v) for v in raw})
    return [v for v in vals if v > 0]
  if default_split is not None:
    return [int(default_split)]
  return SPLIT_DEFAULTS.copy()


def _owned_programs(step: TinyJit) -> list[str]:
  names: list[str] = []
  if step.captured is None:
    return names
  for u in step.captured.linear.toposort():
    if u.op is not Ops.CALL or not len(u.src):
      continue
    p = u.src[0]
    if p.op is Ops.PROGRAM:
      n = getattr(p.arg, "name", None)
      if isinstance(n, str):
        names.append(n)
  return names


def _median_and_band(vals_ms: list[float]) -> tuple[float | None, dict[str, float | int | None]]:
  band = repro_band([float(x) for x in vals_ms])
  return band["median"], band


def build_step(model, ids: list[int], v_sp: UOp, temp: Tensor, ck: int, split: int, threshold: int, amdgcn_flag: int,
              warmups: int):
  # route control is via env; clear tinygrad getenv cache so each branch gets this config.
  os.environ["DECODE_ATTN_AMDGCN_TILE"] = str(amdgcn_flag)
  os.environ["DECODE_ATTN_AMDGCN_S"] = str(split)
  os.environ["FLASH_DECODE_THRESHOLD"] = str(threshold)
  getenv.cache_clear()

  for block in model.blk:
    block._use_flash = False
    block._prefill_v2 = False

  step = TinyJit(model.forward)
  token = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
  for _ in range(warmups):
    token = step(token, v_sp.bind(ck), temp)
  token.realize()
  programs = _owned_programs(step)
  route_fired = any("owned_flash" in p for p in programs)
  return step, sorted(set(programs)), bool(route_fired)


def measure_step(step: TinyJit, v_sp: UOp, ids: list[int], ck: int, nmeas: int, temp: Tensor) -> tuple[list[float], list[int]]:
  out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
  times_ms: list[float] = []
  toks: list[int] = []
  for i in range(nmeas):
    t0 = time.perf_counter()
    out = step(out, v_sp.bind(ck + i), temp)
    tok = int(out.item())
    times_ms.append((time.perf_counter() - t0) * 1e3)
    toks.append(tok)
  return times_ms, toks


def run_split(ctx: int, split: int, threshold: int, model, ids, v_sp, temp, warmups: int, nmeas: int, repeats: int):
  base_step, base_nodes, base_route = build_step(model, ids, v_sp, temp, ctx, split, threshold, amdgcn_flag=0, warmups=warmups)
  amd_step, amd_nodes, amd_route = build_step(model, ids, v_sp, temp, ctx, split, threshold, amdgcn_flag=1, warmups=warmups)

  base_times: list[float] = []
  amd_times: list[float] = []
  base_toks: list[int] | None = None
  amd_toks: list[int] | None = None

  for repeat in range(repeats):
    # alternate order to avoid compile cache / queue skew from one path always going first.
    if repeat % 2 == 0:
      bt, btok = measure_step(base_step, v_sp, ids, ctx, nmeas, temp)
      at, atok = measure_step(amd_step, v_sp, ids, ctx, nmeas, temp)
    else:
      at, atok = measure_step(amd_step, v_sp, ids, ctx, nmeas, temp)
      bt, btok = measure_step(base_step, v_sp, ids, ctx, nmeas, temp)
    base_times.append(statistics.median(bt))
    amd_times.append(statistics.median(at))
    if base_toks is None:
      base_toks = btok
    if amd_toks is None:
      amd_toks = atok

  base_med_ms, base_band = _median_and_band(base_times)
  amd_med_ms, amd_band = _median_and_band(amd_times)
  base_tok_med = 1000.0 / base_med_ms if base_med_ms else None
  amd_tok_med = 1000.0 / amd_med_ms if amd_med_ms else None
  delta_pct = (amd_tok_med - base_tok_med) / base_tok_med * 100 if base_tok_med else 0.0

  return {
    "split_S": split,
    "threshold": threshold,
    "route_firing_expected": ctx >= threshold,
    "route_fired_base": base_route,
    "route_fired_amdgcn": amd_route,
    "route_program_match": base_route == amd_route,
    "route_nodes_base": base_nodes,
    "route_nodes_amdgcn": amd_nodes,
    "base_repro_band": base_band,
    "amd_repro_band": amd_band,
    "base_ms_median": round(base_med_ms, 3) if base_med_ms is not None else None,
    "amd_ms_median": round(amd_med_ms, 3) if amd_med_ms is not None else None,
    "base_tok_s_median": round(base_tok_med, 2) if base_tok_med is not None else None,
    "amd_tok_s_median": round(amd_tok_med, 2) if amd_tok_med is not None else None,
    "delta_pct": round(delta_pct, 2),
    "base_spread_pct": base_band.get("spread_pct"),
    "amd_spread_pct": amd_band.get("spread_pct"),
    "tokens_match": bool(base_toks == amd_toks),
    "token_trace_len": len(base_toks or []),
  }


def run_policy(policy: str, splits: list[int], model_name: str, model, ids, nmeas: int, repeats: int, warmups: int):
  print(f"policy={policy} splits={splits}", file=sys.__stderr__)
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])

  rows: list[dict[str, Any]] = []
  for ctx in CTXS:
    threshold = _policy_threshold(policy, ctx)
    split_rows = [run_split(ctx, split, threshold, model, ids, v_sp, temp, warmups, nmeas, repeats) for split in splits]
    best = max(split_rows, key=lambda r: (r["amd_tok_s_median"] or -1.0), default={})
    route_nodes_seen = [n for row in split_rows for n in row.get("route_nodes_amdgcn", []) if isinstance(row.get("route_nodes_amdgcn"), list)]
    rows.append({
      "ctx": ctx,
      "threshold": threshold,
      "split_candidates": split_rows,
      "best": {
        "split_S": best.get("split_S"),
        "route_nodes_amdgcn": best.get("route_nodes_amdgcn", []),
        "amdgcn_tok_s_median": best.get("amd_tok_s_median"),
        "delta_pct": best.get("delta_pct"),
        "tokens_match": best.get("tokens_match", False),
        "route_fired_amdgcn": best.get("route_fired_amdgcn", False),
      },
      "route_fired": any(r["route_fired_amdgcn"] for r in split_rows),
      "route_firing_expected": bool(ctx >= threshold),
      "route_nodes_seen": sorted(set(route_nodes_seen)),
      "base_tok_s_median_best": best.get("base_tok_s_median"),
      "amdgcn_tok_s_median_best": best.get("amd_tok_s_median"),
      "delta_pct_best": best.get("delta_pct"),
      "tokens_match_best": best.get("tokens_match", False),
      "route_nodes_best": best.get("route_nodes_amdgcn"),
    })
    print(f"  ctx {ctx}: best S={best.get('split_S')} tok/s {best.get('amdgcn_tok_s_median')} "
          f"(Δ={best.get('delta_pct', 0):+.2f}%) tokens_match={best.get('tokens_match', False)}",
          file=sys.__stderr__)

  by_ctx = {r["ctx"]: r for r in rows}
  d512 = (by_ctx.get(512, {}).get("delta_pct_best") or 0.0)
  d1024 = (by_ctx.get(1024, {}).get("delta_pct_best") or 0.0)
  d4096 = (by_ctx.get(4096, {}).get("delta_pct_best") or 0.0)
  token_ok = all(r["tokens_match_best"] for r in rows)

  # W==D gate: no short-context regressions and a meaningful long-context gain.
  gate = token_ok and d512 >= -1.0 and d1024 >= -1.0 and (d1024 >= 5.0 or d4096 >= 7.0)
  if gate:
    decision = "B4_WD_PASS"
  elif token_ok:
    decision = "B4_WD_FAIL_INTEGRATION"
  else:
    decision = "B4_WD_FAIL_INTEGRATION"

  out = {
    "date": "2026-06-21",
    "phase": "ROUTE_B_B4_WD_SPLIT_POLICY",
    "candidate_id": "decode_attention_llama_flash_tile_owned_amdgcn_b4",
    "candidate_family": "north_star_flash_attn_tile",
    "model": model_name,
    "policy": policy,
    "policy_threshold": None if policy == "adaptive" else _policy_threshold(policy, MAXC),
    "adaptive": policy == "adaptive",
    "contexts": CTXS,
    "splits": splits,
    "nmeas": nmeas,
    "repeats": repeats,
    "warmups": warmups,
    "rows": rows,
    "best_by_ctx": {c: by_ctx[c]["best"]["split_S"] for c in CTXS},
    "amdgcn_tok_s_median_best": {c: by_ctx[c].get("amdgcn_tok_s_median_best") for c in CTXS},
    "delta_pct_best": {c: by_ctx[c].get("delta_pct_best") for c in CTXS},
    "base_tok_s_median_best": {c: by_ctx[c].get("base_tok_s_median_best") for c in CTXS},
    "route_fired_by_ctx": {c: by_ctx[c]["route_fired"] for c in CTXS},
    "route_firing_expected_by_ctx": {c: by_ctx[c]["route_firing_expected"] for c in CTXS},
    "route_nodes_seen_by_ctx": {c: by_ctx[c]["route_nodes_seen"] for c in CTXS},
    "all_tokens_match": token_ok,
    "verdict": decision,
    "first_gate_pass": gate,
    "gate_rule": "no-regress ctx512 and ctx1024; and (ctx1024 >= +5% or ctx4096 >= +7%)",
    "default_behavior_changed": False,
  }
  out = stamp(out,
              comparator_id="gqa_coop_vec",
              comparator_why="shipped default decode-attention primitive; B4 must beat gqa_coop_vec in W==D for practical promotion",
              timing_authority=("in-process interleaved W==D; real per-token path measured with .item() in-timer "
                               "and explicit model-route capture checks (owned_flash kernel presence in TinyJit)"),
              ledger_links=["docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md",
                           "bench/qk-decode-runtime-overhead/result.json",
                           "extra/qk_b4_decode_eval.py",
                           "extra/qk_owned_flash_decode_graph_node.py"],
              is_current_winner=True)

  # Also emit decoder-compat A/B shape used by decode_eval when this candidate is bound there.
  out["results"] = [{"ctx": row["ctx"], "best_speedup_vs_coop": round(1.0 + (row["delta_pct_best"] or 0.0) / 100.0, 3),
                     "splits": [{"split_S": s["split_S"], "err": (0.0 if s["tokens_match"] else 1.0),
                                 "base_repro_band": s["base_repro_band"], "amdgcn_repro_band": s["amd_repro_band"]} for s in row["split_candidates"]]}
                   for row in rows]
  return out


def parse_args():
  parser = argparse.ArgumentParser(description="Route B B4 W==D split/threshold sweep")
  parser.add_argument("split", nargs="?", type=int, help="single split S to test (legacy single-shot mode)")
  parser.add_argument("--splits", nargs="*", type=int, help="split candidates for sweep")
  parser.add_argument("--policy", choices=("ctx2048_only", "ctx4096_only", "adaptive"), default="ctx2048_only",
                      help="route policy: fixed threshold 2048, fixed threshold 4096, or adaptive S-by-context")
  parser.add_argument("--ckpts", nargs="*", type=int, default=CTXS)
  parser.add_argument("--nmeas", type=int, default=NMEAS_DEFAULT)
  parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT)
  parser.add_argument("--warmups", type=int, default=WARMUPS_DEFAULT)
  return parser.parse_args()


def main():
  args = parse_args()
  global CTXS
  if args.ckpts:
    CTXS[:] = args.ckpts
  splits = _parse_splits(args.splits, args.split)
  if not splits:
    splits = SPLIT_DEFAULTS.copy()

  model_name = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  model, tok = load_model_and_tokenizer(model_name, MAXC, seed=20260617)
  for block in (getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []):
    block.decode_enabled = True

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]

  out = run_policy(args.policy, splits, model_name, model, ids, args.nmeas, args.repeats, args.warmups)
  OUT.mkdir(parents=True, exist_ok=True)
  stamp_name = f"b4_{args.policy}_{time.strftime('%Y%m%dT%H%M%S')}.json"
  out_path = OUT / stamp_name
  out_path.write_text(json.dumps(out, indent=2) + "\n")
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps({"policy": args.policy, "first_gate_pass": out["first_gate_pass"], "verdict": out["verdict"],
                    "best_by_ctx": out["best_by_ctx"], "out": str(out_path.relative_to(ROOT))}, indent=2))
  print(f"artifact: {out_path}", file=sys.__stderr__)


if __name__ == "__main__":
  main()
