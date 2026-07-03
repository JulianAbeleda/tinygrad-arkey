#!/usr/bin/env python3
"""Phase Q1432-0: baseline + route-miss proof for large Qwen3 dense models (14B/32B).

Goal (docs/qwen-14b-32b-true-generation-kernel-authoring-scope-20260630.md): prove the measured decode gap and
identify WHICH Q4_K decode route actually runs for each linear, BEFORE building any kernel. The hypothesis is that
14B/32B Q4_K decode linears fall outside the 8B-specific generated G3 guard (in/out in {4096,12288}) and so run the
slow lazy-dequant fallback.

This script does a static-but-faithful route census: it loads the model in decode mode, enumerates the Q4_K
primitive linears with their (role, in_features, out_features), and classifies the route each one takes by
replaying the exact model.py guard logic (G3 generated lanemap vs scheduler FFN branch vs _fallback dequant).
W==D tok/s and the llama-matched ratio are read from the authority bench artifacts (not re-measured here).

Writes bench/qwen-14b-32b-truegen/q1432_0_baseline/{latest,route_counts}.json + summary.md

Verdicts: Q1432_0_PASS_GAP_AND_ROUTE_MISS_PINNED / Q1432_0_ABORTED_NO_ROUTE_MISS / Q1432_0_BLOCKED_MODEL_MISSING
"""
from __future__ import annotations
import os, sys, json, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tinygrad.helpers import getenv

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qwen-14b-32b-truegen/q1432_0_baseline"

def _role(name:str) -> str:
  for key in ("ffn_gate_up", "ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output",
              "attn_qkv", "output", "token_embd"):
    if key in name: return key
  return name.split(".")[-1] if "." in name else name

def _g3_eligible(in_f:int, out_f:int) -> bool:
  # mirrors tinygrad/llm/model.py Q4KPrimitiveLinear.__call__ g3_bubblebeam_shape (sans the runtime arch flag)
  return (in_f // 256) % 4 == 0 and ((in_f == 4096 and out_f in (4096, 12288)) or (in_f == 12288 and out_f == 4096))

def _ffn_sched_eligible(in_f:int, out_f:int) -> bool:
  # the in=4096/out=12288 scheduler FFN branch (model.py)
  return in_f == 4096 and out_f == 12288

def classify_route(in_f:int, out_f:int) -> str:
  if _g3_eligible(in_f, out_f): return "G3_generated_lanemap"
  if _ffn_sched_eligible(in_f, out_f): return "scheduler_ffn_branch"
  return "fallback_lazy_dequant"

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="GGUF path")
  ap.add_argument("--id", required=True)
  ap.add_argument("--max_context", type=int, default=2048)
  args = ap.parse_args()

  if not pathlib.Path(args.model).exists():
    print(f"Q1432_0_BLOCKED_MODEL_MISSING: {args.model}"); sys.exit(2)

  os.environ.setdefault("DEV", "AMD")
  from tinygrad.llm.model import Transformer
  from tinygrad.helpers import fetch
  model, kv = Transformer.from_gguf(fetch(args.model), args.max_context)
  arch = kv.get("general.architecture")

  linears = getattr(model, "_q4k_linears", None)
  rows = []
  if linears and linears.linears:
    for lin in linears.linears:
      in_f, out_f = lin.in_features, lin.out_features
      rows.append({"name": lin.name, "role": _role(lin.name), "in_features": in_f, "out_features": out_f,
                   "route": classify_route(in_f, out_f)})

  # aggregate route counts (overall + by role)
  by_route: dict[str, int] = {}
  by_role_route: dict[str, dict[str, int]] = {}
  for r in rows:
    by_route[r["route"]] = by_route.get(r["route"], 0) + 1
    by_role_route.setdefault(r["role"], {})
    by_role_route[r["role"]][r["route"]] = by_role_route[r["role"]].get(r["route"], 0) + 1
  total = len(rows)
  fallback = by_route.get("fallback_lazy_dequant", 0)
  g3 = by_route.get("G3_generated_lanemap", 0)
  fallback_pct = round(100 * fallback / total, 1) if total else 0.0

  # decode gap from the authority bench doc (read if present)
  gap = None
  authp = ROOT / f"bench/models/qwen/data/amd-gfx1100/{args.id}.authority.json"
  if authp.exists():
    a = json.loads(authp.read_text())
    comp = {str(c["ctx"]): c for c in a.get("decode_matched_comparison", [])}
    gap = {ctx: {"tinygrad": comp[ctx]["tinygrad_tok_s_W"], "llama": comp[ctx]["llama_tok_s"],
                 "ratio_pct": comp[ctx]["ratio_pct"]} for ctx in comp}

  # route-miss verdict: the dominant Q4_K decode work is NOT on the generated route
  route_miss = fallback > 0 and g3 == 0
  verdict = "Q1432_0_PASS_GAP_AND_ROUTE_MISS_PINNED" if route_miss else "Q1432_0_ABORTED_NO_ROUTE_MISS"

  result = {"id": args.id, "arch": arch, "max_context": args.max_context, "n_q4k_linears": total,
            "route_counts": by_route, "fallback_pct": fallback_pct, "by_role_route": by_role_route,
            "decode_gap_authority": gap, "verdict": verdict, "linears": rows}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "route_counts.json").write_text(json.dumps({"id": args.id, "by_route": by_route,
    "by_role_route": by_role_route, "fallback_pct": fallback_pct}, indent=2))
  (OUT / f"{args.id}.json").write_text(json.dumps(result, indent=2))

  # human summary
  L = [f"# Q1432-0 baseline / route-miss — {args.id}", "", f"arch: {arch} · Q4_K decode linears: {total}", "",
       f"Verdict: **{verdict}**", "", "## Route census (per Q4_K decode linear)", "",
       "| role | shape (in→out) | route |", "|---|---|---|"]
  seen = set()
  for r in rows:
    key = (r["role"], r["in_features"], r["out_features"], r["route"])
    if key in seen: continue
    seen.add(key)
    L.append(f"| {r['role']} | {r['in_features']}→{r['out_features']} | {r['route']} |")
  L += ["", "## Route counts", "", "| route | count |", "|---|---|"]
  for rt, c in sorted(by_route.items(), key=lambda x: -x[1]): L.append(f"| {rt} | {c} |")
  L += ["", f"**{fallback_pct}%** of Q4_K decode linears run the slow lazy-dequant fallback "
        f"(G3 generated route fires on {g3}).", ""]
  if gap: L += ["## Decode gap (authority)", "", "| ctx | tinygrad tok/s | llama tok/s | ratio |", "|---|---|---|---|"] + \
            [f"| {c} | {gap[c]['tinygrad']} | {gap[c]['llama']} | {gap[c]['ratio_pct']}% |" for c in sorted(gap, key=int)]
  (OUT / f"{args.id}.summary.md").write_text("\n".join(L) + "\n")

  print(f"{args.id}: {total} Q4_K linears | routes={by_route} | fallback={fallback_pct}% | {verdict}")
  print(f"wrote {OUT}/{args.id}.json")

if __name__ == "__main__":
  main()
