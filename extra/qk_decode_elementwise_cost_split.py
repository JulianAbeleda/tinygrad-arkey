#!/usr/bin/env python3
"""Deliverable 3 of docs/decode-attention-elementwise-solution-scope-20260620.md: split current-route decode
ELEMENTWISE cost into actionable buckets, timed (ProfileGraphEvent), per ctx/mode. Confirm whether
`E_49152_32_3` (the FFN `silu(gate)*up` activation) owns ~1.4ms/token.

Reuses the clock-pinned capture from extra/qk_decode_attention_cost_split.py (same two-layer instrument:
clean W==D wall + per-kernel warm GPU timestamps, rescaled to wall; peak clock pinned, auto restored).

Elementwise buckets (scope Deliverable 3):
  ffn_activation   = silu(gate)*up, esp. E_49152_32_3 (ffn dim 12288, ×4 lanes = 49152)
  rope             = RoPE rotary elementwise (q/k, chunk-by-2 cos/sin layout)
  residual_add     = block residual additions (hidden dim 4096 = 32*32*4)
  casts_copies     = cast / copy / layout cleanup
  unclassified_elementwise = must be small / explained (top_programs reported for transparency)

Run:
  PYTHONPATH=. python3 extra/qk_decode_elementwise_cost_split.py \
    --modes baseline,q8 --ckpts 512 1024 4096 --nmeas 20 --warmups 8 \
    --out bench/qk-decode-attention-elementwise/elementwise_cost_split.json
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, re, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-elementwise/elementwise_cost_split.json"
from extra.qk_decode_current_route_attribution import classify, clock_sample, git_sha, rel
from extra.qk_decode_attention_cost_split import capture_token_programs, NMEAS_HOLDER
from extra.qk_clock_pin import pinned_peak

# --- elementwise sub-bucket classifier (applied only to kernels the main classifier calls 'elementwise') ----------
def elem_bucket(name: str) -> str:
  n = name.lower(); nums = set(int(x) for x in re.findall(r"\d+", name))
  if 49152 in nums or 12288 in nums: return "ffn_activation"      # silu(gate)*up over ffn dim
  if n.startswith("copy") or "cast" in n: return "casts_copies"
  if re.match(r"e_2_\d", n): return "rope"                         # rotary: chunk-by-2 cos/sin halves
  if re.match(r"e_32_32_4", n): return "residual_add"             # hidden-dim 4096 add (x + sublayer)
  # per-layer GEMV output reshape/cast glue: E_128_32_3 (1->1, dim 4096=128*32, in role_atlas as ffn/attn glue),
  # E_1536_32_3 (per-layer layout glue). Small 1-in/1-out layout kernels, not a build target.
  if re.match(r"e_128_32", n) or re.match(r"e_1536_32", n): return "casts_copies"
  return "unclassified_elementwise"

def run_child(args: argparse.Namespace) -> int:
  from tinygrad import Tensor, UOp, Device
  from extra.llm_generate import load_model_and_tokenizer
  NMEAS_HOLDER[0] = args.nmeas
  dev = Device[Device.DEFAULT]
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  for lin in (getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + args.max_context // max(1, len(ids))))[:args.max_context]
  v_sp = UOp.variable("start_pos", 0, args.max_context - 1); temp = Tensor([0.0])
  rows = []
  with pinned_peak(enabled=not args.no_pin) as pin_prov:
    time.sleep(0.5)
    for ck in args.ckpts:
      for b in model.blk: b._prefill_v2 = False
      w_ms, d_ms, per, busy = capture_token_programs(model, dev, int(ids[ck]), v_sp, temp, ck, args.warmups)
      scale = w_ms / (busy / 1000.0) if busy > 0 else 0.0
      elem = {nm: v for nm, v in per.items() if classify(nm)[0] == "elementwise"}
      bucket = collections.defaultdict(lambda: {"calls": 0, "us": 0.0, "progs": collections.defaultdict(lambda: [0, 0.0])})
      for nm, v in elem.items():
        b = elem_bucket(nm); bucket[b]["calls"] += v["calls"]; bucket[b]["us"] += v["us"]
        bucket[b]["progs"][nm][0] += v["calls"]; bucket[b]["progs"][nm][1] += v["us"]
      elem_us = sum(v["us"] for v in elem.values())
      buckets = {}
      for b, v in bucket.items():
        tops = sorted(v["progs"].items(), key=lambda kv: -kv[1][1])[:4]
        buckets[b] = {"calls_per_token": v["calls"], "ms_per_token": round(v["us"] / 1000.0 * scale, 4),
                      "pct_of_elementwise": round(100 * v["us"] / elem_us, 1) if elem_us else 0,
                      "pct_of_wall": round(100 * (v["us"] / 1000.0 * scale) / w_ms, 2) if w_ms else 0,
                      "top_programs": [{"name": n, "calls": c, "ms": round(u / 1000.0 * scale, 4)} for n, (c, u) in tops]}
      e49152 = next((v["us"] / 1000.0 * scale for nm, v in elem.items() if "49152" in nm), 0.0)
      rows.append({"ctx": ck, "wall_ms_W": round(w_ms, 3), "tok_s_W": round(1000 / w_ms, 1),
                   "elementwise_ms_per_token": round(elem_us / 1000.0 * scale, 4),
                   "elementwise_pct_of_wall": round(100 * (elem_us / 1000.0 * scale) / w_ms, 2) if w_ms else 0,
                   "E_49152_ms_per_token": round(e49152, 4), "rescale_factor": round(scale, 4), "buckets": buckets})
      print(f"  [{args.mode}] ctx {ck:5}: wall {w_ms:.2f}ms elem {elem_us/1000*scale:.2f}ms E_49152 {e49152:.2f}ms | "
            + " ".join(f"{b}={v['ms_per_token']:.2f}" for b, v in
            sorted(buckets.items(), key=lambda kv: -kv[1]['ms_per_token'])), file=sys.__stderr__)
  result = {"phase": "DECODE_ELEMENTWISE_COST_SPLIT_CHILD", "schema": "decode_elementwise_cost_split_child_v1",
            "mode": args.mode, "q8_enabled": args.mode == "q8", "commit": git_sha(),
            "model_id": pathlib.Path(args.model).stem, "ckpts": args.ckpts, "nmeas": args.nmeas,
            "clock_pinned": not args.no_pin, "clock_pin_prov": pin_prov,
            "clock_provenance": [clock_sample()], "rows": rows, "default_behavior_changed": False}
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"mode": args.mode, "out": rel(args.child_out)}))
  return 0

def aggregate(args: argparse.Namespace, children) -> int:
  per_mode = {}
  for mode in args.modes.split(","):
    f = args.out.parent / f"elementwise_cost_split_{mode}.json"
    if f.exists(): per_mode[mode] = json.loads(f.read_text())
  BUCKETS = ["ffn_activation", "rope", "residual_add", "casts_copies", "unclassified_elementwise"]
  tables = {}
  for mode, child in per_mode.items():
    for r in child["rows"]:
      key = f"{mode}@{r['ctx']}"
      bk = {b: r["buckets"].get(b, {"ms_per_token": 0.0, "pct_of_elementwise": 0, "pct_of_wall": 0,
            "calls_per_token": 0, "top_programs": []}) for b in BUCKETS}
      classified = sum(v["ms_per_token"] for b, v in bk.items() if b != "unclassified_elementwise")
      tables[key] = {"wall_ms": r["wall_ms_W"], "tok_s_W": r["tok_s_W"],
                     "elementwise_ms": r["elementwise_ms_per_token"], "elementwise_pct_wall": r["elementwise_pct_of_wall"],
                     "E_49152_ms": r["E_49152_ms_per_token"], "buckets": bk,
                     "pct_classified": round(100 * classified / r["elementwise_ms_per_token"], 1) if r["elementwise_ms_per_token"] else 100.0}
  # families >=0.25 ms/token at any cell
  families = collections.defaultdict(float)
  for key, t in tables.items():
    for b, v in t["buckets"].items():
      for p in v["top_programs"]: families[p["name"]] = max(families[p["name"]], p["ms"])
  big_families = {k: round(v, 4) for k, v in sorted(families.items(), key=lambda kv: -kv[1]) if v >= 0.25}
  e49152_1024 = tables.get("baseline@1024", {}).get("E_49152_ms", 0.0)
  gates = {
    "classifies_ge_90pct_all_cells": all(t["pct_classified"] >= 90 for t in tables.values()),
    "found_family_ge_0p25ms": len(big_families) > 0,
    "E_49152_owns_~1p4ms_at_1024": 1.1 <= e49152_1024 <= 1.7,
  }
  result = {"date": "2026-06-20", "phase": "DECODE_ELEMENTWISE_COST_SPLIT",
            "schema": "decode_elementwise_cost_split_v1", "commit": git_sha(),
            "model_id": "Qwen3-8B-Q4_K_M", "modes": list(per_mode), "ckpts": args.ckpts, "nmeas": args.nmeas,
            "method": "ProfileGraphEvent timed split (rescaled to clean W wall, peak clock pinned); elementwise sub-buckets",
            "tables": tables, "families_ge_0p25ms": big_families, "E_49152_ms_at_baseline_1024": e49152_1024,
            "gates": gates, "gate_pass": gates["classifies_ge_90pct_all_cells"] and gates["found_family_ge_0p25ms"],
            "clock_provenance": {m: per_mode[m].get("clock_provenance") for m in per_mode},
            "children": children, "default_behavior_changed": False}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"gate_pass": result["gate_pass"], "gates": gates, "E_49152_ms_at_1024": e49152_1024,
                    "families_ge_0p25ms": list(big_families)[:8], "out": rel(args.out)}, indent=2))
  return 0

def run_parent(args: argparse.Namespace) -> int:
  children = []
  for mode in args.modes.split(","):
    out = args.out.parent / f"elementwise_cost_split_{mode}.json"
    env = os.environ.copy(); env.setdefault("DEV", "AMD"); env.setdefault("JIT", "1"); env["PYTHONPATH"] = str(ROOT)
    if mode == "q8": env["Q8_FFN_HANDWRITTEN"] = "1"
    else: env.pop("Q8_FFN_HANDWRITTEN", None)
    cmd = [sys.executable, rel(pathlib.Path(__file__).resolve()), "--child-out", rel(out), "--mode", mode,
           "--nmeas", str(args.nmeas), "--warmups", str(args.warmups), "--model", args.model,
           "--max-context", str(args.max_context), "--seed", str(args.seed),
           *(["--no-pin"] if args.no_pin else []), "--ckpts", *[str(x) for x in args.ckpts]]
    print(f"[parent] elementwise split child mode={mode}", file=sys.__stderr__)
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=None)
    children.append({"mode": mode, "returncode": p.returncode, "stdout": (p.stdout or "")[-2000:], "artifact": rel(out)})
  return aggregate(args, children)

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--modes", default="baseline,q8")
  ap.add_argument("--ckpts", nargs="+", type=int, default=[512, 1024, 4096])
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--seed", type=int, default=20260620)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--aggregate-existing", action="store_true")
  ap.add_argument("--no-pin", action="store_true")
  ap.add_argument("--child-out", type=pathlib.Path)
  ap.add_argument("--mode", choices=["baseline", "q8"], default="baseline")
  args = ap.parse_args()
  if args.child_out is not None: return run_child(args)
  if args.aggregate_existing: return aggregate(args, [])
  return run_parent(args)

if __name__ == "__main__":
  raise SystemExit(main())
