#!/usr/bin/env python3
"""Deliverable 1 of docs/decode-attention-elementwise-solution-scope-20260620.md: split current-route decode
ATTENTION cost into actionable buckets, timed (ProfileGraphEvent), per ctx/mode.

Method = same two-layer instrument as extra/qk_decode_current_route_attribution.py:
  W==D clean wall (PROFILE off) for the per-token wall authority;
  per-kernel warm GPU timestamps (PROFILE on, 2nd jit) for the timed attention split.
Per-token ms is rescaled onto the clean W wall by the timed busy-share. Default decode behavior NOT changed.

Attention buckets (scope Deliverable 1):
  partial_compute   = flash_partial_* (main attention compute)
  reduce_fixup      = flash-decode reduce kernels (r_* with start_pos / head_dim 128 / kv 1024)
  softmax_stats     = flash_prob/max/den/gmax/combine-style stat kernels
  qk_scores_other   = non-flash attention leftovers (SDPA qk scores, masks)
  unclassified_attention = must be small / explained

Run:
  PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py \
    --modes baseline,q8 --ckpts 512 1024 2048 4096 --nmeas 20 --warmups 8 \
    --out bench/qk-decode-attention-elementwise/attention_cost_split.json
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, re, statistics, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-elementwise/attention_cost_split.json"
from extra.qk_decode_current_route_attribution import classify, clock_sample, git_sha, rel  # reuse helpers
from extra.qk_clock_pin import pinned_peak

# --- attention sub-bucket classifier (only applied to kernels the main classifier calls attention_flash) ----------
def attn_bucket(name: str) -> str:
  n = name.lower()
  if "flash_partial" in n: return "partial_compute"
  if n.startswith("flash_"): return "softmax_stats"          # flash_prob/max/den/gmax/combine/gmax...
  if n.startswith("r_"): return "reduce_fixup"               # flash-decode reduce rows (start_pos/128/1024)
  if "start_pos" in n: return "reduce_fixup"
  if "sdpa" in n or "score" in n or "matmul" in n: return "qk_scores_other"
  return "unclassified_attention"

# ================================================================================================================
def capture_token_programs(model, dev, tokid, v_sp, temp, ck, warmups):
  """warm W==D (clean) + one bracketed PROFILE replay -> (w_ms, d_ms, programs[name]->(calls,us), busy_us)."""
  from tinygrad import Tensor, TinyJit, Context
  from tinygrad.device import Compiled
  step = TinyJit(model.forward)
  out = Tensor([[tokid]], dtype="int32").contiguous()
  for i in range(warmups): out = step(out, v_sp.bind(ck + i), temp).realize()
  out = Tensor([[tokid]], dtype="int32").contiguous(); Wl = []
  for i in range(NMEAS_HOLDER[0]):
    t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); _ = int(out.item()); Wl.append(time.perf_counter() - t0)
  out = Tensor([[tokid]], dtype="int32").contiguous(); dev.synchronize(); t0 = time.perf_counter()
  for i in range(NMEAS_HOLDER[0]): out = step(out, v_sp.bind(ck + i), temp)
  dev.synchronize(); d_ms = (time.perf_counter() - t0) / NMEAS_HOLDER[0] * 1e3
  w_ms = statistics.median(Wl) * 1e3
  per = collections.defaultdict(lambda: {"calls": 0, "us": 0.0}); busy = 0.0
  with Context(PROFILE=1):
    ps = TinyJit(model.forward); po = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(max(8, warmups)): po = ps(po, v_sp.bind(ck + i), temp).realize()
    dev.synchronize(); dev._at_profile_finalize(); base = len(Compiled.profile_events)
    po = ps(po, v_sp.bind(ck), temp).realize(); dev.synchronize(); dev._at_profile_finalize()
    for e in [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]:
      sigs = [float(s) for s in e.sigs]
      for ent in e.ents:
        per[str(ent.name)]["calls"] += 1; per[str(ent.name)]["us"] += sigs[ent.en_id] - sigs[ent.st_id]
    busy = sum(v["us"] for v in per.values())
  return w_ms, d_ms, dict(per), busy

NMEAS_HOLDER = [20]

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
    time.sleep(0.5)  # settle pinned clock
    for ck in args.ckpts:
      for b in model.blk: b._prefill_v2 = False
      w_ms, d_ms, per, busy = capture_token_programs(model, dev, int(ids[ck]), v_sp, temp, ck, args.warmups)
      scale = w_ms / (busy / 1000.0) if busy > 0 else 0.0
      # attention kernels = those the main classifier buckets as attention_flash
      attn = {nm: v for nm, v in per.items() if classify(nm)[0] == "attention_flash"}
      bucket = collections.defaultdict(lambda: {"calls": 0, "us": 0.0, "progs": collections.defaultdict(float)})
      for nm, v in attn.items():
        b = attn_bucket(nm); bucket[b]["calls"] += v["calls"]; bucket[b]["us"] += v["us"]; bucket[b]["progs"][nm] += v["us"]
      attn_us = sum(v["us"] for v in attn.values())
      buckets = {}
      for b, v in bucket.items():
        tops = sorted(v["progs"].items(), key=lambda kv: -kv[1])[:4]
        buckets[b] = {"calls_per_token": v["calls"], "ms_per_token": round(v["us"] / 1000.0 * scale, 4),
                      "raw_gpu_ms": round(v["us"] / 1000.0, 4),
                      "pct_of_attention": round(100 * v["us"] / attn_us, 1) if attn_us else 0,
                      "pct_of_wall": round(100 * (v["us"] / 1000.0 * scale) / w_ms, 2) if w_ms else 0,
                      "top_programs": [{"name": n, "ms": round(u / 1000.0 * scale, 4)} for n, u in tops]}
      rows.append({"ctx": ck, "wall_ms_W": round(w_ms, 3), "dispatch_ms_D": round(d_ms, 3),
                   "tok_s_W": round(1000 / w_ms, 1), "tok_s_D_ceiling": round(1000 / d_ms, 1),
                   "host_sync_pct": round(100 * max(0.0, w_ms - d_ms) / w_ms, 1),
                   "attention_ms_per_token": round(attn_us / 1000.0 * scale, 4),
                   "attention_pct_of_wall": round(100 * (attn_us / 1000.0 * scale) / w_ms, 2) if w_ms else 0,
                   "rescale_factor": round(scale, 4), "buckets": buckets})
      am = attn_us / 1000.0 * scale
      print(f"  [{args.mode}] ctx {ck:5}: wall {w_ms:.2f}ms ({1000/w_ms:.1f}t/s) attn {am:.2f}ms "
            f"({100*am/w_ms:.0f}%wall) | " + " ".join(f"{b}={v['ms_per_token']:.2f}" for b, v in
            sorted(buckets.items(), key=lambda kv: -kv[1]['ms_per_token'])), file=sys.__stderr__)
  result = {"phase": "DECODE_ATTENTION_COST_SPLIT_CHILD", "schema": "decode_attention_cost_split_child_v1",
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
    f = args.out.parent / f"attention_cost_split_{mode}.json"
    if f.exists(): per_mode[mode] = json.loads(f.read_text())
  BUCKETS = ["partial_compute", "reduce_fixup", "softmax_stats", "qk_scores_other", "unclassified_attention"]
  tables, gates_per = {}, {}
  for mode, child in per_mode.items():
    for r in child["rows"]:
      key = f"{mode}@{r['ctx']}"
      tables[key] = {"wall_ms": r["wall_ms_W"], "tok_s_W": r["tok_s_W"], "attention_ms": r["attention_ms_per_token"],
                     "attention_pct_wall": r["attention_pct_of_wall"],
                     "buckets": {b: r["buckets"].get(b, {"ms_per_token": 0.0, "pct_of_attention": 0,
                                 "pct_of_wall": 0, "calls_per_token": 0, "top_programs": []}) for b in BUCKETS}}
      classified = sum(v["ms_per_token"] for b, v in tables[key]["buckets"].items() if b != "unclassified_attention")
      tables[key]["pct_classified"] = round(100 * classified / r["attention_ms_per_token"], 1) if r["attention_ms_per_token"] else 100.0
  # ctx slope of attention + dominant bucket
  def biggest(mode, ck):
    t = tables.get(f"{mode}@{ck}");
    if not t: return None
    return max(t["buckets"].items(), key=lambda kv: kv[1]["ms_per_token"])
  dominant = {m: (biggest(m, 1024) or (None, {}))[0] for m in per_mode}
  slope = {m: {str(ck): tables[f"{m}@{ck}"]["attention_ms"] for ck in args.ckpts if f"{m}@{ck}" in tables} for m in per_mode}
  # gates
  b1024 = biggest("baseline", 1024); b4096 = biggest("baseline", 4096)
  gates = {
    "classifies_ge_90pct_all_cells": all(t["pct_classified"] >= 90 for t in tables.values()),
    "a_bucket_owns_ge_1ms_at_1024": bool(b1024 and b1024[1]["ms_per_token"] >= 1.0),
    "a_bucket_owns_ge_2ms_at_4096": bool(b4096 and b4096[1]["ms_per_token"] >= 2.0),
  }
  result = {"date": "2026-06-20", "phase": "DECODE_ATTENTION_COST_SPLIT",
            "schema": "decode_attention_cost_split_v1", "commit": git_sha(),
            "model_id": "Qwen3-8B-Q4_K_M", "hardware": "RX 7900 XTX / gfx1100",
            "modes": list(per_mode), "ckpts": args.ckpts, "nmeas": args.nmeas,
            "method": "ProfileGraphEvent timed split (rescaled to clean W wall); attention sub-buckets",
            "tables": tables, "attention_ms_slope": slope, "dominant_bucket_at_1024": dominant,
            "gates": gates, "gate_pass": gates["classifies_ge_90pct_all_cells"] and
            (gates["a_bucket_owns_ge_1ms_at_1024"] or gates["a_bucket_owns_ge_2ms_at_4096"]),
            "clock_provenance": {m: per_mode[m].get("clock_provenance") for m in per_mode},
            "children": children, "default_behavior_changed": False}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"gate_pass": result["gate_pass"], "gates": gates, "dominant_bucket_at_1024": dominant,
                    "out": rel(args.out)}, indent=2))
  return 0

def run_parent(args: argparse.Namespace) -> int:
  children = []
  for mode in args.modes.split(","):
    out = args.out.parent / f"attention_cost_split_{mode}.json"
    env = os.environ.copy(); env.setdefault("DEV", "AMD"); env.setdefault("JIT", "1"); env["PYTHONPATH"] = str(ROOT)
    if mode == "q8": env["Q8_FFN_HANDWRITTEN"] = "1"
    else: env.pop("Q8_FFN_HANDWRITTEN", None)
    cmd = [sys.executable, rel(pathlib.Path(__file__).resolve()), "--child-out", rel(out), "--mode", mode,
           "--nmeas", str(args.nmeas), "--warmups", str(args.warmups), "--model", args.model,
           "--max-context", str(args.max_context), "--seed", str(args.seed),
           *(["--no-pin"] if args.no_pin else []), "--ckpts", *[str(x) for x in args.ckpts]]
    print(f"[parent] attention split child mode={mode}", file=sys.__stderr__)
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=None)
    children.append({"mode": mode, "returncode": p.returncode, "stdout": (p.stdout or "")[-2000:], "artifact": rel(out)})
  return aggregate(args, children)

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--modes", default="baseline,q8")
  ap.add_argument("--ckpts", nargs="+", type=int, default=[512, 1024, 2048, 4096])
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--seed", type=int, default=20260620)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--aggregate-existing", action="store_true")
  ap.add_argument("--no-pin", action="store_true", help="do NOT pin peak clock (default pins manual_peak, restores auto)")
  ap.add_argument("--child-out", type=pathlib.Path)
  ap.add_argument("--mode", choices=["baseline", "q8"], default="baseline")
  args = ap.parse_args()
  if args.child_out is not None: return run_child(args)
  if args.aggregate_existing: return aggregate(args, [])
  return run_parent(args)

if __name__ == "__main__":
  raise SystemExit(main())
