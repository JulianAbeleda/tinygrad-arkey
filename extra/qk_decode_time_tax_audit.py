#!/usr/bin/env python3
"""Decode primitive time-tax audit: per-kernel ProfileGraphEvent GPU time grouped into primitive buckets, by ctx, for
one-token decode. Attribution ONLY -- no kernel/default change. Picks the next primitive target by token-share + W==D
headroom, not intuition.

Buckets are assigned by kernel-name + dim-signature rules (Qwen3-8B: hidden 4096, FFN 12288, vocab 151936, Hd 128).
token_ms_total = real per-token wall (W path, .item() sync). GPU-busy sum = sum of ProfileGraphEvent kernel durations
(authority for shares). graph/host overhead = wall - effective GPU rate (reported separately).

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_time_tax_audit.py
"""
from __future__ import annotations
import collections, json, os, pathlib, re, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-time-tax-audit"
CTXS = [512, 1024, 2048, 4096]; MAXC = 4608; NMEAS = 40

# kernel-name -> bucket rules (first match wins). Dims are <out>_<in> in the q4k/q6k gemv names.
def classify(name:str) -> str:
  n = name
  if "151936" in n: return "lm_head"
  if n.startswith("flash_") or "start_pos" in n: return "attention_compute"
  if re.search(r"12288_4096", n): return "ffn_gate_up"          # gate/up: 4096 -> 12288
  if re.search(r"4096_12288", n): return "ffn_down"             # down:    12288 -> 4096
  if re.search(r"(^|_)1024_4096", n): return "attn_kv_proj"     # k/v proj: 4096 -> 1024
  if re.search(r"4096_4096", n): return "attn_qo_proj"          # q/o proj: 4096 -> 4096
  if n.startswith("E_49152") or n.startswith("E_1536"): return "ffn_activation"  # silu(gate)*up + gate/up bias-ish
  if n.startswith("E_") or n.startswith("r_") or n.startswith("R_"): return "norm_rope_small_ops"
  if "q8" in n.lower(): return "q8_route"
  return "unknown"

def main():
  from extra.qk_harness_contract import DEFAULT_MODEL
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  from tinygrad import Tensor, UOp, TinyJit, Context, Device, getenv
  from tinygrad.device import Compiled
  from extra.llm_generate import load_model_and_tokenizer
  dev = Device["AMD"]
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  route = {"DECODE_ATTN_AMDGCN_TILE": getenv("DECODE_ATTN_AMDGCN_TILE", 0), "Q8_FFN_HANDWRITTEN": getenv("Q8_FFN_HANDWRITTEN", 0),
           "FLASH_VARIANT": str(getenv("FLASH_VARIANT", "gqa_coop_vec"))}

  rows = []
  for ck in CTXS:
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False   # flash decode on (fires at ctx>=512 via auto-threshold)
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    # token_ms_total: real wall W (per-token .item() sync), median
    W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    token_ms = statistics.median(W)
    # per-kernel GPU-busy via ProfileGraphEvent (authority for shares)
    agg = collections.defaultdict(lambda: [0.0, 0])
    with Context(PROFILE=1):
      sp = TinyJit(m.forward); o2 = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
      for i in range(8): o2 = sp(o2, v.bind(ck + i), temp).realize()
      dev.synchronize(); dev._at_profile_finalize()
      samples = []
      for r in range(5):
        base = len(Compiled.profile_events); sp(o2, v.bind(ck + 20 + r), temp).realize(); dev.synchronize(); dev._at_profile_finalize()
        a = collections.defaultdict(float)
        for e in Compiled.profile_events[base:]:
          if type(e).__name__ != "ProfileGraphEvent": continue
          sigs = [float(s) for s in e.sigs]
          for ent in e.ents: a[str(ent.name)] += sigs[ent.en_id] - sigs[ent.st_id]
        samples.append(a)
      # median per-kernel over samples
      allk = set().union(*[s.keys() for s in samples])
      for k in allk:
        vals = [s.get(k, 0.0) for s in samples]; agg[k] = [statistics.median(vals), 1]
    # bucket
    buckets = collections.defaultdict(float); per_kernel = {}
    for k, (us, _) in agg.items():
      buckets[classify(k)] += us; per_kernel[k] = round(us, 1)
    gpu_busy = sum(buckets.values())
    # graph/host overhead = wall(us) - gpu_busy (host gaps + launch + the part not overlapped). Can be negative if
    # ProfileGraphEvent double-counts overlap; clamp at 0 and note.
    host_overhead = max(0.0, token_ms * 1e3 - gpu_busy)
    bucket_ms = {b: round(us / 1e3, 3) for b, us in sorted(buckets.items(), key=lambda x: -x[1])}
    bucket_share = {b: round(100 * us / gpu_busy, 1) for b, us in buckets.items()}
    rows.append({"ctx": ck, "token_ms_total": round(token_ms, 3), "gpu_busy_ms": round(gpu_busy / 1e3, 3),
                 "host_graph_overhead_ms": round(host_overhead / 1e3, 3),
                 "bucket_ms": bucket_ms, "bucket_share_pct_of_gpu_busy": dict(sorted(bucket_share.items(), key=lambda x:-x[1])),
                 "top_kernels": dict(sorted(per_kernel.items(), key=lambda x:-x[1])[:20]),
                 "first_tokens": toks[:6]})
    print(f"ctx {ck:5}: {token_ms:.2f}ms/tok ({1000/token_ms:.1f} tok/s) | gpu-busy {gpu_busy/1e3:.2f}ms | "
          + " ".join(f"{b}:{bucket_share[b]:.0f}%" for b in list(bucket_ms)[:6]), file=sys.__stderr__)

  try:
    commit = subprocess.run(["git","rev-parse","--short","HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git","status","--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
  except Exception: commit, dirty = None, None
  out_d = {"date": "2026-06-22", "phase": "DECODE_TIME_TAX_AUDIT", "model": os.path.basename(model),
           "hardware": "RX 7900 XTX / gfx1100", "contexts": CTXS, "nmeas": NMEAS, "route_flags": route,
           "timing_authority": "token_ms_total = real per-token wall (W, .item() sync, median); bucket_ms = ProfileGraphEvent "
                               "per-kernel GPU-busy (median-of-5); shares are of gpu_busy_sum (overlap not removed)",
           "kernel_name_to_bucket_rules": {"lm_head":"151936","attention_compute":"flash_*|start_pos","ffn_gate_up":"12288_4096",
             "ffn_down":"4096_12288","attn_kv_proj":"1024_4096","attn_qo_proj":"4096_4096","ffn_activation":"E_49152|E_1536",
             "norm_rope_small_ops":"E_*|r_*","q8_route":"q8","unknown":"else"},
           "commit": commit, "dirty_tree": dirty, "rows": rows, "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"latest.json").write_text(json.dumps(out_d, indent=2))
  print(f"artifact: {OUT/'latest.json'}", file=sys.__stderr__)

if __name__ == "__main__":
  main()
