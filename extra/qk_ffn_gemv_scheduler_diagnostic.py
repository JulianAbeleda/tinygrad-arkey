#!/usr/bin/env python3
"""Decode FFN-GEMV scheduler diagnostic. Attribution ONLY -- no kernel optimization, no new primitive, no default
change. Names the failing layer for the FFN Q4_K/Q6_K GEMV tax and the controlled-toggle transfer test.

Role bandwidth from the time-tax artifact; controlled toggles (default / Q4K_VDOT int-dot / Q4K_VDOT+AMORT / q8 FFN)
measured in-process for the W==D transfer test (avoid the attention mistake where a local win didn't transfer).

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_ffn_gemv_scheduler_diagnostic.py
"""
from __future__ import annotations
import json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-ffn-gemv-scheduler-diagnostic"
TAX = ROOT / "bench/qk-decode-time-tax-audit/latest.json"
HBM_PEAK_GBs = 960.0
Q4K_BLOCK_BYTES = 144; Q4K_BLOCK_ELEMS = 256; Q6K_BLOCK_BYTES = 210
MAXC = 4608; NMEAS = 40

def q4k_bytes(rows:int, k:int) -> int: return rows * (k // Q4K_BLOCK_ELEMS) * Q4K_BLOCK_BYTES
def q6k_bytes(rows:int, k:int) -> int: return rows * (k // Q4K_BLOCK_ELEMS) * Q6K_BLOCK_BYTES

def role_table():
  d = json.loads(TAX.read_text()); r = next(x for x in d["rows"] if x["ctx"] == 1024)
  tk = r["top_kernels"]
  # per-token totals (us) -> per-call (calls/token from layer counts: 36 layers; gate+up=72, down=18+18, qo=72)
  roles = [
    ("ffn_gate_up", "q4k_gemv_partial_12288_4096_1", 12288, 4096, "Q4_K", 72, q4k_bytes(12288, 4096)),
    ("ffn_down_q4k", "q4k_gemv_partial_4096_12288_4", 4096, 12288, "Q4_K", 18, q4k_bytes(4096, 12288)),
    ("ffn_down_q6k", "q6k_coop_partial_4096_12288", 4096, 12288, "Q6_K", 18, q6k_bytes(4096, 12288)),
    ("attn_qo_proj", "q4k_coop_partial_4096_4096", 4096, 4096, "Q4_K", 72, q4k_bytes(4096, 4096)),
  ]
  out = []
  for role, kname, rows, k, q, calls, bytes_per_call in roles:
    tot_us = tk.get(kname, 0.0)
    per_call_us = tot_us / calls if calls else 0.0
    gbps = bytes_per_call / (per_call_us * 1e-6) / 1e9 if per_call_us > 0 else 0.0
    out.append({"role": role, "kernel": kname, "shape": f"{rows}x{k}", "quant": q, "calls_per_token": calls,
                "total_us_per_token": round(tot_us, 1), "per_call_us": round(per_call_us, 2),
                "bytes_per_call": bytes_per_call, "eff_GBs": round(gbps, 1),
                "pct_peak": round(100 * gbps / HBM_PEAK_GBs, 1)})
  return out, r["token_ms_total"], r["bucket_share_pct_of_gpu_busy"]

def main():
  from extra.qk_harness_contract import DEFAULT_MODEL
  model = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  from tinygrad import Tensor, UOp, TinyJit, Device
  from tinygrad.helpers import getenv
  from extra.llm_generate import load_model_and_tokenizer
  dev = Device["AMD"]
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])

  roles, token_ms, shares = role_table()
  print("=== FFN GEMV role bandwidth (ctx1024) ===", file=sys.__stderr__)
  for r in roles:
    print(f"  {r['role']:14} {r['shape']:>10} {r['quant']} per-call {r['per_call_us']:5.1f}us  {r['eff_GBs']:5.0f} GB/s "
          f"({r['pct_peak']:.0f}% peak)  share {shares.get('ffn_gate_up' if 'gate' in r['role'] else 'ffn_down','?')}", file=sys.__stderr__)

  # controlled toggle ladder @ctx1024 (W==D transfer test). Each toggle => fresh jit (getenv cached).
  ck = 1024
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  def measure(env:dict):
    for k in ("Q4K_VDOT", "Q4K_VDOT_AMORT", "Q8_FFN_HANDWRITTEN"): os.environ.pop(k, None)
    for k, val in env.items(): os.environ[k] = str(val)
    getenv.cache_clear()
    step = TinyJit(m.forward); out = Tensor([[int(ids[ck])]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v.bind(ck + i), temp).realize()
    W = []; toks = []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v.bind(ck + i), temp); tid = int(out.item())
      W.append((time.perf_counter() - t0) * 1e3); toks.append(tid)
    return statistics.median(W), toks
  ladder = [("default", {}, "lossless"), ("Q4K_VDOT", {"Q4K_VDOT": 1}, "lossless int-dot"),
            ("Q4K_VDOT_AMORT", {"Q4K_VDOT": 1, "Q4K_VDOT_AMORT": 1}, "lossless int-dot + amortized q8 quant"),
            ("Q8_FFN", {"Q8_FFN_HANDWRITTEN": 1}, "LOSSY q8 weights")]
  base_ms, base_toks = measure({}); results = []
  for name, env, kind in ladder:
    ms, toks = (base_ms, base_toks) if name == "default" else measure(env)
    delta = 100 * (base_ms - ms) / base_ms
    results.append({"toggle": name, "kind": kind, "tok_s": round(1000/ms, 1), "delta_pct": round(delta, 2),
                    "tokens_match_default": toks == base_toks})
    print(f"  {name:16} {1000/ms:5.1f} tok/s  Δ{delta:+5.2f}%  tokens_match={toks==base_toks} ({kind})", file=sys.__stderr__)

  art = {"date": "2026-06-22", "phase": "FFN_GEMV_SCHEDULER_DIAGNOSTIC", "model": os.path.basename(model),
         "hardware": "RX 7900 XTX / gfx1100", "hbm_peak_GBs": HBM_PEAK_GBs, "ctx": ck, "token_ms_total": token_ms,
         "role_bandwidth": roles, "bucket_shares": shares, "controlled_toggle_ladder": results,
         "llama_reference": {"source": "docs/llama-q4k-mmvq-scheduler-audit-20260618.md",
           "tinygrad_default_pct_peak": 40, "tinygrad_coop_pct_peak": 53, "llama_mmvq_pct_peak": 70,
           "named_gap": "WORK DECOMPOSITION (not math/dot): llama = 128 threads/row + K-block-parallel (no serial blk loop) "
                        "+ in-kernel warp-shuffle reduce + one output write; tinygrad default = 1 thread/row serial uncoalesced (40%), "
                        "coop = 8 lanes/row + serial blk + stage-2 sum (53%). dot4 + packed-extract already MATCHED.",
           "enabling_primitives_exist": ["_sdot4 (native v_dot4)", "extra/amd_warp_reduce.warp_reduce_sum (ds_bpermute)"],
           "expressible": True},
         "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT/"latest.json").write_text(json.dumps(art, indent=2))
  print(f"artifact: {OUT/'latest.json'}", file=sys.__stderr__)

if __name__ == "__main__":
  main()
