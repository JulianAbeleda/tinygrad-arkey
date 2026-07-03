#!/usr/bin/env python3
"""Phases SK0+SK4A: split-K role-local discovery on 14B ffn_down (17408->5120), reusing the existing generated
partial kernel (q4k_gemv_packed_load_partial_kernel, whose `parts` axis IS the K split) + Tensor.sum combine.

SK0 (Amdahl precheck): direct G3 serial depth = ceil(k_blocks/4); split serial depth = ceil(k_blocks/(4*parts)).
SK4A (role-local microbench): for parts in {1,2,4,8}, synced min-of-bursts GEMV time (partial kernel + .sum combine)
+ correctness vs dequant reference, compared to the direct G3-anyshape kernel. Decides whether split-K actually
helps role-local for the deepest-serial 14B role before any model binding.

Writes bench/qwen-14b-32b-truegen/sk4a_14b_discovery/{latest,per_candidate}.json + learned_rule.json + summary.md
Verdicts: SK4A_PASS_14B_DISCOVERY_RULE_LEARNED / SK4A_REFUTED_14B_SPLIT_K_NO_ROLE_LOCAL_WIN
"""
from __future__ import annotations
import sys, json, math, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qwen-14b-32b-truegen/sk4a_14b_discovery"
IN_F, OUT_F = 17408, 5120        # 14B ffn_down
QK_K = 256
SPLITS = [1, 2, 4, 8]

def _synced_min(fn, dev, warmup=8, bursts=20, reps=3):
  from tinygrad import TinyJit
  jf = TinyJit(fn)
  for _ in range(warmup): jf().realize()
  dev.synchronize()
  ts = []
  for _ in range(reps):
    dev.synchronize(); t0 = time.perf_counter()
    for _ in range(bursts): jf().realize()
    dev.synchronize(); ts.append((time.perf_counter() - t0) / bursts * 1e3)
  return min(ts), round((max(ts) - min(ts)) / min(ts) * 100, 1)

def main():
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.helpers import fetch
  from tinygrad.llm.model import Transformer
  from extra.qk.quant.q4_k_gemv_primitive import q4k_gemv_packed_load_partial_kernel
  from extra.qk.gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
  dev = Device[Device.DEFAULT]
  k_blocks = IN_F // QK_K
  q4k_bytes = OUT_F * IN_F // 2   # ~4 bits/weight for the role's Q4_K weights

  # SK0 precheck
  direct_serial = math.ceil(k_blocks / 4)
  sk0 = {str(p): {"split_serial_depth": math.ceil(k_blocks / (4 * p)),
                  "serial_speedup_bound": round(direct_serial / math.ceil(k_blocks / (4 * p)), 2)} for p in SPLITS}

  m, kv = Transformer.from_gguf(fetch("/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf"), 2048)
  lin = next(L for L in m._q4k_linears.linears if L.in_features == IN_F and L.out_features == OUT_F and "down" in L.name)
  lin.decode_enabled = True
  Tensor.manual_seed(1)
  x = Tensor.randn(1, 1, IN_F, dtype=dtypes.float16).contiguous().realize()
  ref = lin._fallback(x).reshape(-1).realize().numpy()
  words = lin.q4k_storage.words.to(x.device).contiguous() if lin.q4k_storage.mode == "q4_ondemand" else lin.q4k_storage.words.to(x.device)
  xv = x[:, 0, :].reshape(IN_F).cast(dtypes.float16).contiguous().realize()

  cands = []
  # direct G3 (the shipped generated route) as the baseline
  def g3_run():
    out = Tensor.empty(OUT_F, dtype=dtypes.float32, device=x.device)
    return out.custom_kernel(words, xv, fxn=q4k_g3_lanemap_gemv_kernel(OUT_F, IN_F))[0]
  g3_ms, g3_spread = _synced_min(g3_run, dev)
  g3_rel = float(((ref - g3_run().realize().numpy())**2).sum()**0.5 / ((ref**2).sum()**0.5 + 1e-9))
  cands.append({"candidate": "direct_g3", "split_k_parts": 1, "ms": round(g3_ms, 4),
                "gb_s": round(q4k_bytes / 1e9 / (g3_ms / 1e3), 1), "spread_pct": g3_spread,
                "rel_rmse": g3_rel, "correct": g3_rel < 5e-3})

  # split-K partial kernel + .sum combine, parts in {1,2,4,8}
  for parts in SPLITS:
    def run(parts=parts):
      partials = Tensor.empty(OUT_F, parts, dtype=dtypes.float32, device=x.device)
      p = partials.custom_kernel(words, xv, fxn=q4k_gemv_packed_load_partial_kernel(OUT_F, IN_F, parts, "none", ()))[0]
      return p.sum(axis=1)
    try:
      ms, spread = _synced_min(run, dev)
      got = run().realize().numpy()
      rel = float(((ref - got)**2).sum()**0.5 / ((ref**2).sum()**0.5 + 1e-9))
      cands.append({"candidate": f"split_k_{parts}", "split_k_parts": parts, "ms": round(ms, 4),
                    "gb_s": round(q4k_bytes / 1e9 / (ms / 1e3), 1), "spread_pct": spread,
                    "rel_rmse": rel, "correct": rel < 5e-3,
                    "serial_depth": sk0[str(parts)]["split_serial_depth"]})
    except Exception as e:
      cands.append({"candidate": f"split_k_{parts}", "split_k_parts": parts, "error": str(e)[:160]})

  correct = [c for c in cands if c.get("correct")]
  best = min(correct, key=lambda c: c["ms"]) if correct else None
  win = best and best["candidate"] != "direct_g3" and best["split_k_parts"] > 1
  speedup_vs_g3 = round(g3_ms / best["ms"], 3) if best else None
  verdict = "SK4A_PASS_14B_DISCOVERY_RULE_LEARNED" if win else "SK4A_REFUTED_14B_SPLIT_K_NO_ROLE_LOCAL_WIN"

  result = {"role": "14b_ffn_down", "in_features": IN_F, "out_features": OUT_F, "k_blocks": k_blocks,
            "sk0_precheck": {"direct_serial_depth": direct_serial, "bounds": sk0}, "candidates": cands,
            "best": best, "best_speedup_vs_direct_g3": speedup_vs_g3, "verdict": verdict}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "per_candidate.json").write_text(json.dumps(cands, indent=2))
  (OUT / "latest.json").write_text(json.dumps(result, indent=2))
  learned = {"source_profile": "qwen3-14b-q4k-decode-gfx1100", "source_role": "ffn_down", "k_blocks": k_blocks,
             "serial_blocks_per_lane_direct": direct_serial,
             "winning_split_k_parts": best["split_k_parts"] if win else None,
             "combine_route": "tensor_sum_axis1",
             "selection_reason": f"best role-local ms among correct candidates ({speedup_vs_g3}x vs direct G3)" if win
                                 else "no split_k_parts>1 beat direct G3 role-local",
             "transfer_rule": "split_k_parts = argmin_role_local_ms over {1,2,4,8}; expected to scale with serial depth"}
  (OUT / "learned_rule.json").write_text(json.dumps(learned, indent=2))

  print(f"14B ffn_down {IN_F}->{OUT_F} (k_blocks={k_blocks}, direct serial depth {direct_serial}):")
  for c in cands:
    if "error" in c: print(f"  {c['candidate']:12} ERROR: {c['error']}")
    else: print(f"  {c['candidate']:12} {c['ms']:.3f} ms | {c['gb_s']:6.1f} GB/s | spread {c['spread_pct']}% | "
                f"rel {c['rel_rmse']:.1e} {'OK' if c['correct'] else 'BAD'}")
  print(f"== {verdict} == best={best['candidate'] if best else None} speedup_vs_g3={speedup_vs_g3}")

if __name__ == "__main__":
  main()
