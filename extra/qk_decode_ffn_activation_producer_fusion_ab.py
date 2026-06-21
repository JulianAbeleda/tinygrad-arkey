#!/usr/bin/env python3
"""Phase B1 (decode-fusion-build-scope): FFN activation producer fusion standalone A/B.

Baseline: gate = ffn_gate(x); up = ffn_up(x); act = silu(gate) * up   <- the standalone E_49152 launch.
Fused:    gate = ffn_gate(x); act = fused_up(x, gate)                  <- 'up' GEMV writes silu(gate)*up directly,
          via q4k_gemv_silu_gate_kernel (no separate elementwise launch).

Checks correctness (fused act vs baseline act) and same-process interleaved timing (clock-pinned). This is a
local gate only; full W==D promotion is separate. Default decode behavior NOT changed.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_ffn_activation_producer_fusion_ab.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
from tinygrad import Tensor, dtypes, Device, TinyJit
from extra.llm_generate import load_model_and_tokenizer
from extra.q4_k_gemv_primitive import q4k_gemv_silu_gate_kernel
from extra.qk_clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-fusion-build/ffn_activation_producer_fusion_ab.json"

def _words(lin, dev):
  s = lin.q4k_storage
  return s.words.to(dev).contiguous() if s.mode == "q4_ondemand" else s.words.to(dev)

def fused_up_act(up_lin, gate_lin, x):
  """gate=gate_lin(x); return silu(gate)*up fused in the up GEMV's store. x:[1,1,in]."""
  gate = gate_lin(x)  # [1,1,out] fp32 (decode primitive)
  dev = x.device
  x_vec = x[:, 0, :].reshape(up_lin.in_features).cast(dtypes.float16).contiguous()
  gate_flat = gate.reshape(up_lin.out_features).cast(dtypes.float32).contiguous()
  out = Tensor.empty(up_lin.out_features, dtype=dtypes.float32, device=dev)
  got = out.custom_kernel(_words(up_lin, dev), x_vec, gate_flat,
                          fxn=q4k_gemv_silu_gate_kernel(up_lin.out_features, up_lin.in_features, "none", up_lin.opts))[0]
  return got.reshape(1, 1, up_lin.out_features)

def baseline_act(up_lin, gate_lin, x):
  return gate_lin(x).silu().contiguous() * up_lin(x)

def main():
  dev = Device[Device.DEFAULT]
  m, tok = load_model_and_tokenizer("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 2048, seed=20260620)
  lins = {l.name: l for l in m._q4k_linears.linears}
  for l in lins.values(): l.decode_enabled = True
  gate_lin, up_lin = lins["blk.0.ffn_gate.weight"], lins["blk.0.ffn_up.weight"]
  print(f"gate {gate_lin.out_features}x{gate_lin.in_features} parts={gate_lin.parts} | "
        f"up {up_lin.out_features}x{up_lin.in_features} parts={up_lin.parts}", file=sys.__stderr__)
  assert up_lin.parts == 1, "fused producer requires parts=1"

  Tensor.manual_seed(7)
  x = Tensor.randn(1, 1, up_lin.in_features).realize()

  base = baseline_act(up_lin, gate_lin, x).realize().numpy()
  fused = fused_up_act(up_lin, gate_lin, x).realize().numpy()
  import numpy as np
  rel = float(np.abs(base - fused).max() / (np.abs(base).max() + 1e-9))
  amax = float(np.abs(base - fused).max())
  correct = rel < 1e-3
  print(f"correctness: max|abs|={amax:.3e} rel={rel:.3e} -> {'EXACT-ish OK' if correct else 'FAIL'}", file=sys.__stderr__)

  # same-process interleaved timing (clock-pinned). JIT each path, warm, then interleave.
  jb = TinyJit(lambda xx: baseline_act(up_lin, gate_lin, xx).realize())
  jf = TinyJit(lambda xx: fused_up_act(up_lin, gate_lin, xx).realize())
  res = {}
  with pinned_peak() as pin:
    time.sleep(0.4)
    for _ in range(12): jb(x); jf(x)
    dev.synchronize()
    NB = 200; tb = []; tf = []
    for _ in range(NB):
      t0 = time.perf_counter(); jb(x); dev.synchronize(); tb.append(time.perf_counter() - t0)
      t0 = time.perf_counter(); jf(x); dev.synchronize(); tf.append(time.perf_counter() - t0)
    res["pin"] = pin
  bms, fms = statistics.median(tb) * 1e3, statistics.median(tf) * 1e3
  print(f"baseline {bms*1000:.1f}us | fused {fms*1000:.1f}us | delta {(bms-fms)*1000:.1f}us "
        f"({100*(bms-fms)/bms:.1f}%)", file=sys.__stderr__)
  out = {"date": "2026-06-20", "phase": "FFN_ACT_PRODUCER_FUSION_AB", "role": "ffn", "candidate": "B1",
         "shapes": {"out": up_lin.out_features, "in": up_lin.in_features},
         "correctness": {"max_abs_diff": amax, "rel": rel, "exact_ish": correct, "note": "silu impl = g/(1+exp(-g)); fp-reassoc tol"},
         "local_timing_pinned": {"baseline_us": round(bms * 1000, 1), "fused_us": round(fms * 1000, 1),
                                 "delta_us": round((bms - fms) * 1000, 1), "delta_pct": round(100 * (bms - fms) / bms, 1)},
         "note": "isolated gate+up+act lifecycle, NOT whole-token; E_49152 elimination shows as the launch delta",
         "default_behavior_changed": False}
  OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
  print(json.dumps({"correct": correct, "delta_us": out["local_timing_pinned"]["delta_us"], "out": str(OUT.relative_to(ROOT))}))

if __name__ == "__main__":
  main()
