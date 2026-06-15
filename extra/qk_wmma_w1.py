#!/usr/bin/env python3
"""Phase W1 verification: fused Q4_K dequant -> WMMA.

Forcing tensor cores (TC_OPT=2) makes tinygrad emit WMMA on the FUSED dequant matmul (the matcher
only requires both MUL operands be fp16, which the dequant-cast-to-f16 satisfies). This checks the
three things that matter: correctness, device speed vs the materialized-fp16 WMMA ceiling, and that
the fused kernel reads the COMPRESSED weights (no fp16 round-trip).
"""
from __future__ import annotations

import os
os.environ.setdefault("TC", "1")
os.environ.setdefault("TC_OPT", "2")  # force tensor cores even on the fused kernel
os.environ.setdefault("DEBUG", "2")   # populate GlobalCounters.time_sum_s + global_mem

import argparse, json, pathlib, statistics
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import GlobalCounters
from extra.qk_layout import read_metadata, pick_tensor, tensor_shape, q4_k_reference, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS

DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_ARTIFACT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/wmma-w1")
PEAK_TFLOPS = 83.64

def _time(fn, *, warmup=5, iters=15):
  for _ in range(warmup): fn().realize()
  Device[Device.DEFAULT].synchronize()
  ts, mems = [], []
  for _ in range(iters):
    GlobalCounters.reset()
    fn().realize(); Device[Device.DEFAULT].synchronize()
    ts.append(GlobalCounters.time_sum_s); mems.append(GlobalCounters.global_mem)
  return statistics.median(ts), max(mems)

def run(model, tensor, batches, device, artifact):
  model = model.expanduser().resolve()
  meta = read_metadata(model); info = pick_tensor(meta.infos, tensor); rows, k = tensor_shape(info)
  bs = meta.data_start + info.off; q4 = rows*(k//Q4_K_BLOCK_ELEMS)*Q4_K_BLOCK_BYTES
  raw = Tensor(model)[bs:bs+q4].to(device)
  decoded_f32 = q4_k_reference(raw, rows*k).reshape(rows, k).cast(dtypes.float32).realize()
  decoded_f16 = decoded_f32.cast(dtypes.float16).realize()
  curve = []
  for b in batches:
    Tensor.manual_seed(1337)
    x = Tensor.randn(b, k, dtype=dtypes.float16, device=device).realize()
    ref = (decoded_f32 @ x.cast(dtypes.float32).transpose()).realize()  # [rows, b]
    fused = lambda: (x @ q4_k_reference(raw, rows*k).reshape(rows, k).cast(dtypes.float16).transpose())  # compressed, fused
    dense = lambda: (x @ decoded_f16.transpose())  # materialized fp16 WMMA ceiling
    got = fused().realize()
    rel = (got.transpose() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    t_f, mem_f = _time(fused); t_d, mem_d = _time(dense)
    flops = 2*rows*k*b
    curve.append({
      "batch": b, "correct": rel < 1e-2, "rel_err": round(rel, 6),
      "fused_us": round(t_f*1e6, 2), "fused_tflops": round(flops/t_f/1e12, 2), "fused_pct_peak": round(flops/t_f/1e12/PEAK_TFLOPS*100, 2),
      "fused_global_mb": round(mem_f/1e6, 2),
      "dense_us": round(t_d*1e6, 2), "dense_tflops": round(flops/t_d/1e12, 2), "dense_pct_peak": round(flops/t_d/1e12/PEAK_TFLOPS*100, 2),
      "dense_global_mb": round(mem_d/1e6, 2),
      "fused_vs_dense": round(t_d/t_f, 3),
    })
  return {"tensor": tensor, "shape": [rows, k], "compressed_q4_mb": round(q4/1e6, 2), "curve": curve}

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  p.add_argument("--tensor", default="blk.20.attn_q.weight")
  p.add_argument("--batch", type=int, action="append", default=None)
  p.add_argument("--device", default="AMD")
  p.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  p.add_argument("--all-tensors", action="store_true")
  args = p.parse_args()
  batches = args.batch or [16, 64, 128, 256]
  tensors = ("blk.20.attn_q.weight", "blk.13.ffn_gate.weight") if args.all_tensors else (args.tensor,)
  per = {t: run(args.model, t, batches, args.device, args.artifact) for t in tensors}
  out = {"kind": "qk_flywheel_wmma_w1", "phase": "Phase W1", "fp16_compute_peak_tflops": PEAK_TFLOPS,
         "tc_forced": "TC_OPT=2", "per_tensor": per,
         "note": "fused = compressed Q4_K dequant -> WMMA (TC forced); dense = materialized fp16 -> WMMA (ceiling)."}
  if args.all_tensors:
    args.artifact.mkdir(parents=True, exist_ok=True)
    (args.artifact / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
