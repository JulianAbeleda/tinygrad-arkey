#!/usr/bin/env python3
"""Block-quant decode GEMV on one model's real tensors of one format: the warp route (geometry from the device's facts)
against tinygrad's GGUF dequantizer, and its cold-L2 time. One row per distinct shape.

usage: block_quant_microgate.py --model M.gguf --quant Q5_K [--out result.json] [--peak-gbs 1693.9]"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))


def main() -> int:
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.helpers import getenv
  from tinygrad.llm.gguf import gguf_load_metadata, ggml_data_to_tensor
  from tinygrad.llm.qk_layout import GGML_TYPE_IDS, QUANT_FORMATS, packed_nbytes
  from tinygrad.llm.decode_kernels import block_quant_gemv_geometry, emit_block_quant_gemv_kernel
  from tinygrad.llm.qk_primitives import device_sm_count
  from tinygrad.codegen.late.warp_reduce import WARP
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True); ap.add_argument("--quant", required=True); ap.add_argument("--out")
  ap.add_argument("--peak-gbs", type=float, default=None); ap.add_argument("--iters", type=int, default=50)
  a = ap.parse_args()
  fmt, typ = QUANT_FORMATS[a.quant], GGML_TYPE_IDS[a.quant]
  _kv, meta = gguf_load_metadata(a.model)
  shapes: dict[tuple[int, int], tuple[str, int]] = {}
  for name, dims, t, off in meta["tensor_infos"]:
    if t == typ and len(dims) == 2: shapes.setdefault((int(dims[1]), int(dims[0])), (name, off))
  dev = Device.DEFAULT
  sm = device_sm_count(dev)
  flush = Tensor.empty(64 << 20, dtype=dtypes.float32).contiguous().realize()  # 256 MB: more than the L2
  rows_out = []
  with open(a.model, "rb") as f:
    for (rows, k), (name, off) in sorted(shapes.items()):
      nbytes = packed_nbytes(fmt, rows * k)
      f.seek(meta["data_start"] + off); raw = np.frombuffer(f.read(nbytes), dtype=np.uint8).copy()
      spec = block_quant_gemv_geometry(fmt, rows, k, wave_size=WARP, sm_count=sm, max_threads=1024)
      x_np = np.random.default_rng(0).standard_normal(k).astype(np.float16)
      halfs = Tensor(raw.view(np.uint16), dtype=dtypes.uint16).realize(); x = Tensor(x_np).realize()
      out = Tensor.empty(rows, dtype=dtypes.float32).uop_program(halfs, x, fxn=emit_block_quant_gemv_kernel(spec))[0]
      got = out.realize().numpy()
      w = ggml_data_to_tensor(Tensor(raw), rows * k, typ).numpy().reshape(rows, k).astype(np.float64)
      want = w @ x_np.astype(np.float64)
      rel = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))
      from tinygrad.engine.realize import time_call
      call = Tensor.empty(rows, dtype=dtypes.float32).uop_program(halfs, x, fxn=emit_block_quant_gemv_kernel(spec))[0].schedule_linear().src[0]
      times = []
      for _ in range(a.iters):
        (flush + 1).realize(); Device[dev].synchronize()  # a 256 MB write between samples: the weights start cold
        times.append(time_call(call))
      us = float(np.median(times) * 1e6)
      row = {"tensor": name, "rows": rows, "k": k, "bytes": nbytes, "lanes_per_block": spec.lanes_per_block, "warps_per_cta": spec.warps_per_cta,
             "max_rel_err": rel, "kernel_us_median": us, "gbs": nbytes / us / 1e3}
      if a.peak_gbs: row["pct_peak"] = 100 * row["gbs"] / a.peak_gbs
      rows_out.append(row); print(json.dumps(row))
  res = {"schema": "tinygrad.block_quant_microgate.v1", "model": a.model, "quant": a.quant, "device": dev, "sm_count": sm,
         "timing": "time_call (device timestamps, the kernel alone) after a 256 MB write (cold L2), median",
         "rows": rows_out, "pass": all(r["max_rel_err"] < 1e-3 for r in rows_out)}
  if a.out: pathlib.Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
  print(json.dumps({"pass": res["pass"], "shapes": len(rows_out)})); return 0 if res["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
