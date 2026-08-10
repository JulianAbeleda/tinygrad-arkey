#!/usr/bin/env python3
"""Feasibility gate for the fp32 q/k route: does the candidate cooperative
association (8 threads/row x 16 contiguous serial + per-row serial 8-chain)
reproduce the ordinary NV q-norm reduce+epilogue BITWISE?

Runs the ordinary (32,128) fp32 RMSNorm on NV, then reconstructs the output in
numpy with candidate associations and compares bitwise. A match proves the
fused multi-row body can pass the exact-logits gate."""
from __future__ import annotations

import numpy as np, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, dtypes, nn


def main() -> None:
  rows, dim, eps = 32, 128, 1e-6
  rng = np.random.default_rng(20260810)
  x_np = rng.normal(0, 0.2, (rows, dim)).astype(np.float32)
  w_np = rng.normal(1, 0.05, (dim,)).astype(np.float32)
  x = Tensor(x_np).realize()
  w = Tensor(w_np).realize()
  n = nn.RMSNorm(dim, eps=eps)
  n.weight = w
  ordinary = n(x).numpy()

  def chain(vals: np.ndarray) -> np.ndarray:
    acc = vals[0].astype(np.float32)
    for v in vals[1:]: acc = (acc + v.astype(np.float32)).astype(np.float32)
    return acc

  def serial_sum(seg: np.ndarray) -> np.float32:
    acc = np.float32(0.0)
    for v in seg: acc = np.float32(np.float32(acc) + np.float32(np.float32(v) * np.float32(v)))
    return acc

  def reconstruct(threads_per_row: int, per_thread: int, combine_serial: bool, row_major_threads: bool):
    """thread mapping: thread t covers [base, base+per_thread) contiguous."""
    scale = np.zeros((rows,), dtype=np.float32)
    for r in range(rows):
      partials = []
      for j in range(threads_per_row):
        t = r * threads_per_row + j
        base = t * per_thread if row_major_threads else (r * dim + j * per_thread)
        partials.append(serial_sum(x_np.reshape(-1)[base:base + per_thread]))
      if combine_serial:
        rowsum = chain(np.array(partials, dtype=np.float32))
      else:
        rowsum = partials[0]
        for p in partials[1:]: rowsum = np.float32(rowsum + p)
      scale[r] = np.float32(1.0 / np.sqrt(np.float32(np.float32(rowsum / dim) + eps)))
    out = np.empty_like(x_np)
    for r in range(rows):
      out[r] = (x_np[r].astype(np.float32) * scale[r]).astype(np.float32) * w_np.astype(np.float32)
    return out

  def reconstruct2(threads_per_row: int, per_thread: int, combine_serial: bool, row_major_threads: bool,
                   scale_mul_recip: bool, epilogue_order: str):
    scale = np.zeros((rows,), dtype=np.float32)
    for r in range(rows):
      partials = []
      for j in range(threads_per_row):
        t = r * threads_per_row + j
        base = t * per_thread if row_major_threads else (r * dim + j * per_thread)
        partials.append(serial_sum(x_np.reshape(-1)[base:base + per_thread]))
      rowsum = chain(np.array(partials, dtype=np.float32)) if combine_serial else partials[0]
      if not combine_serial:
        for p in partials[1:]: rowsum = np.float32(rowsum + p)
      mean = np.float32(rowsum * np.float32(1.0 / dim)) if scale_mul_recip else np.float32(rowsum / dim)
      scale[r] = np.float32(1.0 / np.sqrt(np.float32(mean + eps)))
    out = np.empty_like(x_np)
    for r in range(rows):
      if epilogue_order == "x*scale*w":
        out[r] = (x_np[r].astype(np.float32) * scale[r]).astype(np.float32) * w_np.astype(np.float32)
      elif epilogue_order == "x*(scale*w)":
        sw = np.float32(scale[r] * w_np.astype(np.float32))
        out[r] = (x_np[r].astype(np.float32) * sw).astype(np.float32)
      else:
        xw = (x_np[r].astype(np.float32) * w_np.astype(np.float32)).astype(np.float32)
        out[r] = (xw * scale[r]).astype(np.float32)
    return out

  print(f"ordinary out[0][:6] = {ordinary[0][:6]}")
  candidates = {
    "8x16 row-major serial-chain": reconstruct(8, 16, True, True),
    "8x16 row-major serial-acc": reconstruct(8, 16, False, True),
    "8x16 thread-major serial-chain": reconstruct(8, 16, True, False),
    "8x16 thread-major serial-acc": reconstruct(8, 16, False, False),
    "4x32 row-major serial-chain": reconstruct(4, 32, True, True),
    "16x8 row-major serial-chain": reconstruct(16, 8, True, True),
    "32x4 row-major serial-chain": reconstruct(32, 4, True, True),
  }
  for name, cand in candidates.items():
    exact = np.array_equal(cand, ordinary)
    print(f"{name}: bitwise={'MATCH' if exact else 'diff'} "
          f"maxdiff={np.max(np.abs(cand - ordinary)):.3e} first={cand[0][:3]}")
  print("\n--- refine (scale recip + epilogue order):")
  for mul_recip in (False, True):
    for order in ("x*scale*w", "x*(scale*w)", "(x*w)*scale"):
      cand = reconstruct2(8, 16, True, True, mul_recip, order)
      exact = np.array_equal(cand, ordinary)
      bad = np.argwhere(cand != ordinary)
      print(f"recip={mul_recip} order={order}: bitwise={'MATCH' if exact else 'diff'} "
            f"mismatches={len(bad)} first_mismatch={bad[0].tolist() if len(bad) else None}")


if __name__ == "__main__":
  main()
