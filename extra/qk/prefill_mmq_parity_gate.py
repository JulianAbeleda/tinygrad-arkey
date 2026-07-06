#!/usr/bin/env python3
"""Numeric parity helpers for the generated int8-WMMA Q4_K prefill substrate.

The generated substrate expresses the int dot as ordinary `Tensor.matmul(..., dtype=int)`.
These helpers are used by the tiled WMMA gates to build finite synthetic Q4_K weights
and compare against a q8-dequant activation reference.
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, q4_k_reference, q8_1_quantize
from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_prefill, emit_q4k_int8_wmma_prefill_tensor

RTOL = 6e-3


def _make_q4k_words(n:int, k:int, seed:int) -> tuple[Tensor, Tensor]:
  rng = np.random.default_rng(seed)
  assert k % Q4_K_BLOCK_ELEMS == 0
  nblocks = (n * k) // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=nblocks * Q4_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q4_K_BLOCK_BYTES)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16)
  dmin = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16)
  raw[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = dmin.view(np.uint8).reshape(nblocks, 2)
  byte_t = Tensor(raw.reshape(-1).copy()).realize()
  ref = q4_k_reference(byte_t, n * k).reshape(n, k).cast(dtypes.float32).contiguous().realize()
  words = byte_t.bitcast(dtypes.uint32).contiguous().realize()
  return words, ref


def _rel_rmse(got:np.ndarray, ref:np.ndarray) -> float:
  return float(np.sqrt(np.mean((got - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12))


def run(n:int, k:int, m:int, seed:int=1337) -> None:
  words, ref_w = _make_q4k_words(n, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) *
          xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  ref_out = (x_dq @ ref_w.T).numpy()
  spec = describe_q4k_int8_wmma_prefill(n, k, m, role="parity")
  got = emit_q4k_int8_wmma_prefill_tensor(words, xq, xscales, spec).numpy()
  rel = _rel_rmse(got, ref_out)
  print(f"wmma_generated n={n} k={k} m={m} rel_rmse={rel:.3e} {'PASS' if rel < RTOL else 'FAIL'}")
  if rel >= RTOL: raise SystemExit(f"MMQ parity gate FAILED (rtol={RTOL})")


if __name__ == "__main__":
  for shape in [(64, 256, 16), (32, 512, 16), (16, 768, 16)]:
    run(*shape, seed=getenv("SEED", 1337))
  print("MMQ parity gate PASS")
