#!/usr/bin/env python3
"""Numeric parity gate for the generated int8-WMMA Q4_K prefill substrate.

The handwritten scalar-sdot4 / Q8_1 MMQ prefill kernels this gate used to cover
were deleted 2026-07-06 (no backups; confirmed ~237 tok/s dead end). What remains
is the machine-generated int8-WMMA substrate (extra/qk/prefill_int8_wmma_spec.py),
whose int dot is an ordinary Tensor.matmul(..., dtype=int) lowered by the compiler
(tc.py + cstyle) -- no route-local handwritten kernel. This gate makes a
self-contained Q4_K weight (random bytes decoding to a valid block; we only pin
d/dmin to finite fp16), runs the generated substrate the prefill route wires for
PREFILL_Q4K_Q8=wmma, and compares to a q8-dequant activation matmul reference.
Runs on DEV=PYTHON (GPU-free).

  DEV=PYTHON python extra/qk/prefill_mmq_parity_gate.py
"""
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, q4_k_reference, q8_1_quantize
from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_prefill, emit_q4k_int8_wmma_prefill_tensor

RTOL = 6e-3  # q8_1 activation quant + q4_k weight quant; matches the ~4.8e-3 numpy MMQ validation

def _make_q4k_words(n:int, k:int, seed:int) -> tuple[Tensor, Tensor]:
  """Return (words uint32 flat, ref_weight fp32 [n,k]).  Random block bytes with
  d/dmin overwritten by sane finite fp16 so the reference never sees inf/nan."""
  rng = np.random.default_rng(seed)
  assert k % Q4_K_BLOCK_ELEMS == 0
  nblocks = (n * k) // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=nblocks * Q4_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q4_K_BLOCK_BYTES)
  # first 4 bytes of each block are two fp16: d (super-scale) and dmin (super-min)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16)
  dmin = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16)
  raw[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = dmin.view(np.uint8).reshape(nblocks, 2)
  raw = raw.reshape(-1)
  byte_t = Tensor(raw.copy()).realize()
  ref = q4_k_reference(byte_t, n * k).reshape(n, k).cast(dtypes.float32).contiguous().realize()
  words = byte_t.bitcast(dtypes.uint32).contiguous().realize()
  return words, ref

def _rel_rmse(got:np.ndarray, ref:np.ndarray) -> float:
  return float(np.sqrt(np.mean((got - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12))

def run(n:int, k:int, m:int, seed:int=1337) -> None:
  words, ref_w = _make_q4k_words(n, k, seed)
  x = (Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32))).realize()

  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  # The generated substrate operates on the q8_1-QUANTIZED activation, so the correctness target is the
  # dequantized activation matmul (isolates weight-unpack + int-dot correctness from the ~1% q8 activation-quant
  # error, which is a property of the format, not the kernel).
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) * xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  ref_out = (x_dq @ ref_w.T).numpy()  # [m, n]  -- q8-dequant activation reference

  # --- wmma-generated substrate: no handwritten kernel, int dot expressed as Tensor.matmul(..., dtype=int) ---
  wmma_spec = describe_q4k_int8_wmma_prefill(n, k, m, role="parity")
  wmma_out = emit_q4k_int8_wmma_prefill_tensor(words, xq, xscales, wmma_spec).numpy()

  ok = True
  for label, got in (("wmma_generated", wmma_out),):
    r = _rel_rmse(got, ref_out)
    status = "PASS" if r < RTOL else "FAIL"
    if r >= RTOL: ok = False
    print(f"  {label:14s} n={n} k={k} m={m}  rel_rmse={r:.3e}  {status}")
  if not ok: raise SystemExit(f"MMQ parity gate FAILED (rtol={RTOL})")

if __name__ == "__main__":
  # small GPU-free shapes; k multiple of 256, plus a k>256 multi-block case
  for (n, k, m) in [(64, 256, 16), (32, 512, 16), (16, 768, 16)]:
    run(n, k, m, seed=getenv("SEED", 1337))
  print("MMQ parity gate PASS")
