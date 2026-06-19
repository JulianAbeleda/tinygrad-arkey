#!/usr/bin/env python3
"""P7b-2 eager parity for rebindable imported Q4 MMVQ."""
from __future__ import annotations

import json

import numpy as np

from tinygrad import Device, Tensor, dtypes
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.qk_decode_mmvq_graph_route import Q8_BYTES, route_imported_q4_mmvq
from extra.qk_decode_mmvq_p3_q4_correctness import OUT, q4_tensor_bytes


class LinearLike:
  def __init__(self, words: Tensor, rows: int):
    self.out_features = rows
    self.q4k_storage = type("S", (), {"mode": "sidecar", "words": words})()


def q8_dequant(q8: bytes) -> np.ndarray:
  vals = []
  for off in range(0, len(q8), 36):
    d = np.frombuffer(q8[off:off + 2], dtype=np.float16).astype(np.float32)[0]
    vals.append(np.frombuffer(q8[off + 4:off + 36], dtype=np.int8).astype(np.float32) * d)
  return np.concatenate(vals).astype(np.float32)


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P7b-2 requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  q4, rows, k = q4_tensor_bytes("blk.0.attn_output.weight")
  rng = np.random.default_rng(20260619)
  x = rng.standard_normal(k).astype(np.float32)
  q8 = q8_blocks(x)
  ref = q4_ref_rows(q4, rows, k, q8_dequant(q8))

  words = Tensor(np.frombuffer(q4, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  x_t = Tensor(x.reshape(1, 1, k), dtype=dtypes.float32, device="AMD").contiguous().realize()
  q8_side = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_side = Tensor.empty(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()
  out = route_imported_q4_mmvq(LinearLike(words, rows), x_t, q8_side, out_side)
  if out is None:
    raise RuntimeError("route returned None")
  out = out.realize()
  Device["AMD"].synchronize(timeout=10000)
  got = out.numpy().reshape(rows)
  diff = np.abs(got - ref)
  result = {
    "schema": "decode_mmvq_large_project_p7b_eager_parity_v1",
    "date": "2026-06-19",
    "phase": "P7b_2_eager_parity",
    "tensor": "blk.0.attn_output.weight",
    "rows": rows,
    "k": k,
    "max_abs": float(diff.max()),
    "mean_abs": float(diff.mean()),
    "max_rel": float((diff / np.maximum(np.abs(ref), 1e-6)).max()),
    "verdict": "PASS_EAGER_REBIND_PARITY" if float(diff.max()) < 2e-2 else "KILL",
  }
  (OUT / "p7b_eager_parity.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS_EAGER_REBIND_PARITY":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
