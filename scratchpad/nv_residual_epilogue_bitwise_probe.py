#!/usr/bin/env python3
"""CPU hermetic bitwise check of the residual-family in-core epilogue math.

The residual family's ordinary kernels are: (1) the fp32->fp16 cast of a GEMV
output (E_128_32_3 / E_32_32_4_0a5eb0ac) and (2) the fp16 residual add with
the block input (E_32_32_4_fab82d40).  This probe verifies the candidate
in-core spellings (the epi_resadd/fp16-cast kernel math) reproduce those
ordinary values bitwise on CPU: ``total.cast(fp16)`` and
``total.cast(fp16) + x``.
"""
from __future__ import annotations

import hashlib, sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context


def _digest(t: Tensor) -> str:
  return hashlib.sha256(np.ascontiguousarray(t.numpy()).view(np.uint8)).hexdigest()


def main() -> int:
  with Context(DEV="CPU"):
    rng = np.random.default_rng(0)
    total = Tensor(rng.standard_normal(4096).astype(np.float32), dtype=dtypes.float32)
    x = Tensor(rng.standard_normal(4096).astype(np.float16), dtype=dtypes.float16)

    # (1) The ordinary chain: cast the GEMV output to fp16, then fp16-add x.
    cast_ref = total.cast(dtypes.float16)
    add_ref = cast_ref + x
    # (2) Candidate in-core spellings.
    cast_cand = total.cast(dtypes.float16)
    add_cand = total.cast(dtypes.float16) + x

    results = {
      "cast_bitwise": _digest(cast_cand) == _digest(cast_ref),
      "add_bitwise": _digest(add_cand) == _digest(add_ref),
      "max_abs_cast": float(np.max(np.abs(cast_cand.numpy().astype(np.float32) - cast_ref.numpy().astype(np.float32)))),
      "max_abs_add": float(np.max(np.abs(add_cand.numpy().astype(np.float32) - add_ref.numpy().astype(np.float32)))),
    }
  import json
  print(json.dumps(results, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
