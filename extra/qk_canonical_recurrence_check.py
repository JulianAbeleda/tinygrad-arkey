#!/usr/bin/env python3
"""Canonical single-accumulator recurrence reduce: out[h] = sum_j in[h*8+j] via a REG acc + .after(j).
Regression guard for the recurrence-unroll primitive. Run:
  DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk_canonical_recurrence_check.py
"""
from __future__ import annotations
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, AxisType

H, K = 6, 8


def kernel(out: UOp, inp: UOp) -> UOp:
  h = UOp.range(H, 0, AxisType.GLOBAL)
  acc = UOp.placeholder((1,), dtypes.float32, 100, addrspace=AddrSpace.REG)
  acc = acc.after(h)[0].set(0.0)
  j = UOp.range(K, 1, axis_type=AxisType.REDUCE)
  acc = acc[0].set(acc.after(j)[0] + inp[h * K + j], end=j)
  from tinygrad.uop.ops import KernelInfo
  return out[h].store(acc[0]).end(h).sink(arg=KernelInfo(name="canonical_recurrence", opts_to_apply=()))


x = np.random.default_rng(1).normal(size=(H * K,)).astype(np.float32)
got = Tensor.empty(H, dtype=dtypes.float32).custom_kernel(Tensor(x), fxn=kernel)[0].realize().numpy()
ref = x.reshape(H, K).sum(axis=1)
err = float(np.max(np.abs(got - ref)))
print(f"canonical recurrence: max_abs={err:.3e} -> {'PASS' if err < 1e-4 else 'FAIL'}")
raise SystemExit(0 if err < 1e-4 else 1)
