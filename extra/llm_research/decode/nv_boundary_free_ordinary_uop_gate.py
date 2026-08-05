#!/usr/bin/env python3
"""Phase-0 gate for a boundary-free, ordinary-UOp decode RMSNorm route.

This is intentionally a construction gate, not another custom-kernel benchmark.
It establishes the exact ordinary scheduler topology for both a realized input and
a lazy producer view.  A candidate may advance only if it produces one replayable
ordinary program in both cases without a custom-program boundary or CONTIGUOUS.
"""
from __future__ import annotations

import json
from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops

DIM = 4096

def _programs(out: Tensor) -> list[str]:
  linear, _ = out.linear_with_vars()
  return [x.src[0].arg.name for x in linear.src]

def _ordinary(x: Tensor) -> Tensor:
  norm = nn.RMSNorm(DIM, eps=1e-6)
  norm.weight = Tensor.randn(DIM, dtype=dtypes.float16, device=x.device).realize()
  norm._rmsnorm_native_promoted = False
  return norm(x)

def _contains_op(x: Tensor, op: Ops) -> bool:
  return any(u.op is op for u in x.uop.toposort())

def run() -> dict:
  base = Tensor.randn(1, DIM, dtype=dtypes.float16).realize()
  rows = {}
  for label, x in (("realized", base), ("lazy_add", base + base)):
    out = _ordinary(x)
    programs = _programs(out)
    rows[label] = {
      "programs": programs,
      "program_count": len(programs),
      "contains_custom_kernel": _contains_op(out, Ops.CUSTOM),
      "contains_contiguous": _contains_op(out, Ops.CONTIGUOUS),
    }
  return {
    "schema": "tinygrad.nv_boundary_free_ordinary_uop_gate.v1",
    "contract": {
      "shape": [1, DIM],
      "candidate_must_be_ordinary_uop": True,
      "candidate_must_not_materialize_lazy_input": True,
      "candidate_must_be_replay_profile_visible": True,
    },
    "baseline": rows,
    "verdict": "CONSTRUCTION_GAP",
    "reason": "ordinary RMSNorm remains a reduction program plus a dependent epilogue program; no generic cross-thread reduction-to-output scheduler primitive exists",
  }

if __name__ == "__main__": print(json.dumps(run(), indent=2, sort_keys=True))
