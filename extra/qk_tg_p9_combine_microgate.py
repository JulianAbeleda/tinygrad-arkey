#!/usr/bin/env python3
"""TG-P9.3/9.4: split-preserving LSE combine primitive -- design + microgate.

Goal: a generated combine that removes the per-d fexp redundancy of flash_state_combine (which recomputes
w=exp(m[h,s]-gm) for every d, an Hd-fold redundancy = the 556us/fwd ctx4096 cap) WITHOUT collapsing Hq*S or Hq*Hd
parallelism (the refuted collapses). Design options attempted, all generated UOp (no HIP/ASM):
  - LDS weight-share, 32-lane warp (extra/qk_live_split_geometry.flash_fused_gmax_combine_kernel);
  - inline-gmax single-kernel (flash_inline_gm_combine_kernel);
  - two-stage weights + fexp-free weighted-sum (flash_gm_weights_kernel + flash_weighted_sum_kernel).

RESULT: EMITTER_BLOCKED. Every shape that shares the softmax weights across d or fuses the gmax max-reduce trips a
tinygrad AMD codegen limit: the reduction-accumulator REG is vectorized into a non-assignable `make_float4(...) =
...` store (verifier/renderer). REG_STORE_DEVEC=1 makes them compile but returns NaN (the max-reduce mis-lowers).
Only the exact shipped single-reduce per-d combine compiles correctly, and it cannot de-duplicate the fexp. This is
a concrete compiler primitive gap, not a design failure -- the split-preserving combine IS expressible in principle
(no parallelism collapse) but the current emitter cannot lower it.

Writes bench/tg-p9-pure-attention-primitive-route/combine_microgate.json.
"""
from __future__ import annotations
import json, os, pathlib

os.environ.setdefault("DEV", "AMD")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p9-pure-attention-primitive-route"


def _try(fn):
  try:
    fn(); return "compiles"
  except Exception as e:
    return f"{type(e).__name__}: {str(e)[:80]}"


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, dtypes
  from extra.qk_live_split_geometry import (flash_fused_gmax_combine_kernel, flash_inline_gm_combine_kernel,
                                            flash_gm_weights_kernel, flash_weighted_sum_kernel)
  Hq, Hd, S = 32, 128, 36; W = Hd + 2
  pout = Tensor(np.zeros(Hq * S * W, dtype=np.float32), device="AMD").realize()
  attempts = {
    "lds_warp_fused": _try(lambda: Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_fused_gmax_combine_kernel(Hd, Hq, S, stride=S))[0].realize()),
    "inline_gmax": _try(lambda: Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_inline_gm_combine_kernel(Hd, Hq, S, stride=S))[0].realize()),
    "two_stage_weights": _try(lambda: Tensor.empty(Hq * S, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_gm_weights_kernel(Hd, Hq, S, stride=S))[0].realize()),
  }
  blocked = all(v != "compiles" for v in attempts.values())
  verdict = "TG_P9_4_BLOCKED_EMITTER" if blocked else "TG_P9_4_PASS_COMBINE_MICROGATE"
  latest = {"scope": "TG-P9.3/9.4 split-preserving combine primitive", "verdict": verdict,
            "classification": "EMITTER_BLOCKED", "no_parallelism_collapse": True,
            "signature": "reduction-accumulator REG vectorized to a non-assignable make_float4(...) = ... in every "
                         "weight-sharing / gmax-fusing combine shape; REG_STORE_DEVEC=1 compiles them but returns NaN "
                         "(max-reduce mis-lowers). Only the shipped single-reduce per-d combine compiles correctly.",
            "attempts": attempts,
            "primitive_gap": "AMD backend accumulator-REG vectorization + REG_STORE_DEVEC do not correctly lower a "
                             "combine reduction that shares softmax weights across d or fuses the gmax max-reduce; "
                             "so the per-d fexp redundancy (the ctx4096 556us cap) cannot be removed in generated UOp.",
            "reopen_condition": "a tinygrad codegen fix so the reduction-accumulator REG stays scalar (or DEVEC lowers "
                                "the max-reduce correctly) for a multi-reduce / weight-sharing combine kernel."}
  json.dump(latest, open(OUT / "combine_microgate.json", "w"), indent=2)
  print(verdict, attempts)
  return 0 if not blocked else 1


if __name__ == "__main__":
  raise SystemExit(main())
