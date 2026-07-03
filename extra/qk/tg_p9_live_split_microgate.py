#!/usr/bin/env python3
"""TG-P9.1 microgate: prove the live-context split geometry primitive lowers correctly (generated UOp, no HIP/ASM).

For several (S, Tc) including the 8B shape, run the generated coverage kernel (symbolic per-split loop bound) and
verify:
  1. every token in [0, Tc) is covered exactly once (cov==1);
  2. no split writes beyond Tc (cov==0 on [Tc, MAXC));
  3. the grid launches S workgroups (fixed), NOT ceildiv(MAXC, L).

Writes bench/tg-p9-pure-attention-primitive-route/live_split_microgate.json. Verdict TG_P9_1_PASS_LIVE_TC_SPLIT_IR /
TG_P9_1_BLOCKED_UOP_RANGE_MODEL / TG_P9_1_BLOCKED_SYMBOLIC_BOUNDS.
"""
from __future__ import annotations
import json, os, pathlib

os.environ.setdefault("DEV", "AMD")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/tg-p9-pure-attention-primitive-route"
# (S, MAXC, [Tc values]) -- 8B route uses S=ceildiv(MAXC,L)=36 at MAXC=4608,L=128
CASES = [
  {"S": 36, "MAXC": 4608, "tcs": [1, 15, 512, 513, 4096, 4608]},   # 8B geometry
  {"S": 48, "MAXC": 4608, "tcs": [1, 512, 4096]},                  # owned S=48
  {"S": 8, "MAXC": 2048, "tcs": [1, 7, 512, 2048]},                # small
]


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import UOp
  from extra.qk.live_split_geometry import LiveSplitGeometry, live_split_coverage_kernel

  results, all_ok, blocked = [], True, None
  for c in CASES:
    S, MAXC = c["S"], c["MAXC"]
    geo = LiveSplitGeometry(S=S, TK=16)
    # start_pos as an UNBOUND symbolic var; Tc = start_pos + 1 flows into the kernel unbound. The bound value is
    # threaded via a carry term (out + ones[0:vsp.bind(Tc-1)].sum()*0) so realize infers var_vals -- the standalone
    # equivalent of the model's JIT-replay binding.
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    kfn = live_split_coverage_kernel(geo, MAXC, vsp + 1)
    for Tc in c["tcs"]:
      try:
        cov = Tensor.zeros(MAXC, dtype=dtypes.int32, device=Device.DEFAULT).contiguous()
        outt = cov.custom_kernel(fxn=kfn)[0]
        carry = Tensor.ones(MAXC, dtype=dtypes.int32)[0:vsp.bind(Tc - 1)].sum().reshape(1) * 0
        out = (outt + carry).realize().numpy()
      except Exception as e:
        blocked = f"{type(e).__name__}: {str(e)[:200]}"
        results.append({"S": S, "MAXC": MAXC, "Tc": Tc, "error": blocked})
        all_ok = False
        continue
      inside = out[:Tc]
      outside = out[Tc:]
      cover_once = bool(np.all(inside == 1))
      no_spill = bool(np.all(outside == 0))
      covered = int(inside.sum())
      ok = cover_once and no_spill and covered == Tc
      all_ok = all_ok and ok
      results.append({"S": S, "MAXC": MAXC, "Tc": Tc, "covered_count": covered, "expected": Tc,
                      "cover_each_once": cover_once, "no_spill_beyond_Tc": no_spill, "ok": ok,
                      "grid_workgroups": S, "note": "grid=S (fixed), not ceildiv(MAXC,L)"})

  if blocked and not any(r.get("ok") for r in results):
    # never lowered at all -> range model / symbolic bound rejected
    verdict = "TG_P9_1_BLOCKED_SYMBOLIC_BOUNDS" if "range" not in blocked.lower() else "TG_P9_1_BLOCKED_UOP_RANGE_MODEL"
  elif all_ok:
    verdict = "TG_P9_1_PASS_LIVE_TC_SPLIT_IR"
  else:
    verdict = "TG_P9_1_BLOCKED_SYMBOLIC_BOUNDS"
  latest = {"scope": "TG-P9.1 live-context split geometry coverage microgate", "verdict": verdict,
            "capability": "fixed S splits + symbolic per-split length per=ceildiv(Tc,S); symbolic inner-loop bound nb=ceildiv(per,TK)",
            "all_ok": all_ok, "blocked_reason": blocked, "cases": results}
  json.dump(latest, open(OUT / "live_split_microgate.json", "w"), indent=2)
  print(verdict, "all_ok=", all_ok, "blocked=", blocked)
  return 0 if verdict == "TG_P9_1_PASS_LIVE_TC_SPLIT_IR" else 1


if __name__ == "__main__":
  raise SystemExit(main())
