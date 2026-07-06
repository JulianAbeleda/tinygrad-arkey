#!/usr/bin/env python3
"""Scheduler-surface gate for reusable Q4_K/Q8_1 tiled WMMA lowering."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, textwrap
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]

SHAPED_PROBE = r"""
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.schedule.wmma import shaped_wmma

def shaped_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  afrags = [a[((lane & 15) * 16) + i] for i in range(16)]
  bfrags = [b[((lane & 15) * 16) + i] for i in range(16)]
  avec = afrags[0].vectorize(*afrags[1:])
  bvec = bfrags[0].vectorize(*bfrags[1:])
  zero = UOp.const(dtypes.int32, 0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(avec, bvec, acc, dims=(16, 16, 16), device="AMD", threads=32)
  stores = [out[lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).sink(arg=KernelInfo(name="q4k_scheduler_shaped_wmma_probe", opts_to_apply=()))

rng = np.random.default_rng(20260709)
a = rng.integers(-128, 127, size=(256,), dtype=np.int8)
b = rng.integers(0, 16, size=(256,), dtype=np.int8)
out = Tensor.empty(256, dtype=dtypes.int32).custom_kernel(Tensor(a), Tensor(b), fxn=shaped_kernel)[0]
print(out.realize().numpy()[:8])
"""


def _run_shaped_probe() -> dict[str, Any]:
  env = dict(os.environ)
  env.update({"DEV": "AMD", "DEBUG": "4", "ALLOW_DEVICE_USAGE": "1", "PYTHONPATH": str(ROOT)})
  r = subprocess.run([sys.executable, "-c", SHAPED_PROBE], cwd=str(ROOT), env=env,
                     capture_output=True, text=True, timeout=120)
  combined = r.stdout + "\n" + r.stderr
  if "Ops.CONST dtypes.weakint" in combined and "UOp verification failed" in combined:
    cls = "blocked.program_verifier_weak_index_const"
  elif "failed to render Ops.RESHAPE dtypes.char" in combined:
    cls = "blocked.renderer_fragment_load_shape"
  elif "UOp verification failed" in combined:
    cls = "blocked.verifier"
  elif r.returncode:
    cls = "blocked.probe_failed"
  else:
    cls = "available"
  return {"argv": [sys.executable, "-c", textwrap.dedent(SHAPED_PROBE).strip()],
          "returncode": r.returncode, "class": cls,
          "stdout_tail": r.stdout[-5000:], "stderr_tail": r.stderr[-5000:],
          "has_iu8_wmma": "wmma_i32_16x16x16_iu8" in combined,
          "uses_scheduler_helper": "tinygrad.schedule.wmma" in SHAPED_PROBE}


def build() -> dict[str, Any]:
  shaped = _run_shaped_probe()
  shaped_ok = shaped["returncode"] == 0 and shaped["has_iu8_wmma"]
  verdict = "Q4K_WMMA_SCHEDULER_SURFACE_SHAPED_READY" if shaped_ok else \
    "Q4K_WMMA_SCHEDULER_SURFACE_TC_MATCHER_REQUIRED"
  return {"schema": "q4k_wmma_scheduler_surface_gate.v1",
          "scope": "decide whether full-role Q4_K/Q8_1 WMMA should use explicit SHAPED_WMMA or TC matcher surface",
          "verdict": verdict,
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "surfaces": {
            "tc_matcher_tile": {"class": "available", "reason": "existing tiled lowering feasibility gate proves int8 Tensor.matmul reaches iu8 WMMA"},
            "shaped_wmma_tile": {"class": shaped["class"], "probe": shaped,
                                  "reason": "scheduler helper exists, but SHAPED_WMMA custom-kernel fragment lowering is not verifier-clean" if not shaped_ok
                                            else "scheduler helper lowers to iu8 WMMA"}},
          "selected_surface": "shaped_wmma_tile" if shaped_ok else "tc_matcher_tile",
          "next_required": "keep Q4_K full-role work on TC matcher surface until shaped-WMMA fragment rendering is fixed" if not shaped_ok
                           else "build Q4_K full-role tile lifecycle with tinygrad.schedule.wmma.shaped_wmma"}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0)
