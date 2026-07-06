#!/usr/bin/env python3
"""Lowering-feasibility gate for the Q4_K/Q8_1 tiled WMMA prefill route."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, textwrap
from typing import Any

from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "bench/q4k-wmma-tiled-lowering-feasibility/latest.json"

PROBE = r"""
import numpy as np
from tinygrad import Tensor, dtypes
M_TILE=N_TILE=16
GROUP_ELEMS=32
rng=np.random.default_rng(7)
q8=rng.integers(-128,127,size=(M_TILE,GROUP_ELEMS),dtype=np.int8)
q4=rng.integers(0,16,size=(N_TILE,GROUP_ELEMS),dtype=np.int8)
out=Tensor(q8).matmul(Tensor(q4).transpose(), dtype=dtypes.int).realize().numpy()
ref=q8.astype(np.int32) @ q4.astype(np.int32).T
print("shape", out.shape)
print("max_abs", int(np.max(np.abs(out-ref))))
"""


def _run_probe() -> dict[str, Any]:
  env = {**os.environ, "DEV": "AMD", "TC": "1", "TC_OPT": "1", "DEBUG": "4",
         "ALLOW_DEVICE_USAGE": "1", "PYTHONPATH": str(ROOT)}
  r = subprocess.run([sys.executable, "-c", PROBE], cwd=str(ROOT), env=env,
                     capture_output=True, text=True, timeout=120)
  combined = r.stdout + "\n" + r.stderr
  return {"argv": [sys.executable, "-c", textwrap.dedent(PROBE).strip()], "returncode": r.returncode,
          "stdout_tail": r.stdout[-5000:], "stderr_tail": r.stderr[-5000:],
          "has_iu8_wmma": "wmma_i32_16x16x16_iu8" in combined,
          "max_abs_ok": "max_abs 0" in combined}


def build() -> dict[str, Any]:
  spec = describe_q4k_int8_wmma_tiled_prefill(16, 256, 16, role="lowering_feasibility",
                                             m_tile=16, n_tile=16, group_tile=1)
  probe = _run_probe()
  bounded_raw_ok = spec.live_raw_elems <= spec.m_tile * spec.n_tile * spec.group_tile
  ok = probe["returncode"] == 0 and probe["has_iu8_wmma"] and probe["max_abs_ok"] and bounded_raw_ok
  return {"schema": "q4k_wmma_tiled_lowering_feasibility.v1",
          "scope": "bounded Q4_K/Q8_1 RAW tile expressed as int8 Tensor.matmul lowers to RDNA3 iu8 WMMA",
          "verdict": "Q4K_WMMA_TILED_LOWERING_FEASIBLE" if ok else "Q4K_WMMA_TILED_LOWERING_BLOCKED",
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "implementation": spec.implementation,
          "tile": {"m_tile": spec.m_tile, "n_tile": spec.n_tile, "group_tile": spec.group_tile,
                   "group_elems": spec.group_elems, "live_raw_elems": spec.live_raw_elems,
                   "forbidden_full_raw_shape": [spec.groups, spec.m, spec.n],
                   "forbidden_full_raw_elems": spec.forbidden_full_raw_elems,
                   "bounded_raw_ok": bounded_raw_ok},
          "tc_env": {"DEV": "AMD", "TC": "1", "TC_OPT": "1"},
          "probe": probe}


if __name__ == "__main__":
  out = build()
  ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
  ARTIFACT.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_LOWERING_FEASIBLE" else 1)
