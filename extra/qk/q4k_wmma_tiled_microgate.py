#!/usr/bin/env python3
"""One-tile Q4_K/Q8_1 WMMA tiled microgate.

This validates the Phase-2 bounded tile, including Q4_K scale/min correction, against the existing q8-dequant
reference. It is intentionally not a full-role route gate.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, textwrap
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]

PROBE = r"""
import numpy as np
from tinygrad import Tensor, dtypes
from extra.qk.layout import q8_1_quantize
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_prefill_tensor

n,k,m = 16,256,16
words, ref_w = _make_q4k_words(n, k, 20260705)
x = Tensor(np.random.default_rng(20260706).standard_normal((m, k)).astype(np.float32)).realize()
xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) * xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
ref_out = (x_dq @ ref_w.T).numpy()
spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="microgate", m_tile=16, n_tile=16, group_tile=8)
got = emit_q4k_int8_wmma_tiled_prefill_tensor(words, xq, xscales, spec).realize().numpy()
rel = _rel_rmse(got, ref_out)
print("rel_rmse", rel)
print("rtol", RTOL)
print("live_raw_elems", spec.live_raw_elems)
print("forbidden_full_raw_elems", spec.forbidden_full_raw_elems)
if rel >= RTOL:
  raise SystemExit(2)
"""


def _run_probe() -> dict[str, Any]:
  env = dict(os.environ)
  env.update({"DEV": "AMD", "TC": "1", "TC_OPT": "1", "DEBUG": "4",
              "ALLOW_DEVICE_USAGE": "1", "PYTHONPATH": str(ROOT)})
  r = subprocess.run([sys.executable, "-c", PROBE], cwd=str(ROOT), env=env,
                     capture_output=True, text=True, timeout=180)
  combined = r.stdout + "\n" + r.stderr
  return {"argv": [sys.executable, "-c", textwrap.dedent(PROBE).strip()], "returncode": r.returncode,
          "stdout_tail": r.stdout[-6000:], "stderr_tail": r.stderr[-6000:],
          "has_iu8_wmma": "wmma_i32_16x16x16_iu8" in combined,
          "numeric_ok": r.returncode == 0 and "rel_rmse" in combined}


def build() -> dict[str, Any]:
  probe = _run_probe()
  ok = probe["returncode"] == 0 and probe["has_iu8_wmma"] and probe["numeric_ok"]
  return {"schema": "q4k_wmma_tiled_microgate.v1",
          "scope": "one bounded Q4_K/Q8_1 tiled WMMA output tile with scale/min correction",
          "verdict": "Q4K_WMMA_TILED_MICROGATE_PASS" if ok else "Q4K_WMMA_TILED_MICROGATE_FAIL",
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "tile": {"m_tile": 16, "n_tile": 16, "group_tile": 8, "group_elems": 32,
                   "live_raw_elems": 2048, "forbidden_full_raw_shape": [8, 16, 16],
                   "forbidden_full_raw_elems": 2048},
          "probe": probe}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_MICROGATE_PASS" else 1)
