#!/usr/bin/env python3
"""AMD gate for the generated iu8 WMMA substrate.

This intentionally probes ordinary Tensor matmul, not a handwritten kernel. Passing means tinygrad codegen can lower
`matmul(..., dtype=dtypes.int)` on int8 operands to RDNA3 iu8 WMMA and produce correct int32 results.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]


PROBE = r"""
import numpy as np
from tinygrad import Tensor, dtypes
M=N=K=16
rng=np.random.default_rng(0)
a=rng.integers(-128,127,size=(M,K),dtype=np.int8)
b=rng.integers(0,16,size=(N,K),dtype=np.int8)
out=Tensor(a).matmul(Tensor(b).transpose(), dtype=dtypes.int).realize().numpy()
ref=a.astype(np.int32) @ b.astype(np.int32).T
print("max_abs", int(np.max(np.abs(out-ref))))
"""


def build() -> dict:
  env = dict(os.environ)
  env.update({"DEV": "AMD", "TC": "1", "TC_OPT": "1", "DEBUG": "4", "ALLOW_DEVICE_USAGE": "1", "PYTHONPATH": str(ROOT)})
  r = subprocess.run([sys.executable, "-c", PROBE], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
  combined = r.stdout + "\n" + r.stderr
  has_wmma = "wmma_i32_16x16x16_iu8" in combined
  max_abs_ok = "max_abs 0" in combined
  verdict = "INT8_WMMA_CODEGEN_PASS" if r.returncode == 0 and has_wmma and max_abs_ok else "INT8_WMMA_CODEGEN_BLOCKED"
  return {"scope": "int8 Tensor.matmul codegen lowers to RDNA3 iu8 WMMA", "verdict": verdict,
          "returncode": r.returncode, "has_iu8_wmma": has_wmma, "max_abs_ok": max_abs_ok,
          "probe": textwrap.dedent(PROBE).strip(),
          "stdout_tail": r.stdout[-4000:], "stderr_tail": r.stderr[-4000:]}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "INT8_WMMA_CODEGEN_PASS" else 1)
