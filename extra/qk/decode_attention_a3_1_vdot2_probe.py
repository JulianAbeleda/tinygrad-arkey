#!/usr/bin/env python3
"""A3.1 v_dot2 lowering probe.

This answers the first question before wiring attention:
can generated tinygrad code expose AMD v_dot2 at all?
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-a3-1-vdot2"


def static_hook_check() -> dict[str, Any]:
  files = {
    "lowering": ROOT / "extra/qk/fdot2_lowering.py",
    "codegen": ROOT / "tinygrad/codegen/__init__.py",
    "renderer": ROOT / "tinygrad/renderer/cstyle.py",
  }
  txt = {k: p.read_text() if p.exists() else "" for k, p in files.items()}
  return {
    "lowering_file_exists": files["lowering"].exists(),
    "codegen_env_hook": "V_DOT2_LOWERING" in txt["codegen"],
    "fdot2_builtin_template": "__builtin_amdgcn_fdot2" in txt["lowering"],
    "customi_renderer": "Ops.CUSTOMI" in txt["renderer"],
  }


def matcher_unit_check() -> dict[str, Any]:
  from tinygrad.dtype import dtypes
  from tinygrad.uop.ops import Ops, UOp
  from extra.qk.fdot2_lowering import lower_fdot2_add

  a = UOp(Ops.DEFINE_VAR, dtypes.half.vec(2), arg=("a", 0, 1))
  b = UOp(Ops.DEFINE_VAR, dtypes.half.vec(2), arg=("b", 0, 1))
  ax = UOp(Ops.INDEX, dtypes.half, (a, UOp.const(dtypes.int, 0)))
  ay = UOp(Ops.INDEX, dtypes.half, (a, UOp.const(dtypes.int, 1)))
  bx = UOp(Ops.INDEX, dtypes.half, (b, UOp.const(dtypes.int, 0)))
  by = UOp(Ops.INDEX, dtypes.half, (b, UOp.const(dtypes.int, 1)))
  term0 = UOp(Ops.CAST, dtypes.float, (UOp(Ops.MUL, dtypes.half, (ax, bx)),))
  term1 = UOp(Ops.CAST, dtypes.float, (UOp(Ops.MUL, dtypes.half, (ay, by)),))
  add = UOp(Ops.ADD, dtypes.float, (term0, term1))
  lowered = lower_fdot2_add(add)
  return {
    "matcher_rewrites_dot2_pair": lowered is not None,
    "lowered_op": str(lowered.op) if lowered is not None else None,
    "lowered_arg": lowered.arg if lowered is not None else None,
    "uses_builtin": bool(lowered is not None and "__builtin_amdgcn_fdot2" in str(lowered.arg)),
  }


def generated_kernel_smoke() -> dict[str, Any]:
  """Try a tiny generated Tensor dot2 with the existing opt-in lowering.

  This is intentionally permissive: if compilation/running fails, the artifact records the failure instead of hiding
  it. DEBUG output is scanned for the builtin because the exact code-object cache path is backend-dependent.
  """
  code = r'''
import json, os
os.environ["V_DOT2_LOWERING"] = "1"
from tinygrad import Tensor, dtypes
a = Tensor([1.0, 2.0], dtype=dtypes.float16)
b = Tensor([3.0, 4.0], dtype=dtypes.float16)
o = (a*b).sum().realize()
print(json.dumps({"value": float(o.item())}))
'''
  env = {**os.environ, "PYTHONPATH": str(ROOT), "V_DOT2_LOWERING": "1", "DEBUG": "4"}
  r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
  combined = r.stdout + "\n" + r.stderr
  return {
    "returncode": r.returncode,
    "ran": r.returncode == 0,
    "stdout_tail": r.stdout[-2000:],
    "stderr_tail": r.stderr[-4000:],
    "debug_mentions_builtin": "__builtin_amdgcn_fdot2" in combined,
    "debug_mentions_v_dot2": "v_dot2" in combined,
  }


def build() -> dict[str, Any]:
  static = static_hook_check()
  matcher = matcher_unit_check()
  smoke = generated_kernel_smoke() if all(static.values()) and matcher["uses_builtin"] else {
    "ran": False,
    "skip_reason": "static hook or matcher check failed",
  }
  if not all(static.values()):
    verdict = "A3_1_BLOCKED_BY_RENDERER"
  elif not matcher["uses_builtin"]:
    verdict = "A3_1_BLOCKED_BY_MATCHER"
  elif smoke.get("debug_mentions_builtin") or smoke.get("debug_mentions_v_dot2"):
    verdict = "A3_1_RENDERER_VDOT2_PROBE_PASS"
  else:
    verdict = "A3_1_RENDERER_VDOT2_PROBE_INCONCLUSIVE"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "static_hook_check": static,
    "matcher_unit_check": matcher,
    "generated_kernel_smoke": smoke,
    "decision": (
      "Generated v_dot2 hook appears exposable; next wire DECODE_ATTN_SCORE_VDOT2=1 against flash_score_whole_cache."
      if verdict == "A3_1_RENDERER_VDOT2_PROBE_PASS" else
      "Do not wire attention score v_dot2 until the probe produces concrete generated-code evidence."
    ),
  }


def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-1-vdot2-probe-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] in ("A3_1_RENDERER_VDOT2_PROBE_PASS", "A3_1_RENDERER_VDOT2_PROBE_INCONCLUSIVE") else 1


if __name__ == "__main__":
  raise SystemExit(main())
