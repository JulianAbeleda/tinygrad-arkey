#!/usr/bin/env python3
"""Phase 1 — minimal SHAPED_WMMA custom_kernel smoke (decode-attention-v3 prerequisite).

Proves the revived WMMA custom-kernel idiom end-to-end on the smallest single-block matmul:
  - one 128x128x16 fp16xfp16->fp32 WMMA matmul via Tensor.custom_kernel (Ops.SHAPED_WMMA -> Ops.WMMA)
  - lowers to Ops.PROGRAM and runs on AMD (no CPU/CLANG fallback)
  - TinyJit capture + replay
  - replay with SUBSTITUTED inputs yields the new correct result (proves input substitution)
  - correctness vs numpy reference within fp tolerance

Revival required (see docs/qk-decode-attention-v3-prereq-*):
  (1) authoring: AFTER must wrap the placeholder/movement, not an INDEX -> acc.after(k)[idx]
  (2) codegen: a spec_tensor rule for Ops.SHAPED_WMMA (3 srcs, 3-tuple arg) so it survives the
      tensor-graph verify before lower_shaped_wmma converts it to Ops.WMMA.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_wmma_custom_smoke.py
"""
from __future__ import annotations
import io, json, os, pathlib, re, sys, contextlib

os.environ.setdefault("WMMA", "1")
os.environ.setdefault("M", "128"); os.environ.setdefault("N", "128"); os.environ.setdefault("K", "16")
import numpy as np
from tinygrad import Tensor, TinyJit, Device, Context, GlobalCounters
from tinygrad.dtype import dtypes
from extra.gemm.amd_copy_matmul import amd_copy_matmul, M, N, K  # noqa: E402  (module reads M/N/K from env)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROG = re.compile(r"\*\*\*\s+(\S+)\s+\d+\s+(.+?)\s+arg")

def _run(a:Tensor, b:Tensor) -> Tensor:
  c = Tensor.empty(M, N, dtype=dtypes.float)
  return Tensor.custom_kernel(c, a, b, fxn=amd_copy_matmul)[0]

def main():
  dev = Device.DEFAULT
  assert dev == "AMD", f"smoke requires AMD device, got {dev}"
  rng = np.random.default_rng(0)
  a0 = rng.standard_normal((M, K)).astype(np.float16); b0 = rng.standard_normal((K, N)).astype(np.float16)
  a1 = rng.standard_normal((M, K)).astype(np.float16); b1 = rng.standard_normal((K, N)).astype(np.float16)
  ref0, ref1 = a0.astype(np.float32) @ b0.astype(np.float32), a1.astype(np.float32) @ b1.astype(np.float32)
  ta0, tb0 = Tensor(a0).realize(), Tensor(b0).realize()
  ta1, tb1 = Tensor(a1).realize(), Tensor(b1).realize()

  jit = TinyJit(_run)
  # warm(0) + capture(1) + replay(2) on the SAME inputs, then replay(3) on SUBSTITUTED inputs
  outs = []
  for i in range(3): outs.append(jit(ta0, tb0).realize())
  got0 = outs[-1].numpy()
  # capture program/device proof on a replay
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); jit(ta0, tb0).realize()
  progs = [(m.group(1), m.group(2).strip()) for l in buf.getvalue().splitlines() if (m := _PROG.search(_ANSI.sub("", l)))]
  devices = sorted({d for d, _ in progs})
  # input substitution: replay with different inputs
  got1 = jit(ta1, tb1).numpy()

  err0 = float(np.abs(got0 - ref0).max()); err1 = float(np.abs(got1 - ref1).max())
  rel0 = float(np.abs(got0 - ref0).max() / (np.abs(ref0).max() + 1e-9))
  rel1 = float(np.abs(got1 - ref1).max() / (np.abs(ref1).max() + 1e-9))
  tol = 2e-2
  correct0, correct1 = rel0 < tol, rel1 < tol
  substitution_ok = correct1 and not np.allclose(got0, got1)  # replay actually used new inputs
  amd_only = devices == ["AMD"] and len(progs) >= 1
  num_jit_kernels = len(getattr(getattr(jit, "captured", None), "jit_cache", []) or [])
  passed = correct0 and correct1 and substitution_ok and amd_only

  out = {"shape": {"M": M, "N": N, "K": K}, "wmma": True,
         "compiles_runs_correct": correct0, "rel_err_run0": round(rel0, 6), "max_abs_err_run0": round(err0, 6),
         "tinyjit_replay_correct": correct1, "rel_err_run1": round(rel1, 6),
         "input_substitution_ok": bool(substitution_ok), "jit_cache_kernels": num_jit_kernels,
         "program_devices": devices, "num_program_kernels": len(progs), "no_cpu_fallback": amd_only,
         "verdict": "PASS" if passed else "FAIL"}
  print(json.dumps(out, indent=2), file=sys.stderr)
  art = pathlib.Path("bench/qk-wmma-custom-smoke/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"\nSMOKE {'PASS' if passed else 'FAIL'}  (correct={correct0}/{correct1} subst={substitution_ok} "
        f"devices={devices} kernels={len(progs)})  artifact: {art}", file=sys.__stderr__)
  print("@@DONE@@", file=sys.__stderr__)
  sys.exit(0 if passed else 1)

if __name__ == "__main__":
  main()
