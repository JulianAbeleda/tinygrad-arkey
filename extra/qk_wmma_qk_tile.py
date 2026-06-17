#!/usr/bin/env python3
"""Phase 3 — one Q@K^T WMMA tile (decode-attention-v3 prerequisite).

scores[m,n] = sum_d Q[m,d]*K[n,d] = Q @ K^T -- exactly the WMMA A@B^T contraction. Verifies the revived
SHAPED_WMMA path computes attention SCORES correctly via WMMA, at Hd=16 (one WMMA-K chunk) then Hd=128
(8 chunks), through TinyJit (capture/replay). No softmax, no V, no GQA, no causal (those are Phase 4+).

Uses the revived WMMA matmul kernel (extra/gemm/amd_copy_matmul) with B = K (so c = Q @ K^T): the kernel
computes c[m,n] = sum_k a[m,k]*b[k,n]; passing b = K^T gives Q @ K^T. The single FUSED scores->softmax->V
kernel is Phase 4; this rung only gates Q@K^T expressibility/correctness via WMMA.

Run: DEV=AMD QK_HD=16 PYTHONPATH=. .venv/bin/python extra/qk_wmma_qk_tile.py   (then QK_HD=128)
"""
from __future__ import annotations
import io, json, os, pathlib, re, sys, contextlib

HD = int(os.environ.get("QK_HD", "16"))
M = int(os.environ.get("QK_M", "128"))   # query rows (WMMA tile multiple of 16)
NKV = int(os.environ.get("QK_N", "128")) # KV-length tile (multiple of 16)
os.environ.setdefault("WMMA", "1")
os.environ["M"], os.environ["N"], os.environ["K"] = str(M), str(NKV), str(HD)
import numpy as np
from tinygrad import Tensor, TinyJit, Device, Context, GlobalCounters
from tinygrad.dtype import dtypes
from extra.gemm.amd_copy_matmul import amd_copy_matmul  # noqa: E402 (reads M/N/K from env at import)

_ANSI = re.compile(r"\x1b\[[0-9;]*m"); _PROG = re.compile(r"\*\*\*\s+(\S+)\s+\d+\s+(.+?)\s+arg")

def _scores(q:Tensor, kT:Tensor) -> Tensor:           # kT is K transposed to [Hd, NKV]
  c = Tensor.empty(M, NKV, dtype=dtypes.float)
  return Tensor.custom_kernel(c, q, kT, fxn=amd_copy_matmul)[0]

def main():
  assert Device.DEFAULT == "AMD"
  rng = np.random.default_rng(0)
  q0 = rng.standard_normal((M, HD)).astype(np.float16); k0 = rng.standard_normal((NKV, HD)).astype(np.float16)
  q1 = rng.standard_normal((M, HD)).astype(np.float16); k1 = rng.standard_normal((NKV, HD)).astype(np.float16)
  ref0 = q0.astype(np.float32) @ k0.astype(np.float32).T     # Q @ K^T
  ref1 = q1.astype(np.float32) @ k1.astype(np.float32).T
  tq0, tkT0 = Tensor(q0).realize(), Tensor(k0.T.copy()).realize()  # kT = [Hd, NKV]
  tq1, tkT1 = Tensor(q1).realize(), Tensor(k1.T.copy()).realize()

  jit = TinyJit(_scores)
  for _ in range(3): got0 = jit(tq0, tkT0).realize()
  got0 = got0.numpy()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); jit(tq0, tkT0).realize()
  progs = [(m.group(1), m.group(2).strip()) for l in buf.getvalue().splitlines() if (m := _PROG.search(_ANSI.sub("", l)))]
  devices = sorted({d for d, _ in progs})
  got1 = jit(tq1, tkT1).numpy()

  rel0 = float(np.abs(got0 - ref0).max() / (np.abs(ref0).max() + 1e-9))
  rel1 = float(np.abs(got1 - ref1).max() / (np.abs(ref1).max() + 1e-9))
  tol = 2e-2
  passed = rel0 < tol and rel1 < tol and devices == ["AMD"] and not np.allclose(got0, got1)
  out = {"Hd": HD, "M": M, "N_kv": NKV, "rel_err_scores": round(rel0, 6), "rel_err_replay": round(rel1, 6),
         "program_devices": devices, "num_program_kernels": len(progs), "no_cpu_fallback": devices == ["AMD"],
         "input_substitution_ok": bool(not np.allclose(got0, got1)), "verdict": "PASS" if passed else "FAIL"}
  print(json.dumps(out, indent=2), file=sys.stderr)
  art = pathlib.Path(f"bench/qk-wmma-qk-tile/result_hd{HD}.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"\nQK-TILE Hd={HD} {'PASS' if passed else 'FAIL'} (rel_err {rel0:.2g}/{rel1:.2g} dev={devices})  {art}", file=sys.__stderr__)
  print("@@DONE@@", file=sys.__stderr__)
  sys.exit(0 if passed else 1)

if __name__ == "__main__":
  main()
