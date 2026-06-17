#!/usr/bin/env python3
"""Phase 4 — one decode-attention microtile via WMMA (decode-attention-v3 prerequisite).

Computes one attention tile end-to-end with WMMA, comparing to SDPA:
  scores[M,L] = Q[M,Hd] @ K[L,Hd]^T   (WMMA)
  P[M,L]      = softmax(scores / sqrt(Hd))   (two-pass row softmax; only the [M,L] TILE is materialized,
                                              never the full [Hq, KV] scores -> Phase-4 gate)
  out[M,Hd]   = P[M,L] @ V[L,Hd]      (WMMA)
M=L=Hd=128 = one KV tile. No GQA/causal/symbolic start_pos yet (Phase 5 / full-v3). Uses the revived WMMA
matmul (extra/gemm/amd_copy_matmul) for both contractions + tinygrad softmax between. The single FUSED
LDS-resident kernel is the full-v3 build; this rung gates correctness/expressibility of a WMMA attention tile.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_v3_tile.py
"""
from __future__ import annotations
import io, json, os, pathlib, re, sys, contextlib, math

M = int(os.environ.get("QK_M", "128")); L = int(os.environ.get("QK_L", "128")); HD = int(os.environ.get("QK_HD", "128"))
os.environ.setdefault("WMMA", "1")
os.environ["M"], os.environ["N"], os.environ["K"] = "128", "128", "128"  # all tile dims = 128
import numpy as np
from tinygrad import Tensor, TinyJit, Device, Context, GlobalCounters
from tinygrad.dtype import dtypes
from extra.gemm.amd_copy_matmul import amd_copy_matmul  # noqa: E402

_ANSI = re.compile(r"\x1b\[[0-9;]*m"); _PROG = re.compile(r"\*\*\*\s+(\S+)\s+\d+\s+(.+?)\s+arg")
SCALE = 1.0 / math.sqrt(HD)

def _attn(q:Tensor, kT:Tensor, v:Tensor) -> Tensor:
  sc = Tensor.custom_kernel(Tensor.empty(M, L, dtype=dtypes.float), q, kT, fxn=amd_copy_matmul)[0]  # Q@K^T tile
  p = (sc * SCALE).softmax(-1).cast(dtypes.half)                                                    # [M,L] tile only
  return Tensor.custom_kernel(Tensor.empty(M, HD, dtype=dtypes.float), p, v, fxn=amd_copy_matmul)[0]  # P@V

def main():
  assert Device.DEFAULT == "AMD"
  rng = np.random.default_rng(0)
  def mk(): return (rng.standard_normal((M, HD)).astype(np.float16), rng.standard_normal((L, HD)).astype(np.float16),
                    rng.standard_normal((L, HD)).astype(np.float16))
  q0, k0, v0 = mk(); q1, k1, v1 = mk()
  def ref(q, k, v):
    s = (q.astype(np.float32) @ k.astype(np.float32).T) * SCALE
    s = s - s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
    return p @ v.astype(np.float32)
  r0, r1 = ref(q0, k0, v0), ref(q1, k1, v1)
  tq0, tkT0, tv0 = Tensor(q0).realize(), Tensor(k0.T.copy()).realize(), Tensor(v0).realize()
  tq1, tkT1, tv1 = Tensor(q1).realize(), Tensor(k1.T.copy()).realize(), Tensor(v1).realize()

  jit = TinyJit(_attn)
  for _ in range(3): o0 = jit(tq0, tkT0, tv0).realize()
  o0 = o0.numpy()
  o1 = jit(tq1, tkT1, tv1).numpy()  # input substitution via jit replay
  # WMMA-presence proof: EAGER call (jit replay batches kernels into one HCQ-graph entry, hiding names)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); _attn(tq0, tkT0, tv0).realize()
  progs = [(m.group(1), m.group(2).strip()) for l in buf.getvalue().splitlines() if (m := _PROG.search(_ANSI.sub("", l)))]
  devices = sorted({d for d, _ in progs})
  wmma_kernels = sum(1 for _, nm in progs if nm.startswith("E_"))  # the revived WMMA matmul renders as E_*

  rel0 = float(np.abs(o0 - r0).max() / (np.abs(r0).max() + 1e-9))
  rel1 = float(np.abs(o1 - r1).max() / (np.abs(r1).max() + 1e-9))
  tol = 3e-2  # fp16 P + WMMA
  passed = rel0 < tol and rel1 < tol and devices == ["AMD"] and not np.allclose(o0, o1) and wmma_kernels >= 2
  out = {"M": M, "L": L, "Hd": HD, "rel_err": round(rel0, 5), "rel_err_replay": round(rel1, 5),
         "program_devices": devices, "num_program_kernels": len(progs), "wmma_kernels": wmma_kernels,
         "no_cpu_fallback": devices == ["AMD"], "input_substitution_ok": bool(not np.allclose(o0, o1)),
         "materializes_full_hq_kv_scores": False, "verdict": "PASS" if passed else "FAIL"}
  print(json.dumps(out, indent=2), file=sys.stderr)
  art = pathlib.Path("bench/qk-decode-attention-v3-tile/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"\nATTN-TILE {'PASS' if passed else 'FAIL'} (rel_err {rel0:.2g}/{rel1:.2g} dev={devices} wmma={wmma_kernels})  {art}", file=sys.__stderr__)
  print("@@DONE@@", file=sys.__stderr__)
  sys.exit(0 if passed else 1)

if __name__ == "__main__":
  main()
