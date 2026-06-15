#!/usr/bin/env python3
"""Phase N0a: matmul_decoded -- the non-fused competitive path for the batched regime.

dequant pass (compressed Q4_K -> fp16) + tinygrad NATIVE matmul (BEAM/heuristic-tuned, 33-98% peak).
Compares vs the W2 fused split-K kernel (reads compressed, caps ~3-6% peak). Reports both the
per-call cost (dequant + matmul, the honest per-forward-pass price) and the amortized cost (matmul
only, if fp16 weights stay resident -- 2x memory, the price of dropping fusion).

Run: DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/qk_matmul_decoded.py
"""
from __future__ import annotations
import os
os.environ.setdefault("TC", "1")

import json, pathlib, sys
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import GlobalCounters
from extra.qk_layout import q4_k_reference
import extra.qk_marlin_w2 as w2

PEAK = 83.64
ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/native-matmul-N0")


def measure_decoded(M:int, K:int, N:int, tensor:str):
  # raw compressed weights + the fp16 reference weight (resident)
  meta = w2.read_metadata(w2.MODEL); info = w2.pick_tensor(meta.infos, tensor)
  rows, Kfull = w2.tensor_shape(info); k_blocks_full = Kfull // w2.Q4_K_BLOCK_ELEMS
  nb = K // w2.Q4_K_BLOCK_ELEMS; assert M <= rows and nb <= k_blocks_full
  bs = meta.data_start + info.off
  full = Tensor(w2.MODEL)[bs:bs + M*k_blocks_full*w2.Q4_K_BLOCK_BYTES].to("AMD").realize()
  raw = full.reshape(M, k_blocks_full, w2.Q4_K_BLOCK_BYTES)[:, :nb, :].flatten().contiguous().realize()
  wf16 = q4_k_reference(raw, M*K).reshape(M, K).cast(dtypes.float16).realize()
  Tensor.manual_seed(1337)
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ref = (wf16.cast(dtypes.float32) @ B.cast(dtypes.float32)).realize()

  dequant = lambda: q4_k_reference(raw, M*K).reshape(M, K).cast(dtypes.float16)   # the dequant pass
  matmul = lambda: wf16 @ B                                                        # native matmul
  # correctness of the native matmul path
  rel = (matmul().realize().cast(dtypes.float32) - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  t_deq, t_mm = w2._time(dequant), w2._time(matmul)
  flops = 2*M*K*N
  # fused split-K (reads compressed) for comparison
  fk = w2.measure_splitk(M, K, N, 16, min(2048, K), tensor, 1)
  return {
    "shape": {"M": M, "K": K, "N": N}, "tensor": tensor, "correct": rel < 1e-2,
    "dequant_us": round(t_deq*1e6, 2), "matmul_us": round(t_mm*1e6, 2),
    "matmul_tflops": round(flops/t_mm/1e12, 2), "matmul_pct_peak": round(flops/t_mm/1e12/PEAK*100, 1),
    "decoded_percall_us": round((t_deq+t_mm)*1e6, 2),
    "decoded_percall_tflops": round(flops/(t_deq+t_mm)/1e12, 2),
    "fused_splitk_us": fk["us"], "fused_splitk_tflops": fk["tflops"],
    "matmul_vs_fused": round(fk["us"]/(t_mm*1e6), 2),             # amortized native vs fused (x faster)
    "percall_vs_fused": round(fk["us"]/((t_deq+t_mm)*1e6), 2),    # per-call (incl dequant) vs fused
  }


def main():
  tensor = "blk.20.attn_q.weight"
  shapes = [(4096, 4096, 16), (4096, 4096, 64), (4096, 4096, 256), (4096, 4096, 512), (4096, 4096, 2048)]
  curve = [measure_decoded(M, K, N, tensor) for (M, K, N) in shapes]
  out = {"kind": "qk_matmul_decoded", "phase": "Phase N0a", "tensor": tensor, "peak_tflops": PEAK,
         "note": "matmul_decoded = dequant pass + NATIVE matmul. amortized = matmul only (fp16 "
                 "resident, 2x mem). per-call = dequant + matmul. fused = W2 split-K (reads compressed).",
         "curve": curve}
  ART.mkdir(parents=True, exist_ok=True)
  (ART / "n0a_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out, indent=2, sort_keys=True), file=sys.__stdout__)
  return 0 if all(c["correct"] for c in curve) else 1


if __name__ == "__main__":
  raise SystemExit(main())
