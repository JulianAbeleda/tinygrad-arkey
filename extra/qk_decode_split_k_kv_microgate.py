#!/usr/bin/env python3
"""L2 microgate: the generated split-K G3 Q4_K GEMV is numerically identical to the direct G3 GEMV.

Generic correctness proof for the split-K decode capability, on the real occupancy-starved KV shape
(out=1024, in=5120) from Qwen3-14B. Direct G3 launches `out` workgroups; split-K launches `out*parts` and
finalizes with a sum over parts. Same math -> the results must match to fp tolerance.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_split_k_kv_microgate.py
Verdict: L2_MICROGATE_PASS_SPLITK_MATCHES_DIRECT / L2_MICROGATE_FAIL
"""
from __future__ import annotations
import os, sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
  from tinygrad import Tensor
  from tinygrad.dtype import dtypes
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel, q4k_g3_lanemap_gemv_splitk_kernel

  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
  m, _tok = load_model_and_tokenizer(model, 4608, seed=20260701)
  reg = getattr(m, "_q4k_linears", None)
  lins = reg.linears if reg else []
  # pick the occupancy-starved KV shape: out=1024, in=5120 (attn_k). generic: any small-out Q4_K linear.
  target = next((l for l in lins if l.out_features == 1024 and l.in_features == 5120), None)
  if target is None:
    print("L2_MICROGATE_FAIL: no out=1024,in=5120 Q4_K linear found"); print([(l.out_features, l.in_features) for l in lins][:8]); sys.exit(2)

  out_f, in_f = target.out_features, target.in_features
  words = target.q4k_storage.words.to("AMD").contiguous() if target.q4k_storage.mode == "q4_ondemand" else target.q4k_storage.words.to("AMD")
  Tensor.manual_seed(7)
  xv = Tensor.randn(in_f).cast(dtypes.float16).contiguous()

  direct = Tensor.empty(out_f, dtype=dtypes.float32).custom_kernel(
    words, xv, fxn=q4k_g3_lanemap_gemv_kernel(out_f, in_f))[0].numpy()

  results = {}
  for parts in (5,):   # blocks_per_group=5 for k=5120 -> parts in {1,5}; 5 = 5x workgroups (1024->5120)
    partials = Tensor.empty(out_f * parts, dtype=dtypes.float32).custom_kernel(
      words, xv, fxn=q4k_g3_lanemap_gemv_splitk_kernel(out_f, in_f, parts))[0]
    splitk = partials.reshape(out_f, parts).sum(axis=1).numpy()
    import numpy as np
    denom = np.abs(direct).mean() + 1e-9
    rel_rmse = float(np.sqrt(((splitk - direct) ** 2).mean()) / denom)
    max_abs = float(np.abs(splitk - direct).max())
    results[parts] = {"rel_rmse": rel_rmse, "max_abs": max_abs}
    print(f"parts={parts}: rel_rmse={rel_rmse:.2e} max_abs={max_abs:.2e}")

  ok = all(v["rel_rmse"] < 1e-3 for v in results.values())
  verdict = "L2_MICROGATE_PASS_SPLITK_MATCHES_DIRECT" if ok else "L2_MICROGATE_FAIL"
  OUT = ROOT / "bench/qwen-14b-32b-l2-split-k-kv"; OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "microgate.json").write_text(json.dumps(
    {"verdict": verdict, "shape": [out_f, in_f], "results": results}, indent=2))
  print(verdict)
  sys.exit(0 if ok else 2)


if __name__ == "__main__":
  main()
