#!/usr/bin/env python3
"""Isolate the shared-Q8 activation quantization error on a real tail block.

The semantic gate showed the Q6 attention-V tail expansion overshoots the
relative-L2 gate no matter which consumer runs. This microgate measures the
single most direct hypothesis behind that: the shared-Q8 route quantizes the
fp16 attention activation to llama Q8_1 int8 (d=amax/127), while the ordinary
route feeds the fp16 activation straight into the GEMV.

It captures the real RMSNorm output for one tail attention block, quantizes it
exactly the way the shared provider does, then reports two numbers with the
real Q6 V weights:
  1. activation relative L2  = ||x - q8_1(x)|| / ||x||
  2. projection relative L2  = ||W @ x - W @ q8_1(x)|| / ||W @ x||

The second isolates the activation-only error in the exact projection whose
output feeds the residual stream and therefore the final-logit relative L2.
"""
from __future__ import annotations

import argparse, json
import numpy as np

from tinygrad import Tensor, UOp, dtypes

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _load():
  import tinygrad.llm.model as model_module
  model_module._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  from tinygrad.llm.model import Transformer
  model = Transformer.from_gguf(MODEL, 1024)[0]
  object.__setattr__(model.config, "prefill_custom_kernel_attn", False)
  object.__setattr__(model.config, "prefill_tc_attn", False)
  return model


def _capture_activations(model, block_indices: tuple[int, ...]) -> dict[int, np.ndarray]:
  """Return the real fp16 RMSNorm output feeding each requested attention block."""
  captured: dict[int, Tensor] = {}
  hooks = []
  for block_index in block_indices:
    block = model.blk[block_index]
    orig = block._attention
    def hook(x: Tensor, start_pos, ring_freqs=None, residual_for_output=None, _orig=orig, _i=block_index):
      captured[_i] = x
      return _orig(x, start_pos, ring_freqs, residual_for_output)
    block._attention = hook
    hooks.append((block, orig))
  try:
    from tinygrad.helpers import Context
    token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
    start_pos = UOp.variable("start_pos", 0, 1023)
    with Context(JIT=0):
      model.forward_with_logits(token, start_pos.bind(1), temp)[0].realize()
  finally:
    for block, orig in hooks:
      block._attention = orig
  out = {}
  for block_index in block_indices:
    if block_index not in captured:
      raise RuntimeError(f"block {block_index} attention was not traced")
    arr = captured[block_index].detach().numpy().reshape(-1).astype(np.float64)
    if not np.isfinite(arr).all():
      raise RuntimeError(f"block {block_index} activation is non-finite")
    out[block_index] = arr
  return out


def q8_1_quantize_dequantize_np(x: np.ndarray, block_elems: int = 32) -> np.ndarray:
  x = x.astype(np.float64)
  groups = x.reshape(-1, block_elems)
  amax = np.abs(groups).max(axis=1, keepdims=True)
  d = amax / 127.0
  q = np.clip(np.round(groups / d), -128, 127)
  return (q * d).reshape(-1)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--blocks", default="18,21,30,34")
  ap.add_argument("--out", default="/tmp/nv_q8_activation_quantization_error_microgate.json")
  args = ap.parse_args()
  block_indices = tuple(int(x) for x in args.blocks.split(","))

  model = _load()
  activations = _capture_activations(model, block_indices)

  from extra.llm_research.layout import q6_k_reference
  rows = {}
  for block_index in block_indices:
    x = activations[block_index]
    v = model.blk[block_index].attn_v
    halfs = v.q6k_storage.halfs.numpy().reshape(-1)
    raw = halfs.view(np.uint8).copy()
    krows, k = v.out_features, v.in_features
    W = q6_k_reference(Tensor(raw, dtype=dtypes.uint8), krows * k).numpy().astype(np.float64).reshape(krows, k)

    x_hat = q8_1_quantize_dequantize_np(x, 32)
    y_ref = W @ x
    y_q8 = W @ x_hat
    rows[str(block_index)] = {
      "activation": {
        "shape": list(x.shape), "amax": float(np.max(np.abs(x))),
        "relative_l2": float(np.linalg.norm(x - x_hat) / np.linalg.norm(x)),
        "max_abs": float(np.max(np.abs(x - x_hat))),
      },
      "projection": {
        "shape": list(y_ref.shape),
        "relative_l2": float(np.linalg.norm(y_ref - y_q8) / np.linalg.norm(y_ref)),
        "max_abs": float(np.max(np.abs(y_ref - y_q8))),
      },
    }

  payload = {
    "schema": "tinygrad.nv_q8_activation_quantization_error.v1",
    "model": MODEL, "blocks": rows,
    "interpretation": {
      "ordinary_path": "fp16 activation fed directly to the Q6 GEMV (no activation quantization)",
      "shared_q8_path": "fp16 activation quantized to llama Q8_1 int8, d=amax/127, per 32-element group",
      "note": "projection relative L2 is the activation-only error in the exact V projection output that feeds the residual stream",
    },
  }
  with open(args.out, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
