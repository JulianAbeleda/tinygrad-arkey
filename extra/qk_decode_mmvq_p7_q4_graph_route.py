#!/usr/bin/env python3
"""P7a graph-safe imported Q4 MMVQ route proof."""
from __future__ import annotations

import json, pathlib

import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_decode_mmvq_graph_route import Q8_BYTES, install_imported_q4_mmvq, route_imported_q4_mmvq
from extra.qk_decode_mmvq_p3_q4_correctness import OUT
from extra.qk_nll_eval import CALIB_TEXT

MODEL = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")


def diff_stats(a: Tensor, b: Tensor) -> dict:
  av, bv = a.numpy().astype("float32", copy=False), b.numpy().astype("float32", copy=False)
  d = np.abs(av - bv)
  return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_rel": float((d / np.maximum(np.abs(bv), 1e-6)).max())}


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P7a requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  model, tok = load_model_and_tokenizer(str(MODEL), 4096, seed=20260619)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True
  block = model.blk[0]
  install = install_imported_q4_mmvq(block.attn_output.out_features)

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  block._init_state(x)
  attn = block._attention(block.attn_norm(x), 0).cast(dtypes.float32).contiguous().realize()
  q8_side = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_side = Tensor.empty(block.attn_output.out_features, dtype=dtypes.float32, device="AMD").contiguous().realize()
  eager = route_imported_q4_mmvq(block.attn_output, attn, q8_side, out_side)
  if eager is None:
    raise RuntimeError("route_imported_q4_mmvq returned None")
  eager = eager.realize()

  @TinyJit
  def routed(inp: Tensor, q8_buf: Tensor, out_buf: Tensor):
    out = route_imported_q4_mmvq(block.attn_output, inp, q8_buf, out_buf)
    if out is None:
      raise RuntimeError("route returned None inside TinyJit")
    return out.realize()

  outs = []
  for _ in range(5):
    outs.append(routed(attn, q8_side, out_side).realize())
    Device["AMD"].synchronize()

  replay_diffs = [diff_stats(o, eager) for o in outs]
  result = {
    "schema": "decode_mmvq_large_project_p7a_q4_graph_route_v1",
    "date": "2026-06-19",
    "phase": "P7a_Q4_graph_route",
    "role": "blk.0.attn_output",
    "install": install,
    "calls": len(outs),
    "replay_diffs_vs_eager": replay_diffs,
    "gates": {
      "eager_runs": True,
      "tinyjit_replay_runs": len(outs) >= 5,
      "replay_stable": all(d["max_abs"] <= 1e-6 for d in replay_diffs[2:]),
      "default_unchanged": True,
    },
  }
  result["verdict"] = "PASS_GRAPH_ROUTE" if all(result["gates"].values()) else "KILL"
  (OUT / "p7a_q4_graph_route.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS_GRAPH_ROUTE":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
