#!/usr/bin/env python3
"""P7c smoke for DECODE_MMVQ_IMPORT_Q4 one-role model route."""
from __future__ import annotations

import json, os, pathlib

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "1")
os.environ.setdefault("DECODE_MMVQ_IMPORT_Q4", "1")

from tinygrad import Device, Tensor, dtypes
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_decode_mmvq_p3_q4_correctness import OUT
from extra.qk_decode_mmvq_graph_route import install_imported_q4_mmvq
from extra.qk_nll_eval import CALIB_TEXT

MODEL = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P7c requires DEV=AMD, got {Device.DEFAULT!r}")
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
  out = block._attention(block.attn_norm(x), 0).realize()
  Device["AMD"].synchronize(timeout=10000)
  routed_blocks = [0] if hasattr(block, "_decode_mmvq_import_q4_q8") else []
  result = {
    "schema": "decode_mmvq_large_project_p7c_one_role_smoke_v1",
    "date": "2026-06-19",
    "phase": "P7c_one_role_model_route_smoke",
    "flag": "DECODE_MMVQ_IMPORT_Q4=1",
    "role": "blk.0.attn_output",
    "install": install,
    "output_shape": list(out.shape),
    "routed_blocks": routed_blocks,
    "routed_block_count": len(routed_blocks),
    "verdict": "PASS_ONE_ROLE_ROUTE_SMOKE" if len(routed_blocks) > 0 else "NO_ROUTE",
  }
  (OUT / "p7c_one_role_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS_ONE_ROLE_ROUTE_SMOKE":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
