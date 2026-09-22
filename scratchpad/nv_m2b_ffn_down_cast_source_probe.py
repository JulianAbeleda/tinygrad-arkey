#!/usr/bin/env python3
"""NV source probe for the M2b contract: under the candidate arm
(_ffn_down_resadd_lease on model + blocks + ffn_down linears) the ffn_down
Q4K/Q6K GEMVs must render their own *_epi_ffnresadd variants (adding the
hidden-state residual h in-kernel, fp32 store) and the standalone
E_32_32_4_02a9738c fp32 h+ffn_out add must be gone.  Prints every
*_epi_ffnresadd source plus a census of the E_32_32_4 residual family."""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
from extra.llm_research.decode.nv_epilogue_absorption_ab import _model
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _prompt

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def main() -> int:
  import tinygrad.renderer.cstyle as cstyle
  orig = cstyle.CStyleLanguage.render_kernel

  counts = {"E_32_32_4_02a9738c": 0, "E_32_32_4_fab82d40": 0, "E_32_32_4_0a5eb0ac": 0,
            "E_32_32_4_86a23e1a": 0, "E_32_32_4_other": 0}
  epi_names = []

  def hooked(self, function_name, kernel, bufs, uops, prefix=None):
    src = orig(self, function_name, kernel, bufs, uops, prefix)
    for prefix, key in (("E_32_32_4_02a9738c", "E_32_32_4_02a9738c"),
                        ("E_32_32_4_fab82d40", "E_32_32_4_fab82d40"),
                        ("E_32_32_4_0a5eb0ac", "E_32_32_4_0a5eb0ac"),
                        ("E_32_32_4_86a23e1a", "E_32_32_4_86a23e1a")):
      if function_name.startswith(prefix):
        counts[key] += 1
        break
    else:
      if function_name.startswith("E_32_32_4_"): counts["E_32_32_4_other"] += 1
    if function_name.endswith("_epi_ffnresadd") or function_name.startswith("q4k_g3_lanemap_gemv_epi_ffnresadd"):
      epi_names.append(function_name)
      print(f"=== SOURCE {function_name} ===\n{src}\n", flush=True)
    return src

  cstyle.CStyleLanguage.render_kernel = hooked
  with Context(DEBUG=0, CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1,
               CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    model, _ = _model("candidate", MODEL, 32768)
    gen = model.generate(_prompt(MODEL, 64), chunk_size=32, temperature=0.0)
    try:
      int(next(gen))
      for _ in range(3): int(next(gen))
    finally:
      gen.close()
  print(f"CENSUS {counts}", flush=True)
  print(f"EPI_FFNRESADD_NAMES {epi_names}", flush=True)
  # The render hook fires once per unique program name; the decode census counts
  # per-token EXECUTIONS (36 = 18 q4k + 18 q6k), which the AB census gate checks.
  assert len(epi_names) >= 2, f"expected at least the q4k + q6k *_epi_ffnresadd variants, got {epi_names}"
  assert any(name.startswith("q4k_g3_lanemap_gemv_epi_ffnresadd") for name in epi_names), f"missing q4k variant: {epi_names}"
  assert any(name.endswith("_epi_ffnresadd") and "q6k" in name for name in epi_names), f"missing q6k variant: {epi_names}"
  assert counts["E_32_32_4_02a9738c"] == 0, f"expected no E_32_32_4_02a9738c residual adds, got {counts['E_32_32_4_02a9738c']}"
  print("M2B SOURCE PROBE PASS", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
