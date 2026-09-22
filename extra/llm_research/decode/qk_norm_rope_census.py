#!/usr/bin/env python3
"""Count Q/K norm/rope/fused kernels in the decode JIT graph (control vs candidate).

Loads the model, optionally installs the harness-only fused norm+rope hook,
runs enough decode tokens to force JIT capture, then walks every captured
TinyJit LINEAR to count the Q/K norm, rope, and fused kernel programs.  This
proves the 144 -> 72 kernel swap the fused admission claims, independent of
the wall timing.  No production file is modified.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

from tinygrad.uop.ops import KernelInfo, Ops


def _count_decoders(model) -> dict[str, int]:
  from tinygrad.engine.jit import TinyJit
  counts: dict[str, int] = {}
  for attr in dir(model):
    obj = getattr(model, attr)
    if isinstance(obj, TinyJit) and obj.captured is not None:
      linear = obj.captured.linear
      for src in linear.src:
        name = None
        if src.op is Ops.SINK and isinstance(getattr(src, "arg", None), KernelInfo):
          name = src.arg.name
        elif src.op is Ops.CALL and src.src and isinstance(getattr(src.src[0], "arg", None), KernelInfo):
          name = src.src[0].arg.name
        if name:
          counts[name] = counts.get(name, 0) + 1
  return counts


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--enabled", action="store_true")
  ap.add_argument("--depth", type=int, default=8)
  ap.add_argument("--out-json", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _set_admission

  model = _load(MODEL, 1024)
  _set_admission(model, args.enabled)
  gen = model.generate(_prompt(MODEL, args.depth), chunk_size=32, temperature=0.0)
  try:
    for _ in range(4):
      next(gen)
  finally:
    gen.close()

  counts = _count_decoders(model)
  def pick(*needles):
    return {k: v for k, v in counts.items() if any(n in k for n in needles)}
  norm = sum(v for k, v in counts.items() if k.startswith("reduce_output_rmsnorm_"))
  fused = sum(v for k, v in counts.items() if k.startswith("reduce_output_rmsnorm_rope_"))
  rope = sum(v for k, v in counts.items() if "rope" in k.lower())
  result = {
    "schema": "tinygrad.qk_norm_rope_census.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "enabled": args.enabled,
    "relevant": {
      "norm": pick("reduce_output_rmsnorm_"),
      "rope_like": pick("rope"),
    },
    "totals": {"reduce_output_norm": norm, "fused_norm_rope": fused, "rope_like": rope},
    "all": counts,
  }
  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result["totals"], indent=2))
  print(json.dumps(result["relevant"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
