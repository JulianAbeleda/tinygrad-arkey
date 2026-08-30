#!/usr/bin/env python3
"""Bit-exact gate for the cooperative-Q4 in-CTA completion route.

Both arms consume the same Q4_K payload and the same packed Q8_1 buffer.  The
control materializes four warp partials and lets tinygrad sum axis 1; the
candidate performs that four-value merge in the producer CTA.  This isolates
the topology change from activation quantization and model-level argmax.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, subprocess
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.shared_q8_attention import _emit_q4_cooperative, _emit_q8_provider
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.route_class_numerics import _make_q4k_words

K = 4096


def _program(name, emitter):
  return KernelProgram("research.q4k_shared_q8_direct_exact", name,
                       KernelProgramProvenance.RESEARCH_ONLY, emitter)


def run_shape(rows: int, seed: int) -> dict:
  dev = Device.DEFAULT
  raw, _ = _make_q4k_words(rows, K, seed)
  x_np = np.random.default_rng(seed + 1).normal(0, 0.2, K).astype(np.float16)
  w = Tensor(raw, dtype=dtypes.uint32, device=dev).contiguous().realize()
  x = Tensor(x_np, dtype=dtypes.float16, device=dev).contiguous().realize()
  provider = _program("q8_provider", _emit_q8_provider())
  block_var = UOp.variable(f"q4_direct_exact_blocks_{rows}", 1, 4)
  bound_blocks = block_var.bind(4)

  @TinyJit
  def compare(ww, xx, blocks):
    xp = execute_research_program(Tensor.empty((K // 4 + K // 32,), dtype=dtypes.uint32, device=dev),
                                  xx, program=provider)
    xp = Tensor(xp.uop.after(blocks))
    extent = blocks.src[0]
    partial_program = _program("partial", _emit_q4_cooperative(rows, extent, direct_output=False))
    direct_program = _program("direct", _emit_q4_cooperative(rows, extent, direct_output=True))
    partial = execute_research_program(Tensor.empty((rows, 4), dtype=dtypes.float32, device=dev),
                                       ww, xp, program=partial_program)
    control = partial.sum(axis=1).contiguous()
    direct = execute_research_program(Tensor.empty((rows,), dtype=dtypes.float32, device=dev),
                                      ww, xp, program=direct_program)
    return control, direct

  # First call realizes kernels; second call exercises the captured dynamic path.
  compare(w, x, bound_blocks)
  control, direct = compare(w, x, bound_blocks)
  control.realize(), direct.realize()
  Device[dev].synchronize()
  ca, da = control.numpy(), direct.numpy()
  cu, du = ca.view(np.uint32), da.view(np.uint32)
  mismatches = int(np.count_nonzero(cu != du))
  return {
    "rows": rows,
    "elements": rows,
    "bitwise_equal": mismatches == 0,
    "mismatched_float32_words": mismatches,
    "max_abs_delta": float(np.max(np.abs(ca - da))),
    "control_sha256": hashlib.sha256(cu.tobytes()).hexdigest(),
    "direct_sha256": hashlib.sha256(du.tobytes()).hexdigest(),
    "q4_payload_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
    "x_payload_sha256": hashlib.sha256(x_np.tobytes()).hexdigest(),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"):
    raise RuntimeError(f"native NV required, got {dev}")
  shapes = [run_shape(4096, 20260824), run_shape(1024, 20260826)]
  result = {
    "schema": "tinygrad.q4k_shared_q8_direct_output_exact.v1",
    "device": str(dev),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "all_bitwise_equal": all(x["bitwise_equal"] for x in shapes),
    "shapes": shapes,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["all_bitwise_equal"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
