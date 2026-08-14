#!/usr/bin/env python3
"""Hermetic DEV=CPU microgate for the P1 vocab-head aux scatter-chain fusion.

The four aux kernels (E_1187_32_4, r_32_4_1187, r_128_16_8_1187, r_16_8) reduce the
151936-row logits to one token id.  The fused P1 route carries a packed u64 (max, index)
key per GEMV warp tile in the vocab_top1 epilogue, warp-reduces those keys in-kernel, and
finishes with one tiny u64 MAX over the per-tile keys.  This gate runs on DEV=CPU only
(the coop warp-shuffle epilogue is not CPU-numeric): it compares the fused top-1 token id
against the legacy Tensor.argmax chain over normal, all-zero, and tie-containing logits,
and asserts identical token ids (first index wins), bit-exact.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.llm.decode_kernels import emit_q6k_vocab_top1_reduce_kernel, q6k_spec_for_role
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.packed_argmax import packed_argmax_tile_keys_fp32, packed_argmax_tiles_fp32

VOCAB_ROWS, VOCAB_K = 151936, 4096


def _tie_logits() -> np.ndarray:
  """Max value tied across head, middle, tail, plus signed-zero ties."""
  a = np.full((1, VOCAB_ROWS), -100.0, dtype=np.float32)
  a[0, 0], a[0, 1000], a[0, VOCAB_ROWS - 1] = 3.0, 3.0, 3.0
  a[0, 5], a[0, 6] = -0.0, 0.0
  return a


def _datasets() -> dict[str, np.ndarray]:
  rng = np.random.default_rng(20260812)
  return {
    "normal": rng.standard_normal((1, VOCAB_ROWS)).astype(np.float32),
    "allzero": np.zeros((1, VOCAB_ROWS), dtype=np.float32),
    "ties": _tie_logits(),
  }


def run() -> dict:
  dev = Device.DEFAULT
  if str(dev) != "CPU": raise RuntimeError(f"DEV=CPU required, got {dev}")
  spec = q6k_spec_for_role(VOCAB_ROWS, VOCAB_K, row_tile=2, reduction="in_kernel", epilogue="vocab_top1")
  program = KernelProgram("research.vocab_top1", spec.kernel_name,
                          KernelProgramProvenance.RESEARCH_ONLY, emit_q6k_vocab_top1_reduce_kernel(spec))
  cases = {}
  for name, a in _datasets().items():
    x = Tensor(a)
    legacy = int(x.argmax(-1, keepdim=True).numpy().ravel()[0])
    fused = int(packed_argmax_tiles_fp32(x, 2, keepdim=True).numpy().ravel()[0])
    # Second arm: the emitted final packed-reduce kernel consumes one u64 per warp tile
    # exactly as the vocab_top1 GEMV epilogue would carry it.
    tile_keys = packed_argmax_tile_keys_fp32(x, 2).numpy().ravel()
    out = Tensor.empty((1,), dtype=dtypes.int32)
    got = int(execute_research_program(out, Tensor(tile_keys), program=program).numpy().ravel()[0])
    cases[name] = {"legacy_token": legacy, "fused_token": fused, "reduce_kernel_token": got,
                   "fused_identical": legacy == fused, "reduce_kernel_identical": legacy == got,
                   "input_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
  exact = all(c["fused_identical"] and c["reduce_kernel_identical"] for c in cases.values())
  return {"schema": "tinygrad.nv.vocab_top1_fusion_cpu_microgate.v1", "device": str(dev),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": {"logits": [1, VOCAB_ROWS], "k": VOCAB_K, "row_tile": 2, "tiles": VOCAB_ROWS // 2},
    "contract": {"aux_kernels_fused": 4, "tie_semantics": "first index wins",
      "fused": "per-tile packed u64 (max,index) + warp MAX + one cross-tile MAX",
      "legacy": "Tensor.argmax (r_16_8 chain)"},
    "cases": cases, "exact": exact,
    "gate": "PASS" if exact else "FAIL"}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out")
  a = p.parse_args()
  result = run()
  text = json.dumps(result, indent=2, sort_keys=True)
  if a.out: pathlib.Path(a.out).write_text(text + "\n")
  print(text)
  return 0 if result["gate"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
