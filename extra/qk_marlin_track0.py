#!/usr/bin/env python3
"""W1b' Track 0: diagnose WHY the W1 fused dequant->WMMA is 28x slow.

Build the same fused matmul as W1 (compressed Q4_K dequant -> cast f16 -> matmul, TC forced) on a
small shape, render the kernel, and inspect: (a) does WMMA appear, (b) does the Q4_K dequant ALU
(the bit-twiddling: rshift/and/the d*sc*q - dmin*mn expression) appear INSIDE the WMMA/reduce loop
body -- i.e. is it recomputed per tile -- vs hoisted to a one-time staging store. This confirms or
refutes the recompute hypothesis with the actual generated code, not speculation.
"""
from __future__ import annotations
import os
os.environ.setdefault("TC", "1")
os.environ.setdefault("TC_OPT", "2")
os.environ.setdefault("DEBUG", "6")  # render + print the program source

import pathlib, re, sys
from tinygrad import Tensor, dtypes
from extra.qk_layout import read_metadata, pick_tensor, tensor_shape, q4_k_reference, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS

MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf").expanduser()
ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/wmma-w1b")

def main():
  tensor = "blk.20.attn_q.weight"
  meta = read_metadata(MODEL); info = pick_tensor(meta.infos, tensor); rows, k = tensor_shape(info)
  # keep it small/fast: use the first 256 rows of the weight, batch 64
  rows = 256
  bs = meta.data_start + info.off
  q4 = rows*(k//Q4_K_BLOCK_ELEMS)*Q4_K_BLOCK_BYTES
  raw = Tensor(MODEL)[bs:bs+q4].to("AMD").realize()
  b = 64
  Tensor.manual_seed(1337)
  x = Tensor.randn(b, k, dtype=dtypes.float16, device="AMD").realize()

  # capture stderr (DEBUG=6 prints the rendered source there)
  import io, contextlib
  buf = io.StringIO()
  fused = (x @ q4_k_reference(raw, rows*k).reshape(rows, k).cast(dtypes.float16).transpose())
  with contextlib.redirect_stderr(buf):
    fused.realize()
  src = buf.getvalue()
  ART.mkdir(parents=True, exist_ok=True)
  (ART / "track0_w1_fused_source.txt").write_text(src)

  # crude structural analysis on the rendered text
  has_wmma = "wmma" in src.lower() or "WMMA" in src
  # dequant signatures: 4-bit unpack masks and the affine combine
  dequant_markers = len(re.findall(r"& 15|>> ?\d+|& 63|& 0xf", src))
  print("=" * 60, file=sys.__stdout__)
  print(f"rendered source bytes: {len(src)}", file=sys.__stdout__)
  print(f"contains WMMA intrinsic: {has_wmma}", file=sys.__stdout__)
  print(f"dequant bit-twiddle occurrences (& 15 / >> / & 63): {dequant_markers}", file=sys.__stdout__)
  print(f"saved -> {ART/'track0_w1_fused_source.txt'}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
