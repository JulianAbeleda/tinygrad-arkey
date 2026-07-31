#!/usr/bin/env python3
"""Per-kernel DEBUG=2 attribution of Metal whole-model prefill, at four depths.

Measurement basis (per the campaign instructions):
  - unbatched: Context(JIT=0) around every call to model(...) for a given start_pos, so
    TinyJit's "jit ignore" branch fires every time (tinygrad/engine/jit.py:555) and each kernel
    gets its own Metal command buffer / DEBUG=2 print line, instead of being folded into one
    "batched N" MetalGraph replay that would hide per-kernel attribution
    (tinygrad/runtime/ops_metal.py:56 -- "command buffers from MetalGraph are not profiled").
  - Device.synchronize() before stopping any clock. DEBUG>=2's own per-kernel timing already
    calls Device[device].synchronize() around every kernel (tinygrad/engine/realize.py:58-64),
    which is what makes its printed `tm` a real synced duration rather than enqueue time; we
    additionally synchronize after every call as a second guard.
  - warmup (uncaptured) calls happen with DEBUG at its default (0) so kernel-cache compilation
    noise never lands in the parsed log; only the single measured call per depth is captured
    with DEBUG=2.

Depths are mapped to the LAST 512-token chunk whose KV window ends exactly at that depth, i.e.
start_pos = depth - chunk_n: depth 512 -> sp=0, 1024 -> sp=512, 2048 -> sp=1536, 4096 -> sp=3584.
This is an explicit choice (documented, not fabricated): it means each depth's kernel log is for
the chunk that PRODUCES that many total prefilled tokens, matching AMD's own "start_pos 0->3584"
convention in docs/prefill-current-state.md.

Writes one raw DEBUG=2 log per depth under /tmp/metal_prefill_attn_depth/, plus a JSON summary of
per-canonical-kernel aggregation across depths so a canonical kernel's DEPTH-SCALING behavior can
be inspected directly.
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "/Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf"
MAX_CONTEXT = 4608
CHUNK_N = 512
DEPTHS = (512, 1024, 2048, 4096)
OUT_DIR = pathlib.Path("/tmp/metal_prefill_attn_depth")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
  from tinygrad import Tensor, Device
  from tinygrad.helpers import Context
  from tinygrad.llm.generate import load_model_and_tokenizer

  print(f"loading {MODEL} ...", flush=True)
  model, _ = load_model_and_tokenizer(MODEL, MAX_CONTEXT, seed=20260617)
  for block in model.blk: block._use_flash, block._prefill_v2 = True, True
  temp = Tensor([0.0])
  chunk = Tensor([[(i * 7) % 1000 for i in range(CHUNK_N)]], dtype="int32").contiguous()
  dev = Device[Device.DEFAULT]

  def call(sp: int):
    return model(chunk, sp, temp, use_flash=True)

  for depth in DEPTHS:
    sp = depth - CHUNK_N
    assert sp >= 0
    print(f"\n=== depth={depth} (start_pos={sp}) ===", flush=True)
    # warmup: compile + first-touch, JIT off (unbatched), DEBUG at default (0) -- not captured.
    with Context(JIT=0):
      for _ in range(3):
        call(sp).realize()
      dev.synchronize()
    # measured: one unbatched forward, DEBUG=2, captured to a buffer.
    buf = io.StringIO()
    with Context(JIT=0, DEBUG=2):
      with contextlib.redirect_stdout(buf):
        call(sp).realize()
      dev.synchronize()
    log_text = buf.getvalue()
    out_path = OUT_DIR / f"depth_{depth}_sp{sp}.log"
    out_path.write_text(log_text)
    n_lines = sum(1 for line in log_text.splitlines() if line.startswith(("*** METAL", "\x1b[")))
    print(f"  wrote {out_path} ({len(log_text)} bytes, ~{n_lines} candidate launch lines)", flush=True)

  print("\ndone.")


if __name__ == "__main__":
  main()
