#!/usr/bin/env python3
"""P4 standalone performance for imported llama Q4_K MMVQ through HCQ."""
from __future__ import annotations

import json, time

import numpy as np
from tinygrad import Device, Tensor, dtypes

from extra.q8_ffn_handwritten_oracle import q8_blocks
from extra.qk_decode_mmvq_p3_q4_correctness import OBJ, OUT, RawKernargAMDProgram, kd_offset, q4_tensor_bytes


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P4 requires DEV=AMD, got {Device.DEFAULT!r}")
  cap = json.loads((OUT / "p2_kernarg_capture.json").read_text())["selected"]["q4_attn_q_or_o"]
  raw = bytearray(cap["kernarg_bytes"])
  q4, rows, k = q4_tensor_bytes("blk.0.attn_output.weight")
  Tensor.manual_seed(4)
  x = Tensor.randn(k, dtype=dtypes.float32).numpy().astype(np.float32)
  q8 = q8_blocks(x)
  q4_t = Tensor(np.frombuffer(q4, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_t = Tensor(np.frombuffer(q8, dtype=np.uint8).copy(), dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_t = Tensor.zeros(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()
  dev = Device["AMD"]
  dev.synchronize()
  va = lambda t: t.uop.buffer._buf.va_addr
  import struct
  struct.pack_into("<Q", raw, 0, va(q4_t))
  struct.pack_into("<Q", raw, 8, va(q8_t))
  struct.pack_into("<Q", raw, 16, 0)
  struct.pack_into("<Q", raw, 56, va(out_t))
  elf = OBJ.read_bytes()
  prg = RawKernargAMDProgram(dev, "llama_mmvq_q4_p4", elf, kd_offset(elf, cap["kernel_symbol"]), bytes(raw))
  # Warm once, then enqueue many launches into one HCQ submit. This avoids per-launch submit overhead and timestamps
  # the whole batch on the device queue.
  prg(q4_t.uop.buffer._buf, q8_t.uop.buffer._buf, out_t.uop.buffer._buf,
      global_size=tuple(cap["num_workgroups"]), local_size=tuple(cap["local"]), wait=True, timeout=10000)
  iters = 200
  args = prg.fill_kernargs((q4_t.uop.buffer._buf, q8_t.uop.buffer._buf, out_t.uop.buffer._buf))
  sig_st, sig_en = dev.new_signal(), dev.new_signal()
  st = time.perf_counter()
  q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier().timestamp(sig_st)
  for _ in range(iters):
    q.exec(prg, args, tuple(cap["num_workgroups"]), tuple(cap["local"]))
  q.timestamp(sig_en).signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize(timeout=10000)
  wall = time.perf_counter() - st
  wall_ms = wall * 1000.0 / iters
  dev_ms = float(sig_en.timestamp - sig_st.timestamp) / 1000.0 / iters
  ms = dev_ms if dev_ms > 0 else wall_ms
  q4_gbs = len(q4) / (ms * 1e-3) / 1e9
  result = {
    "schema": "decode_mmvq_large_project_p4_q4_perf_v1",
    "date": "2026-06-19",
    "phase": "P4_Q4_standalone_perf",
    "rows": rows,
    "k": k,
    "iters": iters,
    "wall_ms_per_launch": wall_ms,
    "device_ms_per_launch": dev_ms,
    "q4_bytes": len(q4),
    "effective_q4_gbs": q4_gbs,
    "hbm_peak_gbs": 960,
    "pct_hbm_peak": q4_gbs / 960 * 100,
    "gate_pct_hbm_peak": 60,
    "timing_note": "single HCQ submit containing many exec packets; device timestamp used for gate, wall retained for sanity",
    "verdict": "PASS" if q4_gbs / 960 >= 0.60 else "BELOW_GATE",
  }
  (OUT / "p4_q4_perf.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
