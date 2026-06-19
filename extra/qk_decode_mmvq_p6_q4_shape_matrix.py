#!/usr/bin/env python3
"""P6 Q4_K shape matrix for the imported llama MMVQ consumer."""
from __future__ import annotations

import json, struct, time

import numpy as np

from tinygrad import Device, Tensor, dtypes
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.qk_decode_mmvq_p3_q4_correctness import OBJ, OUT, RawKernargAMDProgram, kd_offset, q4_tensor_bytes

TENSORS = ("blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight")


def run_tensor(name: str, cap: dict, elf: bytes) -> dict:
  raw = bytearray(cap["kernarg_bytes"])
  q4, rows, k = q4_tensor_bytes(name)
  rng = np.random.default_rng(abs(hash(name)) & 0xffffffff)
  x = rng.standard_normal(k).astype(np.float32)
  q8 = q8_blocks(x)
  vals = []
  for off in range(0, len(q8), 36):
    d = np.frombuffer(q8[off:off + 2], dtype=np.float16).astype(np.float32)[0]
    vals.append(np.frombuffer(q8[off + 4:off + 36], dtype=np.int8).astype(np.float32) * d)
  ref = q4_ref_rows(q4, rows, k, np.concatenate(vals).astype(np.float32))

  q4_t = Tensor(np.frombuffer(q4, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_t = Tensor(np.frombuffer(q8, dtype=np.uint8).copy(), dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_t = Tensor.zeros(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()
  dev = Device["AMD"]
  dev.synchronize()
  va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 0, va(q4_t))
  struct.pack_into("<Q", raw, 8, va(q8_t))
  struct.pack_into("<Q", raw, 16, 0)
  struct.pack_into("<Q", raw, 56, va(out_t))
  prg = RawKernargAMDProgram(dev, f"llama_mmvq_q4_p6_{name.replace('.', '_')}", elf, kd_offset(elf, cap["kernel_symbol"]), bytes(raw))
  launch = (rows, 1, 1)
  local = tuple(cap["local"])
  prg(q4_t.uop.buffer._buf, q8_t.uop.buffer._buf, out_t.uop.buffer._buf, global_size=launch, local_size=local, wait=True, timeout=10000)

  iters = 100
  args = prg.fill_kernargs((q4_t.uop.buffer._buf, q8_t.uop.buffer._buf, out_t.uop.buffer._buf))
  sig_st, sig_en = dev.new_signal(), dev.new_signal()
  st = time.perf_counter()
  q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier().timestamp(sig_st)
  for _ in range(iters):
    q.exec(prg, args, launch, local)
  q.timestamp(sig_en).signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize(timeout=10000)
  wall_ms = (time.perf_counter() - st) * 1000.0 / iters
  dev_ms = float(sig_en.timestamp - sig_st.timestamp) / 1000.0 / iters
  got = out_t.numpy()
  diff = np.abs(got - ref)
  ms = dev_ms if dev_ms > 0 else wall_ms
  return {
    "tensor": name,
    "rows": rows,
    "k": k,
    "q4_bytes": len(q4),
    "launch": {"num_workgroups": list(launch), "local": list(local)},
    "wall_ms_per_launch": wall_ms,
    "device_ms_per_launch": dev_ms,
    "q4_gbs": len(q4) / (ms * 1e-3) / 1e9,
    "pct_hbm": len(q4) / (ms * 1e-3) / 1e9 / 960.0 * 100.0,
    "max_abs": float(diff.max()),
    "mean_abs": float(diff.mean()),
    "correct": bool(diff.max() < 2e-2),
  }


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P6 requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  cap = json.loads((OUT / "p2_kernarg_capture.json").read_text())["selected"]["q4_attn_q_or_o"]
  elf = OBJ.read_bytes()
  rows = [run_tensor(name, cap, elf) for name in TENSORS]
  result = {
    "schema": "decode_mmvq_large_project_p6_q4_shape_matrix_v1",
    "date": "2026-06-19",
    "phase": "P6_Q4_shape_matrix",
    "rows": rows,
    "summary": {
      "all_correct": all(r["correct"] for r in rows),
      "min_pct_hbm": min(r["pct_hbm"] for r in rows),
      "median_pct_hbm": float(np.median([r["pct_hbm"] for r in rows])),
    },
  }
  result["verdict"] = "PASS_Q4_MATRIX" if result["summary"]["all_correct"] and result["summary"]["min_pct_hbm"] >= 45 else "KILL"
  (OUT / "p6_q4_shape_matrix.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
