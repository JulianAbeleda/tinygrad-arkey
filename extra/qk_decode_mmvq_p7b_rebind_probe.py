#!/usr/bin/env python3
"""P7b raw-kernarg rebind proof for imported llama Q4 MMVQ.

This probe intentionally does not launch the imported kernel. P7b-1's gate is
that fill_kernargs writes the raw template and records pointer patches through
bind_data; correctness/perf launches are later phases.
"""
from __future__ import annotations

import json, pathlib, struct

from tinygrad import Device, Tensor, dtypes
from extra.q8_ffn_oneblock_route import realized_buf
from extra.qk_decode_mmvq_graph_route import ImportedQ4MMVQRunner, Q8_BYTES
from extra.qk_decode_mmvq_p3_q4_correctness import OBJ, OUT, RawKernargAMDProgram, kd_offset


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P7b requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  cap = json.loads((OUT / "p2_kernarg_capture.json").read_text())["selected"]["q4_attn_q_or_o"]
  raw = bytes(cap["kernarg_bytes"])
  dev = Device["AMD"]
  prg = RawKernargAMDProgram(dev, "llama_q4_mmvq_p7b_rebind_proof", OBJ.read_bytes(), kd_offset(OBJ.read_bytes(), cap["kernel_symbol"]), raw)
  runner = ImportedQ4MMVQRunner(prg, raw, (4096, 1, 1), tuple(cap["local"]))
  q4 = Tensor.empty(4096 * (4096 // 256) * 144 // 4, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out = Tensor.empty(4096, dtype=dtypes.float32, device="AMD").contiguous().realize()
  argsbuf = dev.kernargs_buf.offset(offset=dev.kernargs_offset_allocator.alloc(prg.kernargs_alloc_size, 8), size=prg.kernargs_alloc_size)
  outb, q4b, q8b = realized_buf(out), realized_buf(q4), realized_buf(q8)
  st = runner.fill_kernargs((outb, q4b, q8b), kernargs=argsbuf)
  q = dev.hw_compute_queue_t()
  q.bind_args_state(st)
  copied = bytes(argsbuf.cpu_view().view(size=len(raw), fmt="B"))
  patched = {
    "q4": struct.unpack_from("<Q", copied, 0)[0],
    "q8": struct.unpack_from("<Q", copied, 8)[0],
    "out": struct.unpack_from("<Q", copied, 56)[0],
  }
  bind_vals = [tuple(int(v) if isinstance(v, int) else str(v) for v in vals) for vals, _mem, fmt in st.bind_data if fmt == "Q"]
  masked_copied, masked_raw = bytearray(copied), bytearray(raw)
  for off in (0, 8, 56):
    masked_copied[off:off + 8] = b"\x00" * 8
    masked_raw[off:off + 8] = b"\x00" * 8
  result = {
    "schema": "decode_mmvq_large_project_p7b_rebind_probe_v1",
    "date": "2026-06-19",
    "phase": "P7b_1_rebindable_args_state_cpu_proof",
    "patch_offsets": {"q4": 0, "q8": 8, "out": 56},
    "raw_template_copied_except_patch_offsets": bytes(masked_copied) == bytes(masked_raw),
    "bind_records": len(st.bind_data),
    "bind_values": bind_vals,
    "patched_values_after_bind_args_state": patched,
    "expected_values": {
      "q4": q4b.va_addr,
      "q8": q8b.va_addr,
      "out": outb.va_addr,
    },
    "verdict": "PASS_REBIND_CPU_PROOF",
    "note": "No kernel launch. This validates that imported runner fill_kernargs uses bind_data pointer patches instead of raw struct.pack.",
  }
  if result["bind_records"] != 3 or not result["raw_template_copied_except_patch_offsets"] or patched != result["expected_values"]:
    result["verdict"] = "FAIL"
  (OUT / "p7b_rebind_probe.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS_REBIND_CPU_PROOF":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
