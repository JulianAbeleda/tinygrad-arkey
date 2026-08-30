#!/usr/bin/env python3
"""Minimal flush-kernel write verification.

Launches the grid-stride flush once and reads back spread words to prove the
full 128 MiB was written (the original one-element-per-thread flush only wrote
the first block because its 131072-block grid did not launch).

Measurement tooling only; no production code path is touched.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import (  # noqa: E402
  _alloc, _make_queue, _compile_flush, FLUSH_MIB, FLUSH_BLOCK, FLUSH_GRID, FLUSH_XOR,
)


def main() -> int:
  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram

  dev = Device["NV"]
  prg = NVProgram(dev, f"nv_r_flush_{FLUSH_MIB}mib", _compile_flush(dev))
  buf = _alloc(dev, FLUSH_MIB * 1024 * 1024)
  dev.allocator._copyin(buf, memoryview(bytearray(buf.size)))

  q = _make_queue(dev)
  args = prg.fill_kernargs((buf,))
  q.exec(prg, args, FLUSH_GRID, (FLUSH_BLOCK, 1, 1))
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()

  floats = buf.size // 4
  idx = sorted(set([0, 1, 255, 256, 1023, 65536, floats // 4, floats // 2, floats - 1024, floats - 1]))
  ok = True
  for j in idx:
    mv = memoryview(bytearray(4))
    dev.allocator._copyout(mv, buf.offset(j * 4, 4))
    got = struct.unpack("<I", mv)[0]
    expect = j ^ FLUSH_XOR
    match = got == expect
    ok &= match
    print(f"idx={j:9d} expect={expect:#010x} got={got:#010x} match={match}")
  print("FLUSH_OK" if ok else "FLUSH_FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
