#!/usr/bin/env python3
"""Load a captured production NV cubin through the CUDA driver API (thin caller of BoltBeam).

The production NV backend submits QMDs directly through the NV ioctl path, so Nsight Compute reports no kernels.
Replaying the exact cubin with cuModuleLoad/cuLaunchKernel in an ordinary CUDA context makes it visible.  That replay
is measurement, and measurement is BoltBeam's: this file forwards every flag to boltbeam.collectors.cubin_launch
(same flags: --cubin --symbol --grid --block --shared-mem --n-bufs --buf-sizes --vals --val-groups --reps --warmup
--condition-mib --out; plus --capture/--kernel to read a tinygrad.nv_cubin_capture.v1 record directly).  tinygrad keeps
the producer half, nv_cubin_capture.py.  One behaviour difference: --condition-mib now streams the L2-evicting bytes
with cuMemsetD8 (a write) instead of an NVRTC-compiled read kernel, so BoltBeam needs no compiler.
"""
from __future__ import annotations

import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from extra.llm_research.boltbeam_checkout import require_boltbeam


def main(argv:list[str] | None=None) -> int:
  require_boltbeam("boltbeam.collectors.cubin_launch")
  from boltbeam.collectors.cubin_launch import main as boltbeam_main
  return boltbeam_main(argv)


if __name__ == "__main__":
  raise SystemExit(main())
