#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_store_b32, s_endpgm, s_load_b64, s_waitcnt, v_lshlrev_b32_e32, v_mov_b32_e32,
)

def build_local_id_probe(out:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((1, 1, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((128, 1, 1))]
  insts = [
    s_load_b64(sdata=s[4:5], sbase=s[0:1], offset=0, soffset=NULL),
    s_waitcnt(simm16=0),
    # With a 1D local shape, v0 is the flattened tid. The current assemble_linear descriptor does not safely expose v1.
    v_mov_b32_e32(vdst=v[2], src0=v[0]),
    v_mov_b32_e32(vdst=v[4], src0=v[0]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    global_store_b32(addr=v[2], data=v[4], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(out.base, *gidxs, *lidxs, arg=KernelInfo(name="q8_b2b_local_id_probe"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b local id layout probe for AMD DSL kernels")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_local_id_probe.json"))
  args = ap.parse_args()
  out = Tensor.zeros(128, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  y = Tensor.custom_kernel(out, fxn=build_local_id_probe)[0]
  run_linear(y.schedule_linear())
  got = y.numpy().astype(np.uint32)
  expected_flat = np.arange(128, dtype=np.uint32)
  result = {
    "date": "2026-06-19",
    "phase": "B2b_local_id_probe",
    "launch": {"global": [1,1,1], "local": [128,1,1]},
    "first_40": [int(x) for x in got[:40]],
    "nonzero_count": int(np.count_nonzero(got)),
    "matches_flat_tid": bool(np.array_equal(got, expected_flat)),
    "verdict": "PASS" if np.array_equal(got, expected_flat) else "FAIL",
    "next": "If PASS, use local=(128,1,1) and tid=v0 in full-row partials.",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
