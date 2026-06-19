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
from tinygrad.runtime.autogen.amd.rdna3.ins import global_store_b32, s_endpgm, s_load_b64, s_waitcnt, v_dot4_i32_iu8, v_mov_b32_e32

def build_dot4_store_program(out:UOp) -> UOp:
  insts = [
    s_load_b64(sdata=s[4:5], sbase=s[0:1], offset=0, soffset=NULL),
    s_waitcnt(simm16=0),
    v_dot4_i32_iu8(vdst=v[4], src0=0x01010101, src1=0x01010101, src2=0),
    v_mov_b32_e32(v[2], 0),
    v_mov_b32_e32(v[3], 0),
    global_store_b32(addr=v[2], data=v[4], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(out.base, arg=KernelInfo(name="q8_b2b_dot4_store_smoke"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b smoke: HCQ launch of tinygrad AMD DSL dot4 kernel")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_dot4_smoke.json"))
  args = ap.parse_args()

  out = Tensor.empty(1, dtype=dtypes.uint32, device="AMD").contiguous()
  y = Tensor.custom_kernel(out, fxn=build_dot4_store_program)[0]
  linear = y.schedule_linear()
  run_linear(linear)
  got = y.numpy().astype(np.uint32)
  result = {
    "date": "2026-06-19",
    "phase": "B2b0_asm_dot4_smoke",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_no_C_no_hipcc",
    "got": int(got[0]),
    "expected": 4,
    "correct": int(got[0]) == 4,
    "verdict": "PASS" if int(got[0]) == 4 else "FAIL",
    "next": "If PASS, proceed to a standalone AMD DSL fused gate/up consumer skeleton with real q4/q8 buffers.",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
