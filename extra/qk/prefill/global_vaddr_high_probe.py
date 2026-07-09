"""Probe whether RDNA3 GLOBAL addr observes the adjacent VGPR high half.

The generated schedule-search stream uses GLOBAL_LOAD_B128 with a low VGPR byte
offset and assumes the adjacent VGPR is irrelevant. This probe varies only that
adjacent register:

  zero:   v3 = 0
  poison: v3 = nonzero

If poison faults while zero passes, GLOBAL addr must be treated as a 64-bit VGPR
pair in this mode, and generated code must ensure addr+1 is zero/controlled.
"""
from __future__ import annotations

import argparse, os, sys

os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")
sys.path.insert(0, os.getcwd())

import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import colored
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_load_b128, global_store_b32, s_endpgm, s_load_b64, s_sendmsg, s_waitcnt, v_mov_b32_e32,
)


def build_probe(high: int) -> list[UOp]:
  insts = []
  def e(inst): insts.append(UOp(Ops.INS, arg=inst))

  # Kernarg order from asm_kernel: [OUT, IN].
  e(s_load_b64(sdata=s[4:5], sbase=s[0:1], offset=8, soffset=NULL))  # input pointer
  e(s_load_b64(sdata=s[6:7], sbase=s[0:1], offset=0, soffset=NULL))  # output pointer
  e(s_waitcnt(simm16=0))

  # Load address low/high candidate. GLOBAL_LOAD_B128 names v2 as addr; v3 is
  # deliberately adjacent and should be ignored if the address is truly 32-bit.
  e(v_mov_b32_e32(v[2], 0))
  e(v_mov_b32_e32(v[3], high))
  e(global_load_b128(vdst=v[8:11], addr=v[2], saddr=s[4:5], offset=0))
  e(s_waitcnt(simm16=0))

  # Store the first loaded dword back to OUT[0]. Keep the store's adjacent high
  # address VGPR zero so the variant only tests the load-side high half.
  e(v_mov_b32_e32(v[12], 0))
  e(v_mov_b32_e32(v[13], 0))
  e(global_store_b32(addr=v[12], data=v[8], saddr=s[6:7], offset=0))
  e(s_waitcnt(simm16=0))
  e(s_sendmsg(simm16=3))
  e(s_endpgm())
  return insts


def gpu_run(high: int) -> dict[str, object]:
  inp_np = np.arange(16, dtype=np.float32) + 1.25
  out_np = np.zeros(4, dtype=np.float32)
  inp = Tensor(inp_np).contiguous().realize()
  out = Tensor(out_np, dtype=dtypes.float32, device=inp.device).contiguous().realize()
  insts = build_probe(high)

  def asm_kernel(outp, inp_):
    sink = UOp.sink(outp.base, inp_.base, UOp.special(32, "lidx0"),
                    arg=KernelInfo(name=colored(f"global_vaddr_high_{high:x}", "cyan"),
                                   estimates=Estimates(ops=1, mem=20)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(insts))))

  got = out.custom_kernel(inp, fxn=asm_kernel)[0].numpy()
  return {
    "high": high,
    "got0": float(got[0]),
    "expected0": float(inp_np[0]),
    "pass": bool(np.isfinite(got[0]) and got[0] == inp_np[0]),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=("zero", "poison", "both"), default="both")
  ap.add_argument("--poison", type=lambda x: int(x, 0), default=0x10000)
  args = ap.parse_args()
  highs = []
  if args.variant in ("zero", "both"): highs.append(0)
  if args.variant in ("poison", "both"): highs.append(args.poison)
  for high in highs:
    print("PROBE_RESULT", gpu_run(high), flush=True)


if __name__ == "__main__":
  main()
