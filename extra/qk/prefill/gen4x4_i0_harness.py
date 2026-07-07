"""I0 faithful generated-stream harness for the 4x4 WMMA NaN.

This captures the exact AMDISARenderer post-regalloc LINEAR instruction stream for
the forced-TC 64x64x64, WM=WN=4 case, then relaunches that stream through
Tensor.custom_kernel with the generated kernarg order:

  [OUT, A, B]

The hand faultprobe uses [A, Bt, OUT], so this file intentionally does not share
that launch convention. This is the mutation substrate for allocator/isel
terminal isolation: first prove the captured generated stream reproduces the
same hardware NaN as the normal DEV=AMD:ISA path.
"""
from __future__ import annotations

import ctypes, os, sys
from dataclasses import replace

os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")
sys.path.insert(0, os.getcwd())

import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.device import Device as DeviceMap
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import Target, colored
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp

LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")


def generated_4x4_program():
  a = Tensor.empty(64, 64, dtype="half")
  b = Tensor.empty(64, 64, dtype="half")
  lin = (a @ b).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  opts = (Opt(OptOps.TC, axis=0, arg=(0, 0, 1)), Opt(OptOps.UPCAST, axis=0, arg=4), Opt(OptOps.UPCAST, axis=0, arg=4))
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=opts))
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  return to_program(ast, ren), ren


def generated_4x4_insts():
  prg, _ren = generated_4x4_program()
  return list([u for u in prg.src if u.op is Ops.LINEAR][0].src)


def final_bytes(insts:list[UOp]) -> bytes:
  raw = b"".join(u.arg.to_bytes() for u in final_insts(insts))
  assert len(raw) % 4 == 0
  return raw


def final_insts(insts:list[UOp]) -> list[UOp]:
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  from tinygrad.helpers import getenv
  uops = list(insts)
  if getenv("AMD_ISA_SCHED", 1): uops = ren._schedule(uops)
  return ren._resolve_labels(ren._insert_waitcnt(uops))


def compare(got:np.ndarray, ref:np.ndarray) -> dict[str, float | bool]:
  got32, ref32 = got.astype(np.float32), ref.astype(np.float32)
  nanfrac = float(np.isnan(got32).mean())
  ok = np.isfinite(got32)
  rmse = float(np.sqrt(((got32[ok] - ref32[ok]) ** 2).mean())) if ok.any() else float("nan")
  return {"nanfrac": nanfrac, "rmse": rmse, "pass": bool(nanfrac == 0.0 and rmse < 5e-2)}


def remu_run(seed:int=0):
  insts = generated_4x4_insts()
  np.random.seed(seed)
  A = np.random.randn(64, 64).astype(np.float16)
  B = np.random.randn(64, 64).astype(np.float16)
  OUT = np.zeros((64, 64), dtype=np.float16)
  text = final_bytes(insts)
  args = (ctypes.c_uint64 * 3)(OUT.ctypes.data, A.ctypes.data, B.ctypes.data)  # generated kernarg [OUT, A, B]
  lib = ctypes.CDLL(LIBREMU)
  lib.run_asm.restype = ctypes.c_int
  lib.run_asm.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_uint32] * 6 + [ctypes.POINTER(ctypes.c_uint64)]
  rc = lib.run_asm(ctypes.c_char_p(text), len(text), 1, 1, 1, 32, 1, 1, args)
  r = compare(OUT, A.astype(np.float32) @ B.astype(np.float32))
  print(f"REMU gen4x4: insts={len(insts)} bytes={len(text)} rc={rc} nan={r['nanfrac']:.4f} rmse={r['rmse']:.5f} PASS={r['pass']}")
  print(f"  got[0,:6]={OUT.astype(np.float32)[0,:6]}")
  print(f"  ref[0,:6]={(A.astype(np.float32) @ B.astype(np.float32))[0,:6]}")
  return r


def gpu_run(seed:int=0):
  insts = final_insts(generated_4x4_insts())
  np.random.seed(seed)
  A_np = np.random.randn(64, 64).astype(np.float16)
  B_np = np.random.randn(64, 64).astype(np.float16)
  C_np = np.zeros((64, 64), dtype=np.float16)
  A = Tensor(A_np).contiguous().realize()
  B = Tensor(B_np).contiguous().realize()
  C = Tensor(C_np, dtype=dtypes.half, device=A.device).contiguous().realize()
  grid = (1, 1, 1)
  local = 32

  def asm_kernel(Cp, Ap, Bp):
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(Cp.base, Ap.base, Bp.base, *g, UOp.special(local, "lidx0"),
                    arg=KernelInfo(name=colored("gen4x4_i0_exact", "cyan"),
                                   estimates=Estimates(ops=64*64*64*2, mem=(64*64*3)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(insts))))

  out = C.custom_kernel(A, B, fxn=asm_kernel)[0]
  got = out.numpy()
  ref = A_np.astype(np.float32) @ B_np.astype(np.float32)
  r = compare(got, ref)
  dev = DeviceMap.DEFAULT
  print(f"GPU gen4x4 on {dev}: insts={len(insts)} nan={r['nanfrac']:.4f} rmse={r['rmse']:.5f} PASS={r['pass']}")
  print(f"  got[0,:6]={got.astype(np.float32)[0,:6]}")
  print(f"  ref[0,:6]={ref[0,:6]}")
  return r


def summary():
  prg, _ren = generated_4x4_program()
  lin = [u for u in prg.src if u.op is Ops.LINEAR][0]
  print(prg.arg)
  print(f"insts={len(lin.src)}")
  mns = [str(u.arg).split("(", 1)[0] for u in lin.src if not isinstance(u.arg, tuple)]
  for mn in ("v_wmma_f32_16x16x16_f16", "v_pack_b32_f16", "global_load_u16", "global_store_b16"):
    print(f"{mn}: {sum(1 for x in mns if x == mn)}")


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "--summary"
  if mode == "--summary": summary()
  elif mode == "--remu": remu_run()
  elif mode == "--gpu": gpu_run()
  else:
    raise SystemExit("usage: gen4x4_i0_harness.py [--summary|--remu|--gpu]")
