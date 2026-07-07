"""Fixed-register 4x4 WMMA hand-control experiment.

This intentionally owns a standalone stream instead of mutating codegen.  It
uses kernarg order [A, B, OUT]:
  A   fp16 row-major [64, 64]
  B   fp16 row-major [64, 64]  (not transposed)
  OUT fp16 row-major [64, 64]

The experiment mirrors the generated static physical layout under test:
fragments are produced by scalar u16 global loads followed by v_pack_b32_f16,
accumulators live in the generated fixed slots, and epilogue scratch avoids
v8..v135.
"""
from __future__ import annotations

import argparse, ctypes, os, sys
from collections import Counter
from typing import Iterable

os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")
sys.path.insert(0, os.getcwd())

import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Device as DeviceMap
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import Target, colored, getenv
from tinygrad.renderer.amd.dsl import NULL, Reg, s, v
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.uop.ops import KernelInfo, Ops, UOp

LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")

M = N = K = 64
TM = TN = 4
NK = K // 16

A_FRAG = [176, 184, 192, 136]        # logical tm 0..3
B_FRAG = [152, 160, 168, 144]        # logical tn 0..3
ACC = [
  [ 48,  56,  64,  40],
  [ 80,  88,  96,  72],
  [112, 120, 128, 104],
  [ 16,  24,  32,   8],
]
VA = 200
ADDR = 208
SCR = 220


def waitcnt_vm(n:int):
  if getenv("FULLWAIT", 0): return s_waitcnt(simm16=0)
  return s_waitcnt(simm16=(0x7) | (0x3F << 4) | ((n & 0x3F) << 10))


def _append_pack_load(I:list, frag_base:int, va_reg:int, saddr:Reg, *, stride_bytes:int) -> None:
  """Load 16 lane-varying fp16 values into v220..v235, then pack to frag_base..frag_base+7."""
  for i in range(16):
    I.append(v_add_nc_u32_e32(v[ADDR], i * stride_bytes, v[va_reg]))
    I.append(global_load_u16(vdst=v[SCR + i], addr=v[ADDR:ADDR], saddr=saddr, offset=0))
  I.append(waitcnt_vm(0))
  for i in range(8):
    I.append(v_pack_b32_f16(vdst=v[frag_base + i], src0=v[SCR + 2*i], src1=v[SCR + 2*i + 1], opsel=0))


def build_genlayout_fixed_u16_4x4() -> list:
  I:list = []
  def e(inst): I.append(inst); return inst
  def load_a(tm:int): _append_pack_load(I, A_FRAG[tm], VA + tm, s[4:5], stride_bytes=2)
  def load_b(tn:int): _append_pack_load(I, B_FRAG[tn], VA + TM + tn, s[6:7], stride_bytes=N * 2)
  def wmma(tm:int, tn:int):
    ac = ACC[tm][tn]
    e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[A_FRAG[tm]:A_FRAG[tm]+7],
                              src1=v[B_FRAG[tn]:B_FRAG[tn]+7], src2=v[ac:ac+7]))

  # kernarg [A, B, OUT]: A=s[4:5]@0, B=s[6:7]@8, OUT=s[8:9]@16.
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))

  # lane-col/row helpers. s10/s11 are tile origins; single workgroup keeps them zero.
  e(v_and_b32_e32(v[1], 15, v[0]))
  e(s_lshl_b32(s[10], s[3], 6))  # gidx1 * 64 rows
  e(s_lshl_b32(s[11], s[2], 6))  # gidx0 * 64 cols
  e(v_add_nc_u32_e32(v[2], s[10], v[1]))  # A row = tile_m + lane&15
  e(v_add_nc_u32_e32(v[3], s[11], v[1]))  # B col = tile_n + lane&15
  for tm in range(TM):
    e(v_add_nc_u32_e32(v[VA + tm], tm * 16, v[2]) if tm else v_mov_b32_e32(v[VA + tm], v[2]))
    e(v_mul_lo_u32(v[VA + tm], v[VA + tm], K * 2))
  for tn in range(TN):
    e(v_add_nc_u32_e32(v[VA + TM + tn], tn * 16, v[3]) if tn else v_mov_b32_e32(v[VA + TM + tn], v[3]))
    e(v_lshlrev_b32_e32(v[VA + TM + tn], 1, v[VA + TM + tn]))  # byte col; row offset added by load stride

  for r in range(8, 136): e(v_mov_b32_e32(v[r], 0))

  e(s_mov_b32(s[16], 0))
  e(("label", ("top", "kloop")))
  e(s_cmp_lt_i32(s[16], NK))
  e(("branch", "s_cbranch_scc0", ("out", "kloop")))
  # Exact generated physical order under test.
  load_b(0)
  load_a(0); wmma(0, 0)
  load_b(1); wmma(0, 1)
  load_b(2); wmma(0, 2)
  load_b(3); wmma(0, 3)
  load_a(1)
  for tn in range(TN): wmma(1, tn)
  load_a(2)
  for tn in range(TN): wmma(2, tn)
  load_a(3)
  for tn in range(TN): wmma(3, tn)
  for tm in range(TM): e(v_add_nc_u32_e32(v[VA + tm], 32, v[VA + tm]))
  for tn in range(TN): e(v_add_nc_u32_e32(v[VA + TM + tn], 16 * N * 2, v[VA + TM + tn]))
  e(s_add_i32(s[16], s[16], 1))
  e(("branch", "s_branch", ("top", "kloop")))
  e(("label", ("out", "kloop")))

  # Epilogue scratch is v3,v4,v5,v6,v7 only.  v8 is never used here.
  e(v_and_b32_e32(v[4], 15, v[0]))
  e(v_lshrrev_b32_e32(v[5], 4, v[0])); e(v_and_b32_e32(v[5], 1, v[5]))
  for tm in range(TM):
    for tn in range(TN):
      ac = ACC[tm][tn]
      e(v_add_nc_u32_e32(v[7], s[10], v[5])); e(v_add_nc_u32_e32(v[7], tm * 16, v[7]))
      e(v_add_nc_u32_e32(v[3], s[11], v[4])); e(v_add_nc_u32_e32(v[3], tn * 16, v[3]))
      e(v_mul_lo_u32(v[7], v[7], N)); e(v_add_nc_u32_e32(v[7], v[7], v[3]))
      e(v_lshlrev_b32_e32(v[7], 1, v[7]))
      for i in range(8):
        e(v_cvt_f16_f32_e32(v[6], v[ac + i]))
        e(global_store_b16(addr=v[7:7], data=v[6], saddr=s[8:9], offset=0))
        if i < 7: e(v_add_nc_u32_e32(v[7], N * 4, v[7]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  return I


def final_insts(I:list) -> list[UOp]:
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  uops = [UOp(Ops.INS, arg=i) for i in I]
  if getenv("AMD_ISA_SCHED", 1): uops = ren._schedule(uops)
  return ren._resolve_labels(ren._insert_waitcnt(uops))


def final_bytes(I:list) -> bytes:
  raw = b"".join(u.arg.to_bytes() for u in final_insts(I))
  assert len(raw) % 4 == 0
  return raw


def compare(got:np.ndarray, ref:np.ndarray) -> dict[str, float | bool]:
  got32, ref32 = got.astype(np.float32), ref.astype(np.float32)
  nanfrac = float(np.isnan(got32).mean())
  ok = np.isfinite(got32)
  rmse = float(np.sqrt(((got32[ok] - ref32[ok]) ** 2).mean())) if ok.any() else float("nan")
  return {"nan": nanfrac, "rmse": rmse, "pass": bool(nanfrac == 0.0 and rmse < 5e-2)}


def remu_run(seed:int=0) -> dict[str, float | bool | int]:
  I = build_genlayout_fixed_u16_4x4()
  text = final_bytes(I)
  np.random.seed(seed)
  A = np.random.randn(M, K).astype(np.float16)
  B = np.random.randn(K, N).astype(np.float16)
  OUT = np.zeros((M, N), dtype=np.float16)
  args = (ctypes.c_uint64 * 3)(A.ctypes.data, B.ctypes.data, OUT.ctypes.data)
  lib = ctypes.CDLL(LIBREMU)
  lib.run_asm.restype = ctypes.c_int
  lib.run_asm.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_uint32] * 6 + [ctypes.POINTER(ctypes.c_uint64)]
  rc = lib.run_asm(ctypes.c_char_p(text), len(text), 1, 1, 1, 32, 1, 1, args)
  r = compare(OUT, A.astype(np.float32) @ B.astype(np.float32))
  print(f"REMU genlayout_fixed_u16_4x4: rc={rc} bytes={len(text)} insts={len(final_insts(I))} "
        f"nan={r['nan']:.4f} rmse={r['rmse']:.5f} PASS={r['pass']}")
  print(f"  got[0,:6]={OUT.astype(np.float32)[0,:6]}")
  print(f"  ref[0,:6]={(A.astype(np.float32) @ B.astype(np.float32))[0,:6]}")
  return {**r, "rc": rc, "bytes": len(text), "insts": len(final_insts(I))}


def gpu_run(seed:int=0) -> dict[str, float | bool]:
  I = build_genlayout_fixed_u16_4x4()
  np.random.seed(seed)
  A_np = np.random.randn(M, K).astype(np.float16)
  B_np = np.random.randn(K, N).astype(np.float16)
  C_np = np.zeros((M, N), dtype=np.float16)
  A = Tensor(A_np).contiguous().realize()
  B = Tensor(B_np).contiguous().realize()
  C = Tensor(C_np, dtype=dtypes.half, device=A.device).contiguous().realize()

  def asm_kernel(a, b, c):
    g = [UOp.special(1, "gidx0"), UOp.special(1, "gidx1")]
    sink = UOp.sink(a.base, b.base, c.base, *g, UOp.special(32, "lidx0"),
                    arg=KernelInfo(name=colored("genlayout_fixed_u16_4x4", "cyan"),
                                   estimates=Estimates(ops=M*N*K*2, mem=(M*K + K*N + M*N) * 2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in I))))

  out = Tensor.custom_kernel(A, B, C, fxn=asm_kernel)[2]
  got = out.numpy()
  ref = A_np.astype(np.float32) @ B_np.astype(np.float32)
  r = compare(got, ref)
  print(f"GPU genlayout_fixed_u16_4x4 on {DeviceMap.DEFAULT}: nan={r['nan']:.4f} rmse={r['rmse']:.5f} PASS={r['pass']}")
  print(f"  got[0,:6]={got.astype(np.float32)[0,:6]}")
  print(f"  ref[0,:6]={ref[0,:6]}")
  return r


def _inst_name(u:UOp) -> str:
  return str(u.arg).split("(", 1)[0] if not isinstance(u.arg, tuple) else str(u.arg[0])


def _vgprs(insts:Iterable[UOp]) -> set[int]:
  regs:set[int] = set()
  for u in insts:
    if isinstance(u.arg, tuple): continue
    for name, _field in getattr(u.arg, "_fields", ()):
      rr = getattr(u.arg, name, None)
      if isinstance(rr, Reg) and rr.offset >= 256:
        n = getattr(u.arg, "op_regs", {}).get(name, 1)
        regs.update(range(rr.offset - 256, rr.offset - 256 + n))
  return regs


def summary() -> None:
  I = build_genlayout_fixed_u16_4x4()
  fin = final_insts(I)
  hist = Counter(_inst_name(u) for u in fin)
  regs = _vgprs(fin)
  print("genlayout_fixed_u16_4x4")
  print("kernarg: [A row-major fp16[64,64], B row-major fp16[64,64], OUT row-major fp16[64,64]]")
  print(f"pre_insts={len(I)} final_insts={len(fin)} bytes={len(final_bytes(I))}")
  print(f"max_vgpr=v{max(regs)} count={len(regs)}")
  print(f"A_FRAG logical tm0..3: {A_FRAG}")
  print(f"B_FRAG logical tn0..3: {B_FRAG}")
  print(f"ACC logical tm,tn: {ACC}")
  print("reserved: ACC v8..v135, fragments v136..v199, VA v200..v207, addr v208, scratch v220..v235")
  print("histogram:")
  for name, count in sorted(hist.items()):
    print(f"  {name}: {count}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--summary", action="store_true", help="print instruction histogram and register map")
  p.add_argument("--remu", action="store_true", help="run remu validation")
  p.add_argument("--gpu", action="store_true", help="run on DEV=AMD:ISA")
  args = p.parse_args()
  if args.summary: summary()
  elif args.gpu: gpu_run()
  else: remu_run()


if __name__ == "__main__":
  main()
