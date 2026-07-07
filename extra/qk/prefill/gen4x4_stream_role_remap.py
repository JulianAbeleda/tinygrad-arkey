"""Generated 4x4 stream role-remap experiments.

This keeps the generated instruction order and math, but rewrites selected raw
physical VGPR operands in the captured generated stream.  The first experiment
splits scalar global_load_u16 data destinations away from the generated
address/scratch band before v_pack consumes them.
"""
from __future__ import annotations

import argparse, ctypes, os, sys
from collections import Counter

os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")
sys.path.insert(0, os.getcwd())

import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Device as DeviceMap
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import Target, colored, getenv
from tinygrad.renderer.amd.dsl import FixedBitField, NULL, Reg, s, v
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_load_u16, global_store_b16, s_endpgm, s_load_b64, s_sendmsg, s_waitcnt,
  v_add_nc_u32_e32, v_and_b32_e32, v_cvt_f16_f32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32,
  v_mov_b32_e32, v_mul_lo_u32, v_pack_b32_f16,
)
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.prefill.gen4x4_i0_harness import final_bytes, generated_4x4_insts

LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")
DATA_BANK = tuple(range(1, 8)) + tuple(range(236, 245))
A_BASE_HALF = {136: 0, 176: 1024, 184: 2048, 192: 3072}
B_BASE_HALF = {144: 0, 152: 16, 160: 32, 168: 48}
ACC = [
  [8, 16, 24, 32],
  [40, 48, 56, 64],
  [72, 80, 88, 96],
  [104, 112, 120, 128],
]


def _clone_inst(inst, **repl):
  kwargs = {name: getattr(inst, name) for name, field in inst._fields if not isinstance(field, FixedBitField)}
  kwargs.update(repl)
  return type(inst)(**kwargs)


def _vgpr_index(r) -> int | None:
  return r.offset - 256 if isinstance(r, Reg) and 256 <= r.offset < 512 else None


def _with_vgpr(inst, field: str, idx: int):
  old = getattr(inst, field)
  return _clone_inst(inst, **{field: Reg(256 + idx, old.sz, neg=old.neg, abs_=old.abs_, hi=old.hi)})


def remap_load_data_for_pack(insts: list[UOp], saddr_filter: int | None = None) -> list[UOp]:
  out: list[UOp] = []
  load_to_data: dict[int, int] = {}
  bank_i = 0
  for u in insts:
    if isinstance(u.arg, tuple):
      out.append(u)
      continue
    inst = u.arg
    name = str(inst).split("(", 1)[0]
    if name == "global_load_u16":
      dst = _vgpr_index(getattr(inst, "vdst", None))
      saddr = getattr(inst, "saddr", None)
      saddr_idx = saddr.offset if isinstance(saddr, Reg) else None
      if dst is not None and 1 <= dst <= 235 and (saddr_filter is None or saddr_idx == saddr_filter):
        ndst = DATA_BANK[bank_i % len(DATA_BANK)]
        bank_i += 1
        load_to_data[dst] = ndst
        inst = _with_vgpr(inst, "vdst", ndst)
      elif dst is not None:
        load_to_data.pop(dst, None)
    elif name == "v_pack_b32_f16":
      repl = {}
      for field in ("src0", "src1"):
        src = _vgpr_index(getattr(inst, field, None))
        if src in load_to_data:
          old = getattr(inst, field)
          repl[field] = Reg(256 + load_to_data[src], old.sz, neg=old.neg, abs_=old.abs_, hi=old.hi)
      if repl: inst = _clone_inst(inst, **repl)
    else:
      for field in ("vdst", "sdst", "sdata", "vdsty"):
        dst = _vgpr_index(getattr(inst, field, None))
        if dst is not None: load_to_data.pop(dst, None)
    out.append(UOp(Ops.INS, arg=inst, tag=u.tag))
  return out


def mutated_insts(mode: str) -> list[UOp]:
  insts = generated_4x4_insts()
  if mode == "baseline": return insts
  if mode == "load-data-bank": return remap_load_data_for_pack(insts)
  if mode == "load-data-bank-s8": return remap_load_data_for_pack(insts, 8)
  if mode == "load-data-bank-s10": return remap_load_data_for_pack(insts, 10)
  if mode == "pre-wmma-scratch-scrub": return pre_wmma_scratch_scrub(insts)
  if mode == "clean-reload-before-wmma": return clean_reload_before_wmma(insts)
  if mode == "clean-reload-before-wmma-swapptrs": return clean_reload_before_wmma(insts, swap_ptrs=True)
  if mode == "clean-epilogue": return replace_epilogue(insts)
  if mode == "clean-reload-clean-epilogue": return replace_epilogue(clean_reload_before_wmma(insts))
  if mode == "epilogue-temp-remap-low": return remap_epilogue_temps(insts, {200: 7, 201: 3, 202: 6})
  if mode == "epilogue-remap-v200": return remap_epilogue_temps(insts, {200: 7})
  if mode == "epilogue-remap-v201": return remap_epilogue_temps(insts, {201: 3})
  if mode == "epilogue-remap-v202": return remap_epilogue_temps(insts, {202: 6})
  if mode == "epilogue-remap-v200-v201": return remap_epilogue_temps(insts, {200: 7, 201: 3})
  if mode == "epilogue-remap-v200-v202": return remap_epilogue_temps(insts, {200: 7, 202: 6})
  if mode == "epilogue-remap-v201-v202": return remap_epilogue_temps(insts, {201: 3, 202: 6})
  raise ValueError(f"unknown mode {mode}")


def final_insts(insts: list[UOp]) -> list[UOp]:
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  uops = list(insts)
  if getenv("AMD_ISA_SCHED", 1): uops = ren._schedule(uops)
  return ren._resolve_labels(ren._insert_waitcnt(uops))


def pre_wmma_scratch_scrub(insts: list[UOp]) -> list[UOp]:
  out: list[UOp] = []
  scrub = tuple(range(203, 219)) + tuple(range(220, 236))
  for u in insts:
    if not isinstance(u.arg, tuple) and str(u.arg).split("(", 1)[0] == "v_wmma_f32_16x16x16_f16":
      out.extend(UOp(Ops.INS, arg=v_mov_b32_e32(v[i], 0)) for i in scrub)
    out.append(u)
  return out


def _field_base(inst, field: str) -> int:
  r = getattr(inst, field)
  if not isinstance(r, Reg): raise ValueError(f"{field} is not a Reg in {inst}")
  return r.offset - 256


def _clean_load_pack(frag_base: int, *, is_a: bool, swap_ptrs: bool = False) -> list[UOp]:
  # v208/v209 are scratch address temps; v220..v235 are scalar half load data.
  out: list[UOp] = []
  if is_a:
    # Generated A addresses are half-element offsets:
    #   lane*64 + kloop*16 + fragment_base_half + i
    # where fragment_base_half follows the physical fragment register chosen by regalloc.
    out.append(UOp(Ops.INS, arg=v_lshlrev_b32_e32(v[208], 1, v[201])))  # lane*64*2
    base_half = A_BASE_HALF[frag_base]
    if base_half: out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[208], base_half * 2, v[208])))
    out.append(UOp(Ops.INS, arg=v_mov_b32_e32(v[209], s[40])))
    out.append(UOp(Ops.INS, arg=v_mul_lo_u32(v[209], v[209], 16 * 2)))
    out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[208], v[208], v[209])))
    saddr, stride = (s[10:11] if swap_ptrs else s[8:9]), 2
  else:
    # Generated B addresses are half-element offsets:
    #   kloop*1024 + lane + fragment_base_half + i*64
    out.append(UOp(Ops.INS, arg=v_lshlrev_b32_e32(v[208], 1, v[200])))  # lane*2
    base_half = B_BASE_HALF[frag_base]
    if base_half: out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[208], base_half * 2, v[208])))
    out.append(UOp(Ops.INS, arg=v_mov_b32_e32(v[209], s[40])))
    out.append(UOp(Ops.INS, arg=v_mul_lo_u32(v[209], v[209], 16 * 64 * 2)))
    out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[208], v[208], v[209])))
    saddr, stride = (s[8:9] if swap_ptrs else s[10:11]), 64 * 2
  for i in range(16):
    out.append(UOp(Ops.INS, arg=global_load_u16(vdst=v[220+i], addr=v[208:208], saddr=saddr, offset=i*stride)))
  out.append(UOp(Ops.INS, arg=s_waitcnt(simm16=0)))
  for i in range(8):
    out.append(UOp(Ops.INS, arg=v_pack_b32_f16(vdst=v[frag_base+i], src0=v[220+2*i], src1=v[220+2*i+1], opsel=0)))
  return out


def clean_reload_before_wmma(insts: list[UOp], swap_ptrs: bool = False) -> list[UOp]:
  out: list[UOp] = []
  for u in insts:
    if not isinstance(u.arg, tuple) and str(u.arg).split("(", 1)[0] == "v_wmma_f32_16x16x16_f16":
      abase, bbase = _field_base(u.arg, "src0"), _field_base(u.arg, "src1")
      out.extend(_clean_load_pack(abase, is_a=True, swap_ptrs=swap_ptrs))
      out.extend(_clean_load_pack(bbase, is_a=False, swap_ptrs=swap_ptrs))
    out.append(u)
  return out


def _clean_epilogue() -> list[UOp]:
  out: list[UOp] = []
  out.append(UOp(Ops.INS, arg=s_load_b64(sdata=s[6:7], sbase=s[0:1], offset=0, soffset=NULL)))
  out.append(UOp(Ops.INS, arg=s_waitcnt(simm16=0)))
  out.append(UOp(Ops.INS, arg=v_and_b32_e32(v[4], 15, v[0])))
  out.append(UOp(Ops.INS, arg=v_lshrrev_b32_e32(v[5], 4, v[0])))
  out.append(UOp(Ops.INS, arg=v_and_b32_e32(v[5], 1, v[5])))
  for tm in range(4):
    for tn in range(4):
      ac = ACC[tm][tn]
      out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[7], tm * 16, v[5])))
      out.append(UOp(Ops.INS, arg=v_mul_lo_u32(v[7], v[7], 64)))
      out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[3], tn * 16, v[4])))
      out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[7], v[7], v[3])))
      out.append(UOp(Ops.INS, arg=v_lshlrev_b32_e32(v[7], 1, v[7])))
      for i in range(8):
        out.append(UOp(Ops.INS, arg=v_cvt_f16_f32_e32(v[6], v[ac + i])))
        out.append(UOp(Ops.INS, arg=global_store_b16(addr=v[7:7], data=v[6], saddr=s[6:7], offset=0)))
        if i < 7: out.append(UOp(Ops.INS, arg=v_add_nc_u32_e32(v[7], 64 * 4, v[7])))
  out.append(UOp(Ops.INS, arg=s_waitcnt(simm16=0)))
  out.append(UOp(Ops.INS, arg=s_sendmsg(simm16=3)))
  out.append(UOp(Ops.INS, arg=s_endpgm()))
  return out


def replace_epilogue(insts: list[UOp]) -> list[UOp]:
  out: list[UOp] = []
  for u in insts:
    out.append(u)
    if isinstance(u.arg, tuple) and u.arg == ("label", ("out", 0)):
      out.extend(_clean_epilogue())
      return out
  raise ValueError("out label not found")


def _remap_inst_vgprs(inst, regmap: dict[int, int]):
  repl = {}
  for name, _field in getattr(inst, "_fields", ()):
    r = getattr(inst, name, None)
    idx = _vgpr_index(r)
    if idx in regmap:
      repl[name] = Reg(256 + regmap[idx], r.sz, neg=r.neg, abs_=r.abs_, hi=r.hi)
  return _clone_inst(inst, **repl) if repl else inst


def remap_epilogue_temps(insts: list[UOp], regmap: dict[int, int]) -> list[UOp]:
  out: list[UOp] = []
  in_epi = False
  for u in insts:
    if isinstance(u.arg, tuple):
      if u.arg == ("label", ("out", 0)): in_epi = True
      out.append(u)
      if u.arg == ("label", ("out", 0)) and 200 in regmap:
        out.append(UOp(Ops.INS, arg=v_mov_b32_e32(v[regmap[200]], v[200])))
      continue
    inst = _remap_inst_vgprs(u.arg, regmap) if in_epi else u.arg
    out.append(UOp(Ops.INS, arg=inst, tag=u.tag))
  return out


def compare(got: np.ndarray, ref: np.ndarray) -> dict[str, float | bool]:
  got32, ref32 = got.astype(np.float32), ref.astype(np.float32)
  nanfrac = float(np.isnan(got32).mean())
  ok = np.isfinite(got32)
  rmse = float(np.sqrt(((got32[ok] - ref32[ok]) ** 2).mean())) if ok.any() else float("nan")
  return {"nan": nanfrac, "rmse": rmse, "pass": bool(nanfrac == 0.0 and rmse < 5e-2)}


def remu_run(mode: str, seed: int = 0) -> dict:
  insts = mutated_insts(mode)
  text = final_bytes(insts)
  np.random.seed(seed)
  A = np.random.randn(64, 64).astype(np.float16)
  B = np.random.randn(64, 64).astype(np.float16)
  OUT = np.zeros((64, 64), dtype=np.float16)
  args = (ctypes.c_uint64 * 3)(OUT.ctypes.data, A.ctypes.data, B.ctypes.data)
  lib = ctypes.CDLL(LIBREMU)
  lib.run_asm.restype = ctypes.c_int
  lib.run_asm.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_uint32] * 6 + [ctypes.POINTER(ctypes.c_uint64)]
  rc = lib.run_asm(ctypes.c_char_p(text), len(text), 1, 1, 1, 32, 1, 1, args)
  r = compare(OUT, A.astype(np.float32) @ B.astype(np.float32))
  print(f"REMU {mode}: rc={rc} bytes={len(text)} insts={len(insts)} nan={r['nan']:.4f} rmse={r['rmse']:.5f} PASS={r['pass']}")
  print(f"  got[0,:6]={OUT.astype(np.float32)[0,:6]}")
  print(f"  ref[0,:6]={(A.astype(np.float32) @ B.astype(np.float32))[0,:6]}")
  return {**r, "rc": rc}


def gpu_run(mode: str, seed: int = 0) -> dict:
  insts = final_insts(mutated_insts(mode))
  np.random.seed(seed)
  A_np = np.random.randn(64, 64).astype(np.float16)
  B_np = np.random.randn(64, 64).astype(np.float16)
  C_np = np.zeros((64, 64), dtype=np.float16)
  A = Tensor(A_np).contiguous().realize()
  B = Tensor(B_np).contiguous().realize()
  C = Tensor(C_np, dtype=dtypes.half, device=A.device).contiguous().realize()

  def asm_kernel(Cp, Ap, Bp):
    g = [UOp.special(1, "gidx0"), UOp.special(1, "gidx1")]
    sink = UOp.sink(Cp.base, Ap.base, Bp.base, *g, UOp.special(32, "lidx0"),
                    arg=KernelInfo(name=colored(f"gen4x4_{mode}", "cyan"),
                                   estimates=Estimates(ops=64*64*64*2, mem=(64*64*3)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(insts))))

  got = C.custom_kernel(A, B, fxn=asm_kernel)[0].numpy()
  ref = A_np.astype(np.float32) @ B_np.astype(np.float32)
  r = compare(got, ref)
  print(f"GPU {mode} on {DeviceMap.DEFAULT}: nan={r['nan']:.4f} rmse={r['rmse']:.5f} PASS={r['pass']}")
  print(f"  got[0,:6]={got.astype(np.float32)[0,:6]}")
  print(f"  ref[0,:6]={ref[0,:6]}")
  return r


def summary(mode: str) -> None:
  insts = mutated_insts(mode)
  mns = [str(u.arg).split("(", 1)[0] for u in insts if not isinstance(u.arg, tuple)]
  print(f"mode={mode} insts={len(insts)} data_bank={DATA_BANK}")
  for k, v_ in Counter(mns).most_common():
    if k in ("global_load_u16", "v_pack_b32_f16", "v_wmma_f32_16x16x16_f16", "global_store_b16"): print(f"{k}: {v_}")
  for i, u in enumerate(insts):
    s = str(u.arg)
    if 180 <= i <= 230 and ("global_load_u16" in s or "v_pack_b32_f16" in s or "v_wmma" in s): print(f"{i:04d} {s}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--mode", default="load-data-bank-s8",
                 choices=("baseline", "load-data-bank", "load-data-bank-s8", "load-data-bank-s10", "pre-wmma-scratch-scrub",
                          "clean-reload-before-wmma", "clean-reload-before-wmma-swapptrs",
                          "clean-epilogue", "clean-reload-clean-epilogue", "epilogue-temp-remap-low",
                          "epilogue-remap-v200", "epilogue-remap-v201", "epilogue-remap-v202",
                          "epilogue-remap-v200-v201", "epilogue-remap-v200-v202", "epilogue-remap-v201-v202"))
  p.add_argument("--summary", action="store_true")
  p.add_argument("--remu", action="store_true")
  p.add_argument("--gpu", action="store_true")
  args = p.parse_args()
  if args.summary: summary(args.mode)
  elif args.gpu: gpu_run(args.mode)
  else: remu_run(args.mode)


if __name__ == "__main__":
  main()
