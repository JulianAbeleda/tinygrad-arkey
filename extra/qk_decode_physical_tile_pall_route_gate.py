#!/usr/bin/env python3
"""PALL physical-tile route gate.

Attempts to compose the missing physical primitives in one generated hot builder.
This is a pre-route gate: it does not run W==D unless the composed builder proves
numeric correctness and emits the required physical primitive ISA.
"""
from __future__ import annotations

import ctypes, json, os, pathlib, re, time, traceback
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
_F32 = dtypes.float32


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())


def flash_pall_lds_crosslane_score_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, Tc:int):
  """Composed physical-primitive score builder.

  This deliberately targets the first hard composition: LDS K staging + lane-sharded q.k + cross-lane score reduce
  in one generated kernel. v_dot2 is checked in emitted ISA; it is not assumed.
  """
  if Hd % 32 != 0: raise ValueError(f"PALL score requires Hd divisible by 32, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(score:UOp, q:UOp, cache:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    e = lane * R + r
    kvh = h // G
    klds = UOp.placeholder((Hd,), dtypes.half, 190, addrspace=AddrSpace.LOCAL)
    # Cooperative K stage. Each lane writes R positions into LDS.
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t) * Hd + e].cast(dtypes.half)).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 3, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    acc = UOp.placeholder((1,), _F32, 191, addrspace=AddrSpace.REG)
    init = acc.after(h, t)[0].store(0.0)
    acc = acc.after(init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (acc.after(rp)[0], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    upd = acc[0].store(dot2).end(rp)
    total = _warp_reduce_sum_staged(acc.after(upd)[0], lane, 32)
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_pall_lds_crosslane_score_{Hq}_{Hd}"))
  return kernel


def _parse_desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _ = elf_loader(lib)
  rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
  rsrc1 = desc.compute_pgm_rsrc1
  gran_vgpr = rsrc1 & 0x3f
  gran_sgpr = (rsrc1 >> 6) & 0xf
  return {"vgpr": (gran_vgpr + 1) * 8, "sgpr": (gran_sgpr + 1) * 8, "lds": desc.group_segment_fixed_size,
          "scratch": desc.private_segment_fixed_size, "kernarg": desc.kernarg_size, "rsrc1": hex(rsrc1)}


def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  return system(f"{objdump} -d -", input=lib)


def _hist(asm: str) -> dict[str, int]:
  h = {"total":0,"valu":0,"s_inst":0,"vmem_load":0,"vmem_store":0,"ds":0,"cross_lane":0,"fma_dot":0,"scratch":0,"barrier":0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+|s_barrier)\b", line)
    if not m: continue
    op = m.group(1); h["total"] += 1
    if op.startswith("v_"): h["valu"] += 1
    if op.startswith("s_"): h["s_inst"] += 1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"] += 1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"] += 1
    if op.startswith("ds_"): h["ds"] += 1
    if op.startswith(("ds_bpermute", "ds_permute", "ds_swizzle")) or op.startswith("v_permlane"): h["cross_lane"] += 1
    if "fma" in op or "dot" in op or "mac" in op: h["fma_dot"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
    if op == "s_barrier": h["barrier"] += 1
  return h


def _flags(asm: str) -> dict[str, bool]:
  return {
    "has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
    "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
    "has_cross_lane": bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)", asm)),
    "has_barrier": "s_barrier" in asm,
    "has_vector_global_load": "global_load" in asm or "buffer_load" in asm,
    "has_spill": bool(re.search(r"\bscratch_(load|store)", asm)),
  }


def _attempt() -> dict[str, Any]:
  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig_runtime = dev.runtime
  def hook(name, lib, **kw):
    if name.startswith("flash_pall_lds_crosslane_score") and name not in captured: captured[name] = lib
    return orig_runtime(name, lib, **kw)
  dev.runtime = hook

  os.environ.setdefault("V_DOT2_LOWERING", "1")
  Hq,Hkv,Hd,MAXC,Tc = 32,8,128,256,192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0,0.25,(Hq,Hd)).astype(np.float16)
  cache = np.zeros((2,Hkv,MAXC,Hd),np.float16)
  cache[0] = rng.normal(0,0.25,(Hkv,MAXC,Hd)).astype(np.float16)
  got = Tensor.empty(Hq*MAXC, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)), fxn=flash_pall_lds_crosslane_score_kernel(Hd,Hq,Hkv,MAXC,Tc))[0].realize().numpy().reshape(Hq,MAXC)
  ref = np.zeros((Hq,MAXC),np.float32)
  for h in range(Hq): ref[h,:Tc] = (cache[0,h//(Hq//Hkv),:Tc,:].astype(np.float32) @ q[h].astype(np.float32)) * (1.0/np.sqrt(Hd))
  diff = got[:,:Tc] - ref[:,:Tc]
  kernels = {}
  OUT.mkdir(parents=True, exist_ok=True)
  for name,lib in captured.items():
    asm = _disasm(lib)
    (OUT / f"disasm_{name}.txt").write_text(asm)
    d = _parse_desc(lib); d["hist"] = _hist(asm); d["primitive_flags"] = _flags(asm); kernels[name] = d
  return {"checked": True, "shape": {"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"MAXC":MAXC,"Tc":Tc},
          "numeric": {"max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff*diff))),
                      "pass": bool(float(np.max(np.abs(diff))) <= 1e-4)}, "kernels": kernels}


def build() -> dict[str, Any]:
  try:
    attempt = _attempt()
  except Exception as e:
    tb = traceback.format_exc()
    return {"date":"2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "candidate_id":"decode_attention_physical_tile_pall_route", "verdict":"PALL_ROUTE_BLOCKED__BUILDER_EXCEPTION",
            "exception_type": type(e).__name__, "exception": str(e), "traceback_tail": tb[-5000:],
            "decision":"Fix composed LDS+crosslane builder before route integration or W==D."}
  k = next(iter(attempt.get("kernels", {}).values()), {})
  flags = k.get("primitive_flags", {})
  missing = []
  if not flags.get("has_lds"): missing.append("TileMemory.lds_tile")
  if not flags.get("has_cross_lane"): missing.append("CrossLane.reduce_broadcast")
  if not flags.get("has_v_dot2"): missing.append("DotLowering.v_dot2")
  if not attempt.get("numeric", {}).get("pass"): verdict = "PALL_ROUTE_BLOCKED__NUMERIC"
  elif missing: verdict = "PALL_ROUTE_BLOCKED__MISSING_COMPOSED_PRIMITIVES"
  else: verdict = "PALL_ROUTE_BUILDER_READY__ROUTE_NEXT"
  return {"date":"2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id":"decode_attention_physical_tile_pall_route", "verdict": verdict,
          "attempt": attempt, "missing_composed_primitives": missing,
          "required_next_if_ready": ["route flag", "route/materialization gate", "primitive detector", "W==D"],
          "decision": "Builder composes LDS+crosslane but does not yet compose v_dot2; next work is fdot2 pattern inside the composed tile." if missing == ["DotLowering.v_dot2"] else
                      "Proceed to route integration only after all composed primitive checks pass." if missing else
                      "Composed hot builder is ready for default-off route integration."}


def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "route_pall_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"route-pall-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] in ("PALL_ROUTE_BUILDER_READY__ROUTE_NEXT", "PALL_ROUTE_BLOCKED__MISSING_COMPOSED_PRIMITIVES") else 1

if __name__ == "__main__":
  os.environ.setdefault("V_DOT2_LOWERING", "1")
  raise SystemExit(main())
