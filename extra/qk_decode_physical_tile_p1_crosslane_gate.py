#!/usr/bin/env python3
"""P1 primitive gate: generated decode-attention q.k lane map + cross-lane score reduce.

This is not a full decode route. It proves the first missing primitive can be
emitted by generated UOps and detected by the primitive-space tooling:
  LaneMap/CrossLane score sharing via ds_bpermute.
"""
from __future__ import annotations

import ctypes, json, pathlib, re, time
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp
from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
_F32 = dtypes.float32


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())


def flash_p1_crosslane_score_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, Tc: int):
  if Hd % 32 != 0: raise ValueError(f"P1 requires Hd divisible by 32, got {Hd}")
  G = Hq // Hkv; R = Hd // 32
  def kernel(score: UOp, q: UOp, cache: UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    e = lane * R + r
    kvh = h // G
    acc = UOp.placeholder((1,), _F32, 170, addrspace=AddrSpace.REG)
    acc_init = acc.after(h, t)[0].store(0.0)
    acc = acc.after(acc_init)
    qv = q[h * Hd + e].cast(_F32)
    kv = cache[((0 * Hkv + kvh) * MAXC + t) * Hd + e].cast(_F32)
    acc_upd = acc[0].store(acc.after(r)[0] + qv * kv).end(r)
    total = _warp_reduce_sum_staged(acc.after(acc_upd)[0], lane, 32)
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_p1_crosslane_score_{Hq}_{Hd}"))
  return kernel


def _parse_desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _relocs = elf_loader(lib)
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
  h = {"total": 0, "valu": 0, "s_inst": 0, "vmem_load": 0, "vmem_store": 0, "ds": 0, "cross_lane": 0, "fma_dot": 0, "scratch": 0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+)\b", line)
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
  return h


def _numeric_and_isa() -> dict[str, Any]:
  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig_runtime = dev.runtime
  def runtime_hook(name, lib, **kw):
    if name.startswith("flash_p1_crosslane_score") and name not in captured: captured[name] = lib
    return orig_runtime(name, lib, **kw)
  dev.runtime = runtime_hook

  Hq, Hkv, Hd, MAXC, Tc = 32, 8, 128, 256, 192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0.0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, Hkv, MAXC, Hd), dtype=np.float32)
  cache[0] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)
  got = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)),
    fxn=flash_p1_crosslane_score_kernel(Hd, Hq, Hkv, MAXC, Tc))[0].realize().numpy().reshape(Hq, MAXC)
  ref = np.zeros((Hq, MAXC), dtype=np.float32)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    ref[h, :Tc] = (cache[0, kvh, :Tc, :] @ q[h]) * (1.0 / np.sqrt(Hd))
  diff = got[:, :Tc] - ref[:, :Tc]

  kernels = {}
  OUT.mkdir(parents=True, exist_ok=True)
  for name, lib in captured.items():
    asm = _disasm(lib)
    (OUT / f"disasm_{name}.txt").write_text(asm)
    desc = _parse_desc(lib)
    desc["hist"] = _hist(asm)
    desc["primitive_flags"] = {
      "has_v_dot2": "v_dot2" in asm,
      "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
      "has_cross_lane": bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)", asm)),
      "has_vector_global_load": "global_load" in asm or "buffer_load" in asm,
      "has_spill": bool(re.search(r"\bscratch_(load|store)", asm)),
    }
    kernels[name] = desc
  max_abs = float(np.max(np.abs(diff)))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  return {
    "checked": True,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "Tc": Tc},
    "numeric": {"max_abs": max_abs, "rmse": rmse, "pass": bool(max_abs <= 1e-4 and rmse <= 1e-5), "thresholds": {"max_abs": 1e-4, "rmse": 1e-5}},
    "kernels": kernels,
  }


def build() -> dict[str, Any]:
  probe = _numeric_and_isa()
  kernel = next(iter(probe.get("kernels", {}).values()), {})
  flags = kernel.get("primitive_flags", {})
  if not probe["numeric"].get("pass"):
    verdict = "P1_CROSSLANE_FAIL__NUMERIC"
  elif not flags.get("has_cross_lane"):
    verdict = "P1_CROSSLANE_FAIL__NO_CROSS_LANE_ISA"
  elif flags.get("has_lds") or flags.get("has_v_dot2"):
    verdict = "P1_CROSSLANE_PASS__EXTRA_PRIMITIVES_PRESENT"
  else:
    verdict = "P1_CROSSLANE_PASS__LANEMAP_CROSSLANE_VISIBLE"
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "candidate_id": "decode_attention_physical_tile_p1",
    "verdict": verdict,
    "probe": probe,
    "primitive_visibility": {
      "LaneMap": "searchable_probe_visible",
      "CrossLane": "detected_in_isa" if flags.get("has_cross_lane") else "absent",
      "TileMemory": "not_targeted_in_p1",
      "DotLowering": "scalar_fma_not_v_dot2_in_p1",
    },
    "decision": "P1 proves generated cross-lane score reduction can be emitted; next integrate as a decode route only if the detector/gap gate consumes this as partial primitive visibility."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "p1_crosslane_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"p1-crosslane-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"].startswith("P1_CROSSLANE_PASS") else 1


if __name__ == "__main__":
  raise SystemExit(main())
