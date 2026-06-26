#!/usr/bin/env python3
"""All-missing-primitives visibility bundle for decode-attention physical tile search.

This gate intentionally proves primitive visibility in one bundle before route integration:
  - LaneMap / q.k score reuse: P1 cross-lane score reduce probe
  - CrossLane.reduce_broadcast: P1 ds_bpermute ISA
  - DotLowering.v_dot2: existing generated fdot2 lowering probe
  - TileMemory.lds_tile: minimal generated LDS stage probe; barrier is tracked separately because same-lane minimal probes can elide it

It does not claim the current fused route has these primitives. It proves the
building blocks are emit/detect-visible so the next step can be a full route
integration candidate instead of another one-off audit.
"""
from __future__ import annotations

import ctypes, json, os, pathlib, re, subprocess, sys, time
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
_F32 = dtypes.float32


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())


def lds_stage_probe_kernel(N: int = 32):
  def kernel(out: UOp, inp: UOp) -> UOp:
    lane = UOp.special(N, "lidx0")
    lds = UOp.placeholder((N,), _F32, 180, addrspace=AddrSpace.LOCAL)
    st = lds[lane].store(inp[lane])
    bar = UOp.barrier(UOp.group(st))
    return out[lane].store(lds.after(bar)[lane]).sink(arg=_fki("flash_pall_lds_stage_probe"))
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
  h = {"total": 0, "valu": 0, "s_inst": 0, "vmem_load": 0, "vmem_store": 0, "ds": 0, "cross_lane": 0, "barrier_wait": 0, "scratch": 0}
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
    if op == "s_barrier" or "s_waitcnt" in op: h["barrier_wait"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
  return h


def _primitive_flags(asm: str) -> dict[str, bool]:
  return {
    "has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
    "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
    "has_cross_lane": bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)", asm)),
    "has_barrier": "s_barrier" in asm,
    "has_vector_global_load": "global_load" in asm or "buffer_load" in asm,
    "has_spill": bool(re.search(r"\bscratch_(load|store)", asm)),
  }


def _run_json_script(script: str) -> dict[str, Any]:
  p = subprocess.run([sys.executable, script], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), "DEV": "AMD"}, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  if p.returncode != 0:
    return {"ran": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}
  txt = p.stdout or ""
  for i, ch in enumerate(txt):
    if ch != "{": continue
    try:
      d = json.loads(txt[i:])
      d["ran"] = True
      return d
    except Exception:
      pass
  return {"ran": False, "returncode": p.returncode, "error": "no json", "output_tail": txt[-8000:]}


def _lds_probe() -> dict[str, Any]:
  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig_runtime = dev.runtime
  def runtime_hook(name, lib, **kw):
    if name.startswith("flash_pall_lds_stage_probe") and name not in captured: captured[name] = lib
    return orig_runtime(name, lib, **kw)
  dev.runtime = runtime_hook

  x = np.arange(32, dtype=np.float32)
  got = Tensor.empty(32, dtype=dtypes.float32).custom_kernel(Tensor(x), fxn=lds_stage_probe_kernel(32))[0].realize().numpy()
  max_abs = float(np.max(np.abs(got - x)))
  kernels = {}
  OUT.mkdir(parents=True, exist_ok=True)
  for name, lib in captured.items():
    asm = _disasm(lib)
    (OUT / f"disasm_{name}.txt").write_text(asm)
    desc = _parse_desc(lib)
    desc["hist"] = _hist(asm)
    desc["primitive_flags"] = _primitive_flags(asm)
    kernels[name] = desc
  flags = next(iter(kernels.values()), {}).get("primitive_flags", {})
  return {"ran": True, "numeric": {"max_abs": max_abs, "pass": max_abs == 0.0}, "kernels": kernels,
          "pass": bool(max_abs == 0.0 and flags.get("has_lds")),
          "barrier_status": "not_required_or_elided_in_minimal_same-lane_probe" if flags.get("has_lds") and not flags.get("has_barrier") else "present" if flags.get("has_barrier") else "absent"}


def build() -> dict[str, Any]:
  p1 = _run_json_script("extra/qk_decode_physical_tile_p1_crosslane_gate.py")
  vdot2 = _run_json_script("extra/qk_decode_attention_a3_1_vdot2_probe.py")
  lds = _lds_probe()

  p1_kernel = next(iter(p1.get("probe", {}).get("kernels", {}).values()), {})
  p1_flags = p1_kernel.get("primitive_flags", {})
  vdot2_pass = vdot2.get("verdict") == "A3_1_RENDERER_VDOT2_PROBE_PASS"
  lds_flags = next(iter(lds.get("kernels", {}).values()), {}).get("primitive_flags", {})
  checks = {
    "CrossLane.reduce_broadcast": bool(str(p1.get("verdict", "")).startswith("P1_CROSSLANE_PASS") and p1_flags.get("has_cross_lane")),
    "LaneMap.score_reuse_across_output_columns": bool(str(p1.get("verdict", "")).startswith("P1_CROSSLANE_PASS")),
    "DotLowering.v_dot2": bool(vdot2_pass),
    "TileMemory.lds_tile": bool(lds.get("numeric", {}).get("pass") and lds_flags.get("has_lds")),
  }
  missing = [k for k, v in checks.items() if not v]
  verdict = "PALL_PRIMITIVES_VISIBLE__ROUTE_INTEGRATION_NEXT" if not missing else "PALL_PRIMITIVES_PARTIAL__SOME_PROBES_MISSING"
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "candidate_id": "decode_attention_physical_tile_all_primitives_bundle",
    "verdict": verdict,
    "checks": checks,
    "missing": missing,
    "p1_crosslane": p1,
    "vdot2_probe": vdot2,
    "lds_probe": lds,
    "decision": "All missing primitive classes are independently emit/detect-visible. Next step is a single fused route candidate that uses them together, then primitive detector + W==D." if not missing else "Do not route-integrate yet; fix failed primitive probes first."
  }


def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "all_primitives_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"all-primitives-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "PALL_PRIMITIVES_VISIBLE__ROUTE_INTEGRATION_NEXT" else 1

if __name__ == "__main__":
  raise SystemExit(main())
